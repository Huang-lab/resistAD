"""Rare variant burden tests for AD-associated genes using UK Biobank WES.

Uses plink2 to aggregate rare variants (MAF < 1%) in AD-associated genes
and tests burden differences between Resilient and Vulnerable groups.

NOTE: All variant file operations run on DNAnexus. Only summary statistics
(p-values, beta coefficients) are returned to Python.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from src.config import Config

log = logging.getLogger("resistad.genetics.rare_variants")

# Genomic coordinates for AD-associated genes (GRCh38)
AD_GENE_REGIONS = {
    "TREM2":  ("6",  41158506, 41163186),
    "PLCG2":  ("16", 81824526, 82026556),
    "ABI3":   ("17", 42906875, 42937989),
    "ABCA7":  ("19", 1040101,  1065571),
    "SORL1":  ("11", 121452860, 121633763),
    "APP":    ("21", 25880550,  26170613),
    "PSEN1":  ("14", 73136418,  73223691),
    "PSEN2":  ("1",  226870615, 226903753),
    "APOE":   ("19", 44905791,  44909393),
    "CLU":    ("8",  27454434,  27541090),
    "BIN1":   ("2",  127077063, 127107658),
    "CR1":    ("1",  207492058, 207854514),
}


def extract_gene_variants(
    bfile_prefix: str,
    gene: str,
    maf_threshold: float = 0.01,
    cadd_threshold: float | None = None,
    output_dir: Path = Path("data/raw/rare_variants"),
    plink2_bin: str = "plink2",
) -> Path:
    """Extract rare variants in a gene region using plink2.

    Parameters
    ----------
    bfile_prefix : plink .bed/.bim/.fam prefix (WES data on DNAnexus)
    gene : gene name (must be in AD_GENE_REGIONS)
    maf_threshold : minor allele frequency cutoff (default 0.01 = rare)
    output_dir : where to write the per-gene plink files
    plink2_bin : path to plink2 binary

    Returns
    -------
    Path to output plink prefix for this gene.
    """
    if gene not in AD_GENE_REGIONS:
        raise ValueError(f"Gene '{gene}' not in AD_GENE_REGIONS. Available: {list(AD_GENE_REGIONS)}")

    chrom, start, end = AD_GENE_REGIONS[gene]
    output_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = output_dir / gene

    cmd = [
        plink2_bin,
        "--bfile", str(bfile_prefix),
        "--chr", str(chrom),
        "--from-bp", str(start),
        "--to-bp", str(end),
        "--max-maf", str(maf_threshold),
        "--make-bed",
        "--out", str(out_prefix),
        "--memory", "4000",
    ]
    log.info(f"Extracting {gene} variants: chr{chrom}:{start}-{end}, MAF < {maf_threshold}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"plink2 failed for {gene}:\n{result.stderr}")

    return out_prefix


def compute_burden_score(
    gene_bfile: Path,
    participants: pd.Series,
    plink2_bin: str = "plink2",
) -> pd.Series:
    """Compute per-participant variant burden (count of rare alleles) for a gene.

    Returns
    -------
    Series indexed by eid, values = number of rare alleles carried.
    """
    out = gene_bfile.parent / f"{gene_bfile.name}_burden"
    # Use plink2 --score with all-ones effect sizes to count alleles
    bim = pd.read_csv(str(gene_bfile) + ".bim", sep="\t",
                      names=["CHR", "SNP", "CM", "BP", "A1", "A2"])
    score_file = gene_bfile.parent / f"{gene_bfile.name}_ones.score"
    bim[["SNP", "A1"]].assign(beta=1).to_csv(score_file, sep="\t", index=False, header=False)

    cmd = [
        plink2_bin, "--bfile", str(gene_bfile),
        "--score", str(score_file), "1", "2", "3", "sum",
        "--out", str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"plink2 burden failed:\n{result.stderr}")

    sscore = pd.read_csv(str(out) + ".sscore", sep="\t")
    score_col = [c for c in sscore.columns if "SUM" in c.upper() or "SCORE" in c.upper()]
    burden = sscore.set_index("IID")[score_col[0]].rename("burden")
    burden.index = burden.index.astype(str)
    return burden


def run_burden_tests(
    pheno_df: pd.DataFrame,
    wes_bfile_prefix: str,
    cfg: "Config",
    output_dir: Path = Path("analysis/out/rare_variants"),
    plink2_bin: str = "plink2",
) -> pd.DataFrame:
    """Run SKAT-like burden tests for all AD genes.

    Tests whether rare variant burden differs between Resilient and Vulnerable
    participants using Wilcoxon rank-sum test (non-parametric, robust to rare events).

    Parameters
    ----------
    pheno_df : cohort DataFrame with 'cohort' column
    wes_bfile_prefix : plink2 prefix for WES data on DNAnexus
    cfg : pipeline Config

    Returns
    -------
    DataFrame: gene × {n_variants, mean_burden_R, mean_burden_V, U_stat, p_value, padj}
    """
    from statsmodels.stats.multitest import multipletests

    resilient = pheno_df[pheno_df["cohort"] == "Resilient"].index.astype(str)
    vulnerable = pheno_df[pheno_df["cohort"] == "Vulnerable"].index.astype(str)

    results = []
    gene_list = cfg.rare_variants.gene_list

    for gene in gene_list:
        try:
            gene_bfile = extract_gene_variants(
                bfile_prefix=wes_bfile_prefix,
                gene=gene,
                maf_threshold=cfg.rare_variants.maf_threshold,
                output_dir=output_dir / "gene_bfiles",
                plink2_bin=plink2_bin,
            )
            burden = compute_burden_score(gene_bfile, pheno_df.index, plink2_bin)

            r_burden = burden.reindex(resilient).dropna()
            v_burden = burden.reindex(vulnerable).dropna()

            if len(r_burden) < 10 or len(v_burden) < 10:
                log.warning(f"{gene}: insufficient data (R={len(r_burden)}, V={len(v_burden)})")
                continue

            u_stat, p_val = stats.mannwhitneyu(r_burden, v_burden, alternative="two-sided")
            n_var = len(pd.read_csv(str(gene_bfile) + ".bim", sep="\t"))

            results.append({
                "gene": gene,
                "n_rare_variants": n_var,
                "mean_burden_Resilient": r_burden.mean(),
                "mean_burden_Vulnerable": v_burden.mean(),
                "U_statistic": u_stat,
                "p_value": p_val,
            })
            log.info(f"  {gene}: n_var={n_var}, R={r_burden.mean():.3f}, V={v_burden.mean():.3f}, p={p_val:.3g}")

        except Exception as exc:
            log.warning(f"  {gene}: failed — {exc}")

    df = pd.DataFrame(results)
    if len(df) > 0:
        _, padj, _, _ = multipletests(df["p_value"], method="fdr_bh")
        df["padj"] = padj
    return df.set_index("gene")
