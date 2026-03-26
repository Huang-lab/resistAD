"""AD Polygenic Risk Score (PRS) computation and analysis.

Computes AD-PRS from UK Biobank imputed genotypes using published
GWAS summary statistics (Bellenguez 2022, Nature Genetics).

Pipeline:
  1. Download GWAS summary stats (public FTP)
  2. QC and clump variants (plink2 on DNAnexus)
  3. Score participants (plink2 --score)
  4. Analyse PRS distribution across cohort groups

NOTE: plink2 is run inside the DNAnexus environment via subprocess.
The actual genotype files never leave DNAnexus.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

if TYPE_CHECKING:
    from src.config import Config

log = logging.getLogger("resistad.genetics.prs")


# ── GWAS weight download ──────────────────────────────────────────────────────

def download_gwas_weights(cfg: "Config") -> Path:
    """Download Bellenguez 2022 AD GWAS summary statistics.

    These are public-access files (not individual-level data) and can
    be downloaded to the local working directory.

    Returns
    -------
    Path to downloaded / existing weights file.
    """
    import urllib.request

    weights_path = Path(cfg.prs.gwas_weights_file)
    if weights_path.exists():
        log.info(f"Using cached GWAS weights: {weights_path}")
        return weights_path

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    url = cfg.prs.gwas_weights_url

    log.info(f"Downloading GWAS summary stats from {url} ...")
    try:
        urllib.request.urlretrieve(url, weights_path)
        log.info(f"Downloaded to {weights_path}")
    except Exception as exc:
        raise RuntimeError(
            f"Could not download GWAS weights from {url}. "
            "Please download manually and place at: " + str(weights_path)
        ) from exc

    return weights_path


def prepare_score_file(
    gwas_path: Path,
    output_path: Path,
    p_threshold: float = 5e-8,
    exclude_apoe: bool = True,
    apoe_chr: int = 19,
    apoe_start_mb: float = 44.4,
    apoe_end_mb: float = 46.5,
) -> Path:
    """Extract genome-wide significant SNPs for PRS scoring.

    Filters GWAS summary stats to:
      - p < p_threshold (default: genome-wide significant)
      - Excludes APOE region (chr19:44.4-46.5 Mb) if exclude_apoe=True

    Output format for plink2 --score: SNP_ID, effect_allele, beta

    Returns
    -------
    Path to plink2-ready score file.
    """
    log.info(f"Reading GWAS summary stats: {gwas_path}")
    df = pd.read_csv(gwas_path, sep="\t", compression="infer")

    # Standardise column names (Bellenguez 2022 format)
    col_map = {
        "variant_id": "SNP", "snp": "SNP", "rsid": "SNP",
        "effect_allele": "A1", "a1": "A1",
        "beta": "BETA", "effect_size": "BETA",
        "p_value": "P", "pval": "P", "p": "P",
        "chr": "CHR", "chromosome": "CHR",
        "bp": "BP", "pos": "BP", "position": "BP",
    }
    df = df.rename(columns={c: col_map[c.lower()] for c in df.columns if c.lower() in col_map})

    required = {"SNP", "A1", "BETA", "P"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"GWAS file missing columns: {missing}. Columns found: {list(df.columns)}")

    # P-value filter
    df = df[df["P"].astype(float) < p_threshold].copy()
    log.info(f"After p < {p_threshold}: {len(df)} SNPs")

    # APOE region exclusion
    if exclude_apoe and "CHR" in df.columns and "BP" in df.columns:
        mask = (
            (df["CHR"].astype(str) == str(apoe_chr)) &
            (df["BP"].astype(float) >= apoe_start_mb * 1e6) &
            (df["BP"].astype(float) <= apoe_end_mb * 1e6)
        )
        n_excl = mask.sum()
        df = df[~mask]
        log.info(f"Excluded {n_excl} SNPs in APOE region (chr{apoe_chr}:{apoe_start_mb}-{apoe_end_mb} Mb)")

    score_df = df[["SNP", "A1", "BETA"]].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(output_path, sep="\t", index=False, header=False)
    log.info(f"Score file written: {output_path} ({len(score_df)} SNPs)")
    return output_path


# ── plink2 scoring (runs on DNAnexus) ────────────────────────────────────────

def run_plink2_score(
    bfile_prefix: str,
    score_file: Path,
    output_prefix: Path,
    plink2_bin: str = "plink2",
) -> Path:
    """Run plink2 --score to compute per-participant PRS.

    This should be run inside the DNAnexus environment where genotype
    files are accessible. The output .sscore file contains PRS per participant.

    Parameters
    ----------
    bfile_prefix : path prefix to plink .bed/.bim/.fam files (on DNAnexus)
    score_file : path to SNP score file (3 columns: SNP, A1, BETA)
    output_prefix : output path prefix for plink2 results
    plink2_bin : path to plink2 binary (available via Swiss Army Knife on DNAnexus)

    Returns
    -------
    Path to the .sscore output file.
    """
    cmd = [
        plink2_bin,
        "--bfile", str(bfile_prefix),
        "--score", str(score_file), "1", "2", "3", "no-mean-imputation",
        "--out", str(output_prefix),
        "--memory", "8000",
    ]
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"plink2 failed:\n{result.stderr}")

    sscore_path = Path(str(output_prefix) + ".sscore")
    log.info(f"PRS computed: {sscore_path}")
    return sscore_path


def load_sscore(sscore_path: Path) -> pd.Series:
    """Load plink2 .sscore output and return standardised PRS Series.

    Returns
    -------
    Series indexed by IID (eid string), values = standardised PRS.
    """
    df = pd.read_csv(sscore_path, sep="\t")
    # plink2 .sscore columns: #FID, IID, ALLELE_CT, NAMED_ALLELE_DOSAGE_SUM, SCORE1_AVG
    score_col = [c for c in df.columns if "SCORE" in c.upper()]
    if not score_col:
        raise ValueError(f"No SCORE column in {sscore_path}. Columns: {list(df.columns)}")

    df["eid"] = df["IID"].astype(str)
    prs = df.set_index("eid")[score_col[0]]
    prs_std = (prs - prs.mean()) / prs.std()
    prs_std.name = "prs_score"
    log.info(f"Loaded PRS: {len(prs_std)} participants, mean={prs_std.mean():.3f}, sd={prs_std.std():.3f}")
    return prs_std


# ── PRS analysis ──────────────────────────────────────────────────────────────

def prs_distribution_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Plot PRS distribution stratified by cohort and APOE genotype."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: PRS by cohort
    palette = {"Resilient": "#2196F3", "Vulnerable": "#F44336", "Control": "#4CAF50"}
    sns.violinplot(
        data=df[df["cohort"].notna()],
        x="cohort", y="prs_std",
        palette=palette, inner="box", ax=axes[0],
    )
    axes[0].set_title("AD-PRS by cohort")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("AD-PRS (standardised)")

    # Panel B: PRS by APOE genotype
    apoe_order = ["e2e3", "e3e3", "e2e4", "e3e4", "e4e4"]
    present = [g for g in apoe_order if g in df["apoe_genotype"].values]
    sns.boxplot(
        data=df[df["apoe_genotype"].isin(present)],
        x="apoe_genotype", y="prs_std",
        order=present, ax=axes[1],
    )
    axes[1].set_title("AD-PRS by APOE genotype")
    axes[1].set_xlabel("APOE genotype")
    axes[1].set_ylabel("AD-PRS (standardised)")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"PRS distribution plot saved: {output_path}")


def prs_cohort_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Test PRS differences across cohort groups (ANOVA + post-hoc t-tests).

    Returns
    -------
    DataFrame with group means and pairwise p-values.
    """
    groups = ["Resilient", "Vulnerable", "Control"]
    present = [g for g in groups if g in df["cohort"].values]
    grp_data = {g: df[df["cohort"] == g]["prs_std"].dropna() for g in present}

    # One-way ANOVA
    f_stat, p_anova = stats.f_oneway(*grp_data.values())
    log.info(f"PRS ANOVA: F={f_stat:.2f}, p={p_anova:.3g}")

    rows = []
    for i, g1 in enumerate(present):
        for g2 in present[i+1:]:
            t, p = stats.ttest_ind(grp_data[g1], grp_data[g2])
            rows.append({
                "comparison": f"{g1} vs {g2}",
                f"mean_{g1}": f"{grp_data[g1].mean():.3f}",
                f"mean_{g2}": f"{grp_data[g2].mean():.3f}",
                "t_stat": f"{t:.3f}",
                "p_value": f"{p:.3g}",
            })

    return pd.DataFrame(rows)
