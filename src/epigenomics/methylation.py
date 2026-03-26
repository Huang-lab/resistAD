"""DNA methylation analysis: differentially methylated positions (DMPs) and regions (DMRs).

Analyzes Illumina EPIC methylation array data (UK Biobank subset, ~5000 participants)
to identify epigenomic signatures of AD resilience. Tests whether Resilient individuals
have distinct methylation patterns compared to Vulnerable and Control groups.

Data source: UK Biobank EPIC array data on DNAnexus (field 20097 subset).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from src.config import Config
from src.utils import log, save_df


def run_dmp_analysis(
    beta_matrix: pd.DataFrame,
    cohort_df: pd.DataFrame,
    group_col: str = "cohort",
    contrast: tuple[str, str] = ("Resilient", "Vulnerable"),
    confounders: list[str] | None = None,
) -> pd.DataFrame:
    """Run differentially methylated position (DMP) analysis.

    For each CpG: beta ~ group_indicator + confounders (OLS).

    Parameters
    ----------
    beta_matrix : methylation beta values (samples x CpGs), indexed by projid
    cohort_df : clinical data with cohort_group
    contrast : (numerator, denominator) groups
    confounders : covariate columns

    Returns
    -------
    DataFrame with: cpg, delta_beta, se, t_stat, p_value, padj, mean_num, mean_denom
    """
    num, denom = contrast

    # Align samples
    common = beta_matrix.index.intersection(cohort_df.index)
    groups = cohort_df.loc[common, group_col]
    mask = groups.isin([num, denom])
    samples = mask[mask].index

    betas = beta_matrix.loc[samples]
    pheno = cohort_df.loc[samples]

    n_num = (pheno[group_col] == num).sum()
    n_denom = (pheno[group_col] == denom).sum()
    log.info(f"DMP analysis: {num} (n={n_num}) vs {denom} (n={n_denom}), {betas.shape[1]} CpGs")

    # Group indicator
    group_indicator = (pheno[group_col] == num).astype(int)

    # Design matrix
    design = pd.DataFrame({"group": group_indicator}, index=samples)
    if confounders:
        for cov in confounders:
            if cov in pheno.columns:
                design[cov] = pheno[cov].values

    design = design.dropna()
    betas = betas.loc[design.index]

    # Filter CpGs with low variance
    cpg_var = betas.var()
    valid_cpgs = cpg_var[cpg_var > 0.001].index
    betas = betas[valid_cpgs]
    log.info(f"Testing {len(valid_cpgs)} CpGs after variance filter")

    results = []
    X = sm.add_constant(design)

    for i, cpg in enumerate(betas.columns):
        if i % 50000 == 0 and i > 0:
            log.info(f"  Progress: {i}/{len(betas.columns)} CpGs")

        y = betas[cpg]
        valid = y.dropna().index.intersection(X.index)
        if len(valid) < 20:
            continue

        try:
            model = sm.OLS(y.loc[valid], X.loc[valid]).fit()
            results.append({
                "cpg": cpg,
                "delta_beta": model.params["group"],
                "se": model.bse["group"],
                "t_stat": model.tvalues["group"],
                "p_value": model.pvalues["group"],
                "mean_num": y.loc[valid][design.loc[valid, "group"] == 1].mean(),
                "mean_denom": y.loc[valid][design.loc[valid, "group"] == 0].mean(),
            })
        except Exception:
            continue

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        _, padj, _, _ = multipletests(results_df["p_value"], method="fdr_bh")
        results_df["padj"] = padj
        results_df = results_df.sort_values("padj")

    n_sig = (results_df["padj"] < 0.05).sum() if not results_df.empty else 0
    log.info(f"DMP results: {n_sig} significant CpGs (padj<0.05)")
    return results_df


def annotate_cpgs(
    dmp_results: pd.DataFrame,
    annotation_path: Path | None = None,
) -> pd.DataFrame:
    """Annotate CpGs with genomic location, nearest gene, CpG island context.

    If annotation_path is not provided, attempts to use the Illumina 450K
    manifest annotation.
    """
    if annotation_path and annotation_path.exists():
        annot = pd.read_csv(annotation_path, low_memory=False)
        # Standard 450K manifest columns
        cols = ["Name", "CHR", "MAPINFO", "UCSC_RefGene_Name", "UCSC_RefGene_Group",
                "Relation_to_UCSC_CpG_Island", "Regulatory_Feature_Group"]
        available = [c for c in cols if c in annot.columns]
        annot = annot[available].rename(columns={"Name": "cpg"})
        merged = dmp_results.merge(annot, on="cpg", how="left")
        log.info(f"Annotated {merged['CHR'].notna().sum()}/{len(merged)} CpGs")
        return merged

    log.warning("No CpG annotation file provided; returning unannotated results")
    return dmp_results


def run_dmr_analysis(
    dmp_results: pd.DataFrame,
    max_gap: int = 1000,
    min_cpgs: int = 3,
    p_thresh: float = 0.05,
) -> pd.DataFrame:
    """Identify differentially methylated regions (DMRs) from DMPs.

    Groups nearby significant CpGs into regions using a simple
    distance-based approach. For production use, consider bumphunter
    or DMRcate-style algorithms.

    Parameters
    ----------
    dmp_results : annotated DMP results with CHR and MAPINFO columns
    max_gap : maximum gap between CpGs to be in the same region (bp)
    min_cpgs : minimum CpGs per region
    p_thresh : p-value threshold for including CpGs
    """
    if "CHR" not in dmp_results.columns or "MAPINFO" not in dmp_results.columns:
        log.warning("Cannot compute DMRs without genomic coordinates (CHR, MAPINFO)")
        return pd.DataFrame()

    sig = dmp_results[dmp_results["p_value"] < p_thresh].copy()
    sig["MAPINFO"] = pd.to_numeric(sig["MAPINFO"], errors="coerce")
    sig = sig.dropna(subset=["CHR", "MAPINFO"])
    sig = sig.sort_values(["CHR", "MAPINFO"])

    regions = []
    current_region = None

    for _, row in sig.iterrows():
        if current_region is None:
            current_region = {
                "chr": row["CHR"], "start": row["MAPINFO"], "end": row["MAPINFO"],
                "cpgs": [row["cpg"]], "delta_betas": [row["delta_beta"]],
                "p_values": [row["p_value"]],
            }
        elif (row["CHR"] == current_region["chr"] and
              row["MAPINFO"] - current_region["end"] <= max_gap):
            current_region["end"] = row["MAPINFO"]
            current_region["cpgs"].append(row["cpg"])
            current_region["delta_betas"].append(row["delta_beta"])
            current_region["p_values"].append(row["p_value"])
        else:
            if len(current_region["cpgs"]) >= min_cpgs:
                regions.append(current_region)
            current_region = {
                "chr": row["CHR"], "start": row["MAPINFO"], "end": row["MAPINFO"],
                "cpgs": [row["cpg"]], "delta_betas": [row["delta_beta"]],
                "p_values": [row["p_value"]],
            }

    if current_region and len(current_region["cpgs"]) >= min_cpgs:
        regions.append(current_region)

    dmr_df = pd.DataFrame([
        {
            "chr": r["chr"],
            "start": int(r["start"]),
            "end": int(r["end"]),
            "width": int(r["end"] - r["start"]),
            "n_cpgs": len(r["cpgs"]),
            "mean_delta_beta": np.mean(r["delta_betas"]),
            "min_p_value": np.min(r["p_values"]),
            "cpgs": ";".join(r["cpgs"]),
        }
        for r in regions
    ])

    if not dmr_df.empty:
        dmr_df = dmr_df.sort_values("min_p_value")

    log.info(f"DMR analysis: {len(dmr_df)} regions (>={min_cpgs} CpGs, gap<={max_gap}bp)")
    return dmr_df
