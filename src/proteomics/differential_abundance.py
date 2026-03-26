"""Differential protein abundance analysis — Olink Explore (UK Biobank).

Uses OLS regression with confounder adjustment to identify proteins
differentially abundant between Resilient, Vulnerable, and Control groups.

Input: Olink NPX values (log2-scale, already normalised by UKB).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

from src.config import Config
from src.utils import log, save_df


def olink_qc(
    olink_df: pd.DataFrame,
    lod_frac: float = 0.50,
) -> pd.DataFrame:
    """Olink-specific QC: remove proteins above LOD in <50% of samples.

    Parameters
    ----------
    olink_df : eid × protein NPX DataFrame
    lod_frac : drop proteins with > this fraction of missing (below LOD)

    Returns
    -------
    Filtered DataFrame.
    """
    frac_missing = olink_df.isna().mean(axis=0)
    keep = frac_missing[frac_missing <= lod_frac].index
    n_dropped = olink_df.shape[1] - len(keep)
    if n_dropped:
        log.info(f"Olink LOD filter: removed {n_dropped} proteins (>{lod_frac*100:.0f}% missing)")
    return olink_df[keep]


def run_protein_de(
    protein_matrix: pd.DataFrame,
    cohort_df: pd.DataFrame,
    group_col: str = "cohort",
    contrast: tuple[str, str] = ("Resilient", "Vulnerable"),
    confounders: list[str] | None = None,
) -> pd.DataFrame:
    """Run differential protein abundance analysis.

    For each protein: abundance ~ group_indicator + confounders (OLS).

    Parameters
    ----------
    protein_matrix : Olink NPX values (samples x proteins), indexed by eid
    cohort_df : cohort data with 'cohort' column, indexed by eid
    group_col : column name for group assignment (default 'cohort')
    contrast : (numerator, denominator) groups e.g. ('Resilient', 'Vulnerable')
    confounders : covariate columns to adjust for

    Returns
    -------
    DataFrame with: protein, log2FC, se, t_stat, p_value, padj, mean_num, mean_denom
    """
    num, denom = contrast

    # Align samples
    common = protein_matrix.index.intersection(cohort_df.index)
    groups = cohort_df.loc[common, group_col]
    mask = groups.isin([num, denom])
    samples = mask[mask].index

    prot = protein_matrix.loc[samples]
    pheno = cohort_df.loc[samples]

    n_num = (pheno[group_col] == num).sum()
    n_denom = (pheno[group_col] == denom).sum()
    log.info(f"Protein DE: {num} (n={n_num}) vs {denom} (n={n_denom}), {prot.shape[1]} proteins")

    # Group indicator (1 = numerator, 0 = denominator)
    group_indicator = (pheno[group_col] == num).astype(int)

    # Build design matrix
    design_data = pd.DataFrame({"group": group_indicator}, index=samples)
    if confounders:
        for cov in confounders:
            if cov in pheno.columns:
                design_data[cov] = pheno[cov].values

    design_data = design_data.dropna()
    prot = prot.loc[design_data.index]

    results = []
    for protein in prot.columns:
        y = prot[protein].dropna()
        valid = y.index.intersection(design_data.index)
        if len(valid) < 20:
            continue

        X = sm.add_constant(design_data.loc[valid])
        try:
            model = sm.OLS(y.loc[valid], X).fit()
            results.append({
                "protein": protein,
                "log2FC": model.params["group"],
                "se": model.bse["group"],
                "t_stat": model.tvalues["group"],
                "p_value": model.pvalues["group"],
                "mean_num": y.loc[valid][design_data.loc[valid, "group"] == 1].mean(),
                "mean_denom": y.loc[valid][design_data.loc[valid, "group"] == 0].mean(),
                "n": len(valid),
            })
        except Exception:
            continue

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        _, padj, _, _ = multipletests(results_df["p_value"], method="fdr_bh")
        results_df["padj"] = padj
        results_df = results_df.sort_values("padj")

    n_sig = (results_df["padj"] < 0.05).sum() if not results_df.empty else 0
    log.info(f"Protein DE: {n_sig} significant proteins (padj<0.05)")
    return results_df


def run_all_protein_contrasts(
    protein_matrix: pd.DataFrame,
    cohort_df: pd.DataFrame,
    config: Config,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Run protein DE for all resilience contrasts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    confounders = [c for c in config.cohort.confounders if c in cohort_df.columns]
    log.info(f"Confounders in model: {confounders}")

    contrasts = [
        ("resilient_vs_vulnerable", "Resilient", "Vulnerable"),
        ("resilient_vs_control", "Resilient", "Control"),
    ]

    results = {}
    for name, num, denom in contrasts:
        log.info(f"--- Protein contrast: {name} ---")
        res = run_protein_de(
            protein_matrix, cohort_df,
            contrast=(num, denom),
            confounders=confounders,
        )
        results[name] = res
        save_df(res, output_dir / f"protein_de_{name}.csv", index=False)

    return results
