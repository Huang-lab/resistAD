"""Brain MRI imaging derived phenotype (IDP) analysis for AD resilience.

Tests which brain structural and diffusion MRI measures differ between
Resilient, Vulnerable, and Control cohort groups, and builds a
brain age gap measure as a continuous resilience biomarker.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

if TYPE_CHECKING:
    from src.config import Config

log = logging.getLogger("resistad.imaging.mri_idps")

IDP_LABELS = {
    "hippocampus_L":         "Left hippocampal volume (mm³)",
    "hippocampus_R":         "Right hippocampal volume (mm³)",
    "total_brain_vol":       "Total brain volume (mm³)",
    "wmh_volume":            "White matter hyperintensity volume (mm³)",
    "fa_corticospinal_L":    "FA — left corticospinal tract",
    "fa_corticospinal_R":    "FA — right corticospinal tract",
    "md_corticospinal_L":    "MD — left corticospinal tract",
    "md_corticospinal_R":    "MD — right corticospinal tract",
    "cortical_thickness_mean": "Mean cortical thickness (mm)",
}


# ── IDP QC & normalisation ────────────────────────────────────────────────────

def qc_idps(idp_df: pd.DataFrame, max_na_frac: float = 0.30) -> pd.DataFrame:
    """Remove IDPs with too much missing data and winsorise outliers (5 SD).

    Parameters
    ----------
    idp_df : eid × IDP DataFrame
    max_na_frac : drop IDPs with more than this fraction missing

    Returns
    -------
    Cleaned IDP DataFrame.
    """
    # Drop IDPs with too much missing
    frac_na = idp_df.isna().mean()
    keep = frac_na[frac_na <= max_na_frac].index
    n_dropped = idp_df.shape[1] - len(keep)
    if n_dropped:
        log.info(f"Dropped {n_dropped} IDPs (>{max_na_frac*100:.0f}% missing)")
    idp_df = idp_df[keep].copy()

    # Winsorise at ±5 SD
    for col in idp_df.columns:
        mu, sd = idp_df[col].mean(), idp_df[col].std()
        idp_df[col] = idp_df[col].clip(mu - 5 * sd, mu + 5 * sd)

    return idp_df


def normalise_idps(idp_df: pd.DataFrame, tbv_col: str = "total_brain_vol") -> pd.DataFrame:
    """Normalise volumetric IDPs by total brain volume.

    Volumetric measures (hippocampus, WMH) are divided by TBV.
    Diffusion and thickness measures are left unchanged.
    """
    vol_idps = ["hippocampus_L", "hippocampus_R", "wmh_volume"]
    idp_df = idp_df.copy()
    if tbv_col in idp_df.columns:
        for col in vol_idps:
            if col in idp_df.columns:
                idp_df[col] = idp_df[col] / idp_df[tbv_col]
    return idp_df


# ── Differential IDP analysis ─────────────────────────────────────────────────

def test_idp_differences(
    merged_df: pd.DataFrame,
    idp_cols: list[str],
    cfg: "Config",
    reference_group: str = "Control",
) -> pd.DataFrame:
    """Test each IDP for differences across cohort groups using OLS.

    Model: IDP ~ cohort + total_brain_vol + age_at_assessment + sex + assessment_centre

    Parameters
    ----------
    merged_df : DataFrame with cohort column and IDP columns
    idp_cols : list of IDP column names to test
    reference_group : baseline cohort for contrasts

    Returns
    -------
    DataFrame: IDP × {beta_Resilient, beta_Vulnerable, p_Resilient, p_Vulnerable, padj}
    """
    cov = cfg.imaging.idp_covariates
    available_cov = [c for c in cov if c in merged_df.columns]

    results = []
    for idp in idp_cols:
        if idp not in merged_df.columns:
            continue
        sub = merged_df[["cohort", idp] + available_cov].dropna()
        if len(sub) < 50:
            log.warning(f"  {idp}: too few obs ({len(sub)}), skipping")
            continue

        # Set reference group
        sub = sub.copy()
        sub["cohort"] = pd.Categorical(sub["cohort"], categories=["Control", "Resilient", "Vulnerable"])

        cov_str = " + ".join(available_cov) if available_cov else "1"
        formula = f"{idp} ~ C(cohort, Treatment(reference='{reference_group}')) + {cov_str}"

        try:
            model = smf.ols(formula, data=sub).fit()
            row = {"IDP": idp}
            for group in ["Resilient", "Vulnerable"]:
                key = f"C(cohort, Treatment(reference='{reference_group}'))[T.{group}]"
                if key in model.params:
                    row[f"beta_{group}"] = model.params[key]
                    row[f"se_{group}"] = model.bse[key]
                    row[f"p_{group}"] = model.pvalues[key]
            row["n"] = len(sub)
            results.append(row)
        except Exception as exc:
            log.warning(f"  {idp}: model failed — {exc}")

    df = pd.DataFrame(results)
    if df.empty:
        log.warning("No IDP results produced (all models failed or no data).")
        return pd.DataFrame(columns=["IDP", "beta_Resilient", "beta_Vulnerable"]).set_index("IDP")
    df = df.set_index("IDP")

    # FDR correction (across all IDPs)
    for group in ["Resilient", "Vulnerable"]:
        p_col = f"p_{group}"
        if p_col in df.columns:
            _, padj, _, _ = multipletests(df[p_col].fillna(1), method="fdr_bh")
            df[f"padj_{group}"] = padj

    sig_R = (df.get("padj_Resilient", pd.Series(dtype=float)) < cfg.thresholds.padj).sum()
    sig_V = (df.get("padj_Vulnerable", pd.Series(dtype=float)) < cfg.thresholds.padj).sum()
    log.info(f"Significant IDPs: {sig_R} (Resilient), {sig_V} (Vulnerable) at FDR < {cfg.thresholds.padj}")

    return df


# ── Brain age gap ─────────────────────────────────────────────────────────────

def compute_brain_age_gap(
    merged_df: pd.DataFrame,
    idp_cols: list[str],
    age_col: str = "age_at_assessment",
) -> pd.Series:
    """Estimate brain age from IDPs and compute brain age gap (BAG).

    Trains a ridge regression model predicting chronological age from IDPs
    in the Control group, then applies it to all participants.
    BAG = predicted brain age - chronological age.
    Positive BAG = brain appears older than expected.

    Returns
    -------
    Series indexed by eid with BAG values.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    available = [c for c in idp_cols if c in merged_df.columns]
    if not available or age_col not in merged_df.columns:
        raise ValueError("Insufficient IDPs or age column missing for brain age gap.")

    sub = merged_df[available + [age_col, "cohort"]].dropna()
    ctrl = sub[sub["cohort"] == "Control"]

    if len(ctrl) < 100:
        log.warning(f"Only {len(ctrl)} controls for brain age model. Results may be unreliable.")

    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV())])
    pipe.fit(ctrl[available], ctrl[age_col])

    pred_age = pd.Series(
        pipe.predict(sub[available]),
        index=sub.index,
        name="predicted_brain_age",
    )
    bag = (pred_age - sub[age_col]).rename("brain_age_gap")
    log.info(f"Brain age gap computed for {len(bag)} participants. "
             f"Mean BAG = {bag.mean():.2f} years")
    return bag


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_idp_profile(
    results_df: pd.DataFrame,
    output_path: Path,
    padj_threshold: float = 0.05,
) -> None:
    """Heatmap of standardised IDP effect sizes (beta) across cohort contrasts."""
    beta_cols = [c for c in results_df.columns if c.startswith("beta_")]
    if not beta_cols:
        log.warning("No beta columns found in IDP results.")
        return

    data = results_df[beta_cols].copy()
    data.columns = [c.replace("beta_", "") for c in data.columns]
    data.index = [IDP_LABELS.get(i, i) for i in data.index]

    fig, ax = plt.subplots(figsize=(6, max(4, len(data) * 0.4)))
    sns.heatmap(
        data, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("IDP effect sizes vs Control\n(beta from OLS)")
    ax.set_xlabel("Cohort group")
    ax.set_ylabel("Brain IDP")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"IDP profile heatmap saved: {output_path}")


def plot_brain_age_gap(
    merged_df: pd.DataFrame,
    bag_col: str = "brain_age_gap",
    output_path: Path = Path("analysis/out/imaging/brain_age_gap.png"),
) -> None:
    """Violin plot of brain age gap across cohort groups."""
    if bag_col not in merged_df.columns or "cohort" not in merged_df.columns:
        return

    palette = {"Resilient": "#2196F3", "Vulnerable": "#F44336", "Control": "#4CAF50"}
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(
        data=merged_df[merged_df["cohort"].notna()],
        x="cohort", y=bag_col,
        palette=palette, inner="box", ax=ax,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Brain age gap by cohort")
    ax.set_xlabel("")
    ax.set_ylabel("Brain age gap (predicted − chronological age, years)")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Brain age gap plot saved: {output_path}")
