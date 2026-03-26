"""Resilience cohort definition for UK Biobank.

Defines three groups based on genetic AD risk and cognitive status:
  - Resilient : high AD genetic risk + cognitively intact at age >= 60
  - Vulnerable : high AD genetic risk + dementia diagnosis
  - Control   : low AD genetic risk + cognitively intact

Resilience index (continuous) = residual cognitive score after
removing effects of age and AD-PRS.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

if TYPE_CHECKING:
    from src.config import Config

log = logging.getLogger("resistad.cohort")

GROUPS = ["Resilient", "Vulnerable", "Control"]


# ── APOE genotype ────────────────────────────────────────────────────────────

def extract_apoe_genotype(df: pd.DataFrame) -> pd.Series:
    """Derive APOE isoform (e2/e3/e4) from rs429358 and rs7412 genotype calls.

    UKB field 22527 = rs429358 (C allele defines e4)
    UKB field 22528 = rs7412   (T allele defines e2)

    Returns
    -------
    Series with values like 'e3e3', 'e3e4', 'e2e3', 'e4e4' indexed same as df.
    """
    r429 = df.get("apoe_rs429358", pd.Series("T/T", index=df.index)).fillna("T/T")
    r7412 = df.get("apoe_rs7412", pd.Series("C/C", index=df.index)).fillna("C/C")

    def _classify(rs429: str, rs7412: str) -> str:
        e4_count = str(rs429).upper().count("C")
        e2_count = str(rs7412).upper().count("T")
        e3_count = 2 - e4_count - e2_count
        alleles = sorted(
            ["e4"] * e4_count + ["e2"] * e2_count + ["e3"] * max(e3_count, 0)
        )
        return "".join(alleles) if len(alleles) == 2 else "unknown"

    return pd.Series(
        [_classify(a, b) for a, b in zip(r429, r7412)],
        index=df.index,
        name="apoe_genotype",
    )


def is_apoe_e4_carrier(apoe_series: pd.Series) -> pd.Series:
    """Return boolean mask: True if participant carries at least one e4 allele."""
    return apoe_series.str.contains("e4", na=False)


# ── Cognitive score ───────────────────────────────────────────────────────────

def build_composite_cognition(df: pd.DataFrame, cfg: "Config") -> pd.Series:
    """Compute an age-adjusted composite cognitive score.

    Uses fluid intelligence (field 20016) across available instances.

    Returns
    -------
    Series of age-adjusted composite scores, indexed same as df.
    """
    fi_cols = [c for c in df.columns if c.startswith("fluid_intel_i")]
    if not fi_cols:
        raise ValueError("No fluid intelligence columns found. Check phenotype loading.")

    fi_mean = df[fi_cols].mean(axis=1)
    age = df.get("age_at_assessment", df.get("age_i0"))

    tmp = pd.DataFrame({"fi": fi_mean, "age": age}).dropna()
    if len(tmp) < 100:
        log.warning("Fewer than 100 participants for age-adjustment; check data.")

    model = smf.ols("fi ~ age", data=tmp).fit()
    resid = pd.Series(np.nan, index=df.index)
    resid[tmp.index] = model.resid
    return resid.rename("cognition_adj")


# ── PRS loading ───────────────────────────────────────────────────────────────

def load_prs(prs_path: str) -> pd.Series:
    """Load pre-computed AD polygenic risk scores.

    Expects a two-column file: eid, prs_score
    (output of scripts/03_genetics.py).

    Returns
    -------
    Series indexed by eid (string).
    """
    prs_path = str(prs_path)
    if prs_path.endswith(".parquet"):
        df = pd.read_parquet(prs_path)
        df["eid"] = df["eid"].astype(str) if "eid" in df.columns else df.index.astype(str)
    else:
        df = pd.read_csv(prs_path, dtype={"eid": str})
    if "prs_score" not in df.columns:
        score_col = [c for c in df.columns if "SCORE" in c.upper()]
        if not score_col:
            raise ValueError(f"Cannot find PRS score column in {prs_path}")
        df = df.rename(columns={score_col[0]: "prs_score"})
        id_col = "IID" if "IID" in df.columns else df.columns[0]
        df = df.rename(columns={id_col: "eid"})

    df["eid"] = df["eid"].astype(str)
    return df.set_index("eid")["prs_score"]


# ── Cohort assignment ─────────────────────────────────────────────────────────

def assign_cohort(
    pheno: pd.DataFrame,
    prs: pd.Series,
    hes_dx: pd.Series,
    cfg: "Config",
) -> pd.DataFrame:
    """Assign participants to Resilient / Vulnerable / Control groups.

    Parameters
    ----------
    pheno : phenotype DataFrame (from ukbiobank.load_phenotypes)
    prs : AD-PRS Series indexed by eid
    hes_dx : HES diagnosis Series ('dementia', 'mci', or NaN) indexed by eid
    cfg : pipeline Config

    Returns
    -------
    pheno DataFrame with added columns:
        apoe_genotype, apoe_e4, prs_std, prs_percentile,
        cognition_adj, high_risk, dementia, cohort,
        resilience_index (continuous)
    """
    df = pheno.copy()

    # ── APOE genotype ────────────────────────────────────────────────────
    has_apoe = "apoe_rs429358" in df.columns and "apoe_rs7412" in df.columns
    if has_apoe:
        df["apoe_genotype"] = extract_apoe_genotype(df)
        df["apoe_e4"] = is_apoe_e4_carrier(df["apoe_genotype"])
    else:
        log.warning(
            "APOE genotype columns not found in phenotypes. "
            "APOE e4 status will be unknown (set to False) until "
            "genotype extraction in step 03."
        )
        df["apoe_genotype"] = "unknown"
        df["apoe_e4"] = False

    # ── PRS ───────────────────────────────────────────────────────────────
    df["prs_raw"] = prs.reindex(df.index)
    has_real_prs = df["prs_raw"].nunique() > 1  # True if PRS has real variation
    if has_real_prs:
        df["prs_std"] = (df["prs_raw"] - df["prs_raw"].mean()) / df["prs_raw"].std()
        df["prs_percentile"] = df["prs_raw"].rank(pct=True)
    else:
        log.warning("PRS scores are constant (placeholder). Using dementia-only grouping.")
        df["prs_std"] = 0.0
        df["prs_percentile"] = 0.5

    df["hes_dx"] = hes_dx.reindex(df.index)
    df["dementia"] = df["hes_dx"] == "dementia"

    df["cognition_adj"] = build_composite_cognition(df, cfg)

    # Age filter
    df = df[df["age_at_assessment"] >= cfg.cohort.min_age].copy()
    log.info(f"After age >= {cfg.cohort.min_age}: {len(df)} participants")

    # Cognitive intact flag
    cog_thresh = df["cognition_adj"].quantile(cfg.cohort.cognition_resilient_percentile)
    df["cog_intact"] = (df["cognition_adj"] >= cog_thresh) & (~df["dementia"])

    # ── Assign cohort ─────────────────────────────────────────────────────
    if has_real_prs or has_apoe:
        # Full genetic risk stratification
        df["high_risk"] = (
            (df["prs_percentile"] >= cfg.cohort.prs_risk_percentile) |
            df["apoe_e4"]
        )
        df["low_risk"] = (
            ~df["apoe_e4"] &
            (df["prs_percentile"] <= cfg.cohort.prs_control_max_percentile)
        )
        conditions = [
            df["high_risk"] & df["cog_intact"],
            df["high_risk"] & df["dementia"],
            df["low_risk"] & df["cog_intact"],
        ]
    else:
        # Fallback: no genetic risk data → group by dementia + cognition only.
        # Re-run step 02 after step 03 provides PRS for proper stratification.
        log.info("Using fallback grouping (no genetic risk data):")
        log.info("  Resilient  = no dementia + top 25% cognition")
        log.info("  Vulnerable = dementia diagnosis")
        log.info("  Control    = no dementia + lower cognition")
        df["high_risk"] = False
        df["low_risk"] = False
        conditions = [
            df["cog_intact"] & (~df["dementia"]),
            df["dementia"],
            (~df["cog_intact"]) & (~df["dementia"]),
        ]

    df["cohort"] = np.select(conditions, GROUPS, default="")
    df = df[df["cohort"].isin(GROUPS)].copy()

    counts = df["cohort"].value_counts()
    for g in GROUPS:
        log.info(f"  {g:<12}: {counts.get(g, 0):>6} participants")

    if cfg.cohort.compute_resilience_index:
        df["resilience_index"] = compute_resilience_index(df)

    return df


def compute_resilience_index(df: pd.DataFrame) -> pd.Series:
    """Continuous resilience index: cognition residuals after removing PRS + age."""
    sub = df[["cognition_adj", "prs_std", "age_at_assessment"]].dropna()
    if len(sub) < 10:
        log.warning("Too few participants for resilience index. Returning NaN.")
        return pd.Series(np.nan, index=df.index, name="resilience_index")
    model = smf.ols("cognition_adj ~ prs_std + age_at_assessment", data=sub).fit()
    idx = pd.Series(np.nan, index=df.index)
    idx[sub.index] = model.resid
    return idx.rename("resilience_index")


# ── Summary statistics ────────────────────────────────────────────────────────

def cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate demographic balance table across cohort groups."""
    groups = GROUPS
    rows = []

    numeric_vars = [
        ("Age", "age_at_assessment"),
        ("Fluid intelligence (adj.)", "cognition_adj"),
        ("AD-PRS (std.)", "prs_std"),
    ]
    cat_vars = [
        ("Sex (% female)", "sex"),
        ("APOE e4 carrier (%)", "apoe_e4"),
    ]

    for label, col in numeric_vars:
        if col not in df.columns:
            continue
        grp_data = [df[df["cohort"] == g][col].dropna() for g in groups]
        means = [f"{x.mean():.2f} +/- {x.std():.2f}" for x in grp_data]
        f_stat, p_val = stats.f_oneway(*grp_data)
        rows.append({"variable": label, **dict(zip(groups, means)), "p_value": f"{p_val:.3g}"})

    for label, col in cat_vars:
        if col not in df.columns:
            continue
        grp_data = []
        for g in groups:
            sub = df[df["cohort"] == g][col].dropna()
            pct = sub.mean() * 100 if col == "apoe_e4" else (sub == 0).mean() * 100
            grp_data.append(f"{pct:.1f}%")
        contingency = pd.crosstab(df["cohort"], df[col])
        chi2, p_val, _, _ = stats.chi2_contingency(contingency)
        rows.append({"variable": label, **dict(zip(groups, grp_data)), "p_value": f"{p_val:.3g}"})

    return pd.DataFrame(rows).set_index("variable")
