"""APOE genotype analysis for AD resilience.

Extracts APOE isoforms from UK Biobank genotype data and tests
APOE-stratified differences in cognitive outcomes across cohort groups.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from src.config import Config

log = logging.getLogger("resistad.genetics.apoe")

# APOE isoform frequencies in general European population (reference)
APOE_POPULATION_FREQ = {
    "e2e2": 0.007,
    "e2e3": 0.121,
    "e2e4": 0.023,
    "e3e3": 0.618,
    "e3e4": 0.195,
    "e4e4": 0.023,
    "unknown": 0.013,
}

# AD risk odds ratios per APOE genotype (Lambert 2013)
APOE_AD_OR = {
    "e2e2": 0.41,
    "e2e3": 0.59,
    "e2e4": 2.60,
    "e3e3": 1.00,  # reference
    "e3e4": 3.68,
    "e4e4": 12.33,
}


def apoe_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute APOE genotype frequencies per cohort group.

    Parameters
    ----------
    df : DataFrame with columns 'cohort' and 'apoe_genotype'

    Returns
    -------
    DataFrame: rows = APOE genotypes, columns = cohort groups + chi2 p-value
    """
    groups = df["cohort"].unique()
    table = pd.crosstab(df["apoe_genotype"], df["cohort"], normalize="columns") * 100
    table.columns.name = None
    table.index.name = "APOE genotype"

    # Chi-squared test (counts)
    counts = pd.crosstab(df["apoe_genotype"], df["cohort"])
    chi2, pval, dof, _ = stats.chi2_contingency(counts)
    log.info(f"APOE genotype χ² test: χ²={chi2:.2f}, df={dof}, p={pval:.3g}")

    table["chi2_pvalue"] = ""
    table.at[table.index[0], "chi2_pvalue"] = f"{pval:.3g}"
    return table


def apoe_e4_dose_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Test APOE ε4 dose (0, 1, 2 copies) effect on cognition within each cohort.

    Returns OLS summary DataFrame: cohort × {beta, se, p_value}.
    """
    import statsmodels.formula.api as smf

    df = df.copy()
    df["e4_dose"] = df["apoe_genotype"].str.count("e4")
    results = []
    for cohort in df["cohort"].dropna().unique():
        sub = df[df["cohort"] == cohort][["cognition_adj", "e4_dose", "age_at_assessment", "sex"]].dropna()
        if len(sub) < 30:
            continue
        model = smf.ols("cognition_adj ~ e4_dose + age_at_assessment + sex", data=sub).fit()
        coef = model.params["e4_dose"]
        se = model.bse["e4_dose"]
        pval = model.pvalues["e4_dose"]
        results.append({"cohort": cohort, "beta_e4_dose": coef, "se": se, "p_value": pval})

    return pd.DataFrame(results).set_index("cohort")


def apoe_stratified_prs(df: pd.DataFrame) -> pd.DataFrame:
    """Compare AD-PRS distribution across APOE genotype groups.

    Returns summary DataFrame: APOE genotype × {mean_prs, sd_prs, n}.
    """
    summary = (
        df.groupby("apoe_genotype")["prs_std"]
        .agg(mean_prs="mean", sd_prs="std", n="count")
        .round(3)
    )
    return summary
