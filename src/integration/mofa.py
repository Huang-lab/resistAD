"""Multi-Omics Factor Analysis (MOFA+) integration — UK Biobank layers.

Uses muon's MOFA+ interface to perform unsupervised integration of
UK Biobank omic layers:
  - proteomics : Olink Explore NPX values
  - imaging    : Brain MRI IDPs (hippocampus, WMH, diffusion, cortical thickness)
  - methylation: EPIC array beta values (subset, optional)
  - metabolomics: NMR metabolites (optional)

Identifies latent factors capturing cross-omic variation and tests their
association with the AD resilience phenotype.
"""

from pathlib import Path

import muon as mu
import numpy as np
import pandas as pd
from scipy import stats

from src.config import Config
from src.utils import log, save_df


def prepare_mofa_input(
    proteomics_df: pd.DataFrame,
    imaging_df: pd.DataFrame,
    methylation_df: pd.DataFrame | None = None,
    metabolomics_df: pd.DataFrame | None = None,
    cohort_df: pd.DataFrame | None = None,
    n_top_features: int = 2000,
) -> mu.MuData:
    """Prepare multi-omic MuData object for MOFA+.

    Aligns samples across omic layers by eid and creates a MuData container.
    Proteomics and imaging are required. Methylation and metabolomics are added
    if sufficient overlapping samples exist.

    Parameters
    ----------
    proteomics_df : Olink NPX values (samples x proteins)
    imaging_df : brain MRI IDPs (samples x IDPs)
    methylation_df : EPIC beta values (samples x CpGs), optional
    metabolomics_df : NMR metabolites (samples x metabolites), optional
    cohort_df : cohort data with 'cohort' column for obs annotation
    n_top_features : max features per omic layer (most variable)
    """
    import anndata as ad

    # Required layers: proteomics + imaging
    common = proteomics_df.index.intersection(imaging_df.index)
    if len(common) < 50:
        raise ValueError(
            f"Only {len(common)} participants overlap between proteomics and imaging. "
            "Check that both DataFrames are indexed by eid."
        )

    # Optionally expand with methylation and metabolomics
    for name, df in [("methylation", methylation_df), ("metabolomics", metabolomics_df)]:
        if df is not None:
            overlap = common.intersection(df.index)
            if len(overlap) >= 100:
                common = overlap
                log.info(f"Added {name} layer ({len(overlap)} participants)")
            else:
                log.warning(f"Skipping {name}: only {len(overlap)} overlapping participants")

    log.info(f"MOFA input: {len(common)} shared participants across layers")

    def top_var(df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Select top-n most variable features."""
        variance = df.loc[common].var()
        return df.loc[common, variance.nlargest(min(n, len(variance))).index]

    def make_adata(df: pd.DataFrame, n: int) -> ad.AnnData:
        sub = top_var(df, n)
        return ad.AnnData(
            X=sub.values,
            obs=pd.DataFrame(index=common),
            var=pd.DataFrame(index=sub.columns),
        )

    modalities: dict[str, ad.AnnData] = {
        "proteomics": make_adata(proteomics_df, n_top_features),
        "imaging":    make_adata(imaging_df, min(n_top_features, 50)),  # IDPs are few
    }

    if methylation_df is not None and len(common.intersection(methylation_df.index)) >= 100:
        modalities["methylation"] = make_adata(methylation_df, min(n_top_features, 10000))

    if metabolomics_df is not None and len(common.intersection(metabolomics_df.index)) >= 100:
        modalities["metabolomics"] = make_adata(metabolomics_df, n_top_features)

    mdata = mu.MuData(modalities)

    # Annotate with cohort metadata
    if cohort_df is not None:
        meta_cols = ["cohort", "apoe_genotype", "apoe_e4", "prs_std",
                     "cognition_adj", "age_at_assessment", "sex",
                     "resilience_index"]
        for col in meta_cols:
            if col in cohort_df.columns:
                mdata.obs[col] = mdata.obs_names.map(cohort_df[col].to_dict())

    log.info(
        "MuData created: "
        + ", ".join(f"{k}={v.shape[1]} features" for k, v in modalities.items())
    )
    return mdata


def run_mofa(
    mdata: mu.MuData,
    n_factors: int = 15,
    seed: int = 42,
    output_path: Path | None = None,
) -> mu.MuData:
    """Run MOFA+ on multi-omic MuData.

    Parameters
    ----------
    mdata : MuData from prepare_mofa_input
    n_factors : number of latent factors
    seed : random seed
    output_path : path to save MOFA model (.hdf5)

    Returns
    -------
    MuData with MOFA results stored in .obsm and .uns
    """
    log.info(f"Running MOFA+ with {n_factors} factors...")

    mu.tl.mofa(
        mdata,
        n_factors=n_factors,
        seed=seed,
        outfile=str(output_path) if output_path else None,
        gpu_mode=False,
    )

    log.info("MOFA+ complete")

    # Variance explained
    if "mofa" in mdata.uns:
        r2 = mdata.uns["mofa"]["variance_explained"]
        log.info(f"Variance explained per view: {r2}")

    return mdata


def interpret_factors(
    mdata: mu.MuData,
    cohort_df: pd.DataFrame,
    group_col: str = "cohort",
    n_top_features: int = 20,
) -> dict:
    """Interpret MOFA factors in the context of resilience.

    Tests which factors discriminate resilient from AD, and extracts
    top feature loadings for informative factors.

    Returns
    -------
    dict with:
        'factor_associations': DataFrame of factor-group associations
        'top_loadings': dict of {factor: DataFrame} with top feature loadings
    """
    # Get factor values
    if "X_mofa" not in mdata.obsm:
        log.warning("MOFA factors not found in mdata.obsm")
        return {}

    factors = pd.DataFrame(
        mdata.obsm["X_mofa"],
        index=mdata.obs_names,
        columns=[f"Factor{i+1}" for i in range(mdata.obsm["X_mofa"].shape[1])],
    )

    # Test each factor across groups
    common = factors.index.intersection(cohort_df.index)
    factor_vals = factors.loc[common]
    groups = cohort_df.loc[common, group_col]

    associations = []
    for factor in factor_vals.columns:
        resilient = factor_vals.loc[groups == "Resilient", factor].dropna()
        ad = factor_vals.loc[groups == "Vulnerable", factor].dropna()
        control = factor_vals.loc[groups == "Control", factor].dropna()

        if len(resilient) < 3 or len(ad) < 3:
            continue

        # Resilient vs AD
        stat_ra, p_ra = stats.mannwhitneyu(resilient, ad, alternative="two-sided")
        # Resilient vs Control
        if len(control) >= 3:
            stat_rc, p_rc = stats.mannwhitneyu(resilient, control, alternative="two-sided")
        else:
            stat_rc, p_rc = np.nan, np.nan

        # ANOVA across all three groups
        group_list = [resilient, ad]
        if len(control) >= 3:
            group_list.append(control)
        _, p_anova = stats.kruskal(*group_list)

        associations.append({
            "factor": factor,
            "mean_Resilient": resilient.mean(),
            "mean_Vulnerable": ad.mean(),
            "mean_Control": control.mean() if len(control) >= 3 else np.nan,
            "p_Resilient_vs_Vulnerable": p_ra,
            "p_Resilient_vs_Control": p_rc,
            "p_kruskal": p_anova,
        })

    assoc_df = pd.DataFrame(associations)
    if not assoc_df.empty:
        from statsmodels.stats.multitest import multipletests
        _, padj, _, _ = multipletests(assoc_df["p_kruskal"], method="fdr_bh")
        assoc_df["padj_kruskal"] = padj

    # Get top loadings for significant factors
    top_loadings = {}
    sig_factors = assoc_df[assoc_df.get("padj_kruskal", pd.Series(dtype=float)) < 0.1]["factor"].tolist()

    for factor in sig_factors:
        factor_idx = int(factor.replace("Factor", "")) - 1
        for mod_name, mod in mdata.mod.items():
            if "LFs" in mod.varm:
                loadings = pd.Series(
                    mod.varm["LFs"][:, factor_idx],
                    index=mod.var_names,
                    name=f"{mod_name}_{factor}",
                )
                top = loadings.abs().nlargest(n_top_features)
                top_loadings[f"{mod_name}_{factor}"] = pd.DataFrame({
                    "feature": top.index,
                    "loading": loadings.loc[top.index].values,
                    "abs_loading": top.values,
                    "view": mod_name,
                })

    return {
        "factor_associations": assoc_df,
        "top_loadings": top_loadings,
    }
