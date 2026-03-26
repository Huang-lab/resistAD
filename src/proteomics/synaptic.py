"""Synaptic proteome characterization using SynGO gene sets.

Tests whether differentially abundant proteins in resilient brains
are enriched in specific synaptic compartments (presynaptic, postsynaptic,
synaptic vesicle, etc.) — a key question for understanding how resilient
individuals maintain synaptic function despite pathology.
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import log


# Core SynGO categories (manually curated from SynGO ontology)
SYNGO_CATEGORIES = {
    "presynapse": [
        "SYT1", "SYN1", "SYN2", "SNAP25", "STX1A", "VAMP2", "SYP",
        "CPLX1", "CPLX2", "RAB3A", "MUNC13-1", "BSN", "PCLO",
        "SV2A", "SV2B", "SYNJ1", "DNM1", "AP2A1", "AP2B1",
    ],
    "postsynapse": [
        "DLG4", "DLG1", "DLG2", "DLG3", "SHANK1", "SHANK2", "SHANK3",
        "HOMER1", "HOMER2", "GRIA1", "GRIA2", "GRIN1", "GRIN2A", "GRIN2B",
        "CAMK2A", "CAMK2B", "SYNGAP1", "NLGN1", "NLGN2",
    ],
    "synaptic_vesicle": [
        "SYP", "SYT1", "VAMP2", "SV2A", "SV2B", "RAB3A",
        "VGLUT1", "VGLUT2", "VGAT", "SLC17A7", "SLC17A6", "SLC32A1",
    ],
    "active_zone": [
        "BSN", "PCLO", "RIM1", "RIMS1", "RIMS2", "MUNC13-1",
        "UNC13A", "UNC13B", "ELKS1", "ERC1", "ERC2", "LIPRIN",
    ],
    "postsynaptic_density": [
        "DLG4", "DLG1", "SHANK1", "SHANK2", "SHANK3",
        "HOMER1", "SYNGAP1", "CAMK2A", "SAPAP1", "DLGAP1",
    ],
}


def load_syngo_genes(syngo_path: str | None = None) -> dict[str, list[str]]:
    """Load SynGO gene sets.

    If syngo_path is provided, reads from a downloaded SynGO export file.
    Otherwise, uses the built-in curated gene lists above.
    """
    if syngo_path:
        try:
            df = pd.read_csv(syngo_path, sep="\t")
            categories = {}
            for cat in df["category"].unique():
                categories[cat] = df.loc[df["category"] == cat, "gene"].tolist()
            log.info(f"Loaded SynGO: {len(categories)} categories from file")
            return categories
        except Exception as e:
            log.warning(f"Could not load SynGO file: {e}; using built-in categories")

    log.info(f"Using built-in SynGO categories: {list(SYNGO_CATEGORIES.keys())}")
    return SYNGO_CATEGORIES


def test_synaptic_enrichment(
    de_results: pd.DataFrame,
    syngo_genes: dict[str, list[str]] | None = None,
    gene_col: str = "protein",
    padj_col: str = "padj",
    padj_thresh: float = 0.05,
    direction: str = "both",
    lfc_col: str = "log2FC",
) -> pd.DataFrame:
    """Test enrichment of DE proteins in synaptic compartments.

    Uses Fisher's exact test for each SynGO category.

    Parameters
    ----------
    de_results : protein DE results
    syngo_genes : dict of {category: [genes]}
    direction : 'up', 'down', or 'both' — which DE proteins to test

    Returns
    -------
    DataFrame with: category, n_overlap, n_category, n_de, n_total,
                    odds_ratio, p_value, overlap_genes
    """
    if syngo_genes is None:
        syngo_genes = load_syngo_genes()

    # Define DE gene set
    sig = de_results[de_results[padj_col] < padj_thresh]
    if direction == "up":
        sig = sig[sig[lfc_col] > 0]
    elif direction == "down":
        sig = sig[sig[lfc_col] < 0]

    de_genes = set(sig[gene_col].str.upper())
    all_genes = set(de_results[gene_col].str.upper())

    results = []
    for category, cat_genes in syngo_genes.items():
        cat_set = set(g.upper() for g in cat_genes)
        cat_in_bg = cat_set & all_genes

        if not cat_in_bg:
            continue

        # 2x2 contingency table
        overlap = de_genes & cat_in_bg
        a = len(overlap)                          # DE and in category
        b = len(de_genes - cat_in_bg)             # DE but not in category
        c = len(cat_in_bg - de_genes)             # In category but not DE
        d = len(all_genes - de_genes - cat_in_bg) # Neither

        odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative="greater")

        results.append({
            "category": category,
            "n_overlap": a,
            "n_category_in_data": len(cat_in_bg),
            "n_de": len(de_genes),
            "n_total": len(all_genes),
            "odds_ratio": odds_ratio,
            "p_value": p_value,
            "overlap_genes": ", ".join(sorted(overlap)) if overlap else "",
        })

    results_df = pd.DataFrame(results).sort_values("p_value")
    n_sig = (results_df["p_value"] < 0.05).sum()
    log.info(f"Synaptic enrichment: {n_sig}/{len(results_df)} categories enriched (p<0.05)")
    return results_df
