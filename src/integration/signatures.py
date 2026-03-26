"""Convergent resilience signature extraction.

Aggregates findings across all omic layers to identify a core set of
genes/proteins/pathways consistently associated with resilience.
Applies meta-analysis to combine evidence across layers.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.utils import log, save_df


def extract_resilience_signature(
    de_genes: pd.DataFrame | None = None,
    de_proteins: pd.DataFrame | None = None,
    dmps: pd.DataFrame | None = None,
    celltype_de: dict[str, pd.DataFrame] | None = None,
    mofa_loadings: dict[str, pd.DataFrame] | None = None,
    padj_thresh: float = 0.05,
    min_layers: int = 2,
) -> pd.DataFrame:
    """Aggregate top hits across all omic layers into a resilience signature.

    Prioritizes genes appearing in >= min_layers and applies
    Fisher's method for meta-analysis of p-values.
    """
    gene_evidence = {}

    # Transcriptomics
    if de_genes is not None:
        sig = de_genes[de_genes["padj"] < padj_thresh]
        for _, row in sig.iterrows():
            gene = row["gene"]
            if gene not in gene_evidence:
                gene_evidence[gene] = {"layers": [], "p_values": [], "directions": []}
            gene_evidence[gene]["layers"].append("transcriptomics")
            gene_evidence[gene]["p_values"].append(row["padj"])
            gene_evidence[gene]["directions"].append("up" if row["log2FoldChange"] > 0 else "down")

    # Proteomics
    if de_proteins is not None:
        sig = de_proteins[de_proteins["padj"] < padj_thresh]
        for _, row in sig.iterrows():
            gene = row["protein"]
            if gene not in gene_evidence:
                gene_evidence[gene] = {"layers": [], "p_values": [], "directions": []}
            gene_evidence[gene]["layers"].append("proteomics")
            gene_evidence[gene]["p_values"].append(row["padj"])
            gene_evidence[gene]["directions"].append("up" if row["log2FC"] > 0 else "down")

    # Epigenomics (DMP-associated genes)
    if dmps is not None and "UCSC_RefGene_Name" in dmps.columns:
        sig = dmps[dmps["padj"] < padj_thresh]
        for _, row in sig.iterrows():
            genes = str(row.get("UCSC_RefGene_Name", "")).split(";")
            for gene in genes:
                gene = gene.strip()
                if not gene:
                    continue
                if gene not in gene_evidence:
                    gene_evidence[gene] = {"layers": [], "p_values": [], "directions": []}
                if "epigenomics" not in gene_evidence[gene]["layers"]:
                    gene_evidence[gene]["layers"].append("epigenomics")
                    gene_evidence[gene]["p_values"].append(row["padj"])
                    gene_evidence[gene]["directions"].append(
                        "hypo" if row["delta_beta"] < 0 else "hyper"
                    )

    # Cell-type-specific DE
    if celltype_de:
        for ct, de_df in celltype_de.items():
            sig = de_df[de_df["padj"] < padj_thresh]
            for _, row in sig.iterrows():
                gene = row["gene"]
                layer_name = f"singlecell_{ct}"
                if gene not in gene_evidence:
                    gene_evidence[gene] = {"layers": [], "p_values": [], "directions": []}
                gene_evidence[gene]["layers"].append(layer_name)
                gene_evidence[gene]["p_values"].append(row["padj"])
                gene_evidence[gene]["directions"].append("up" if row["log2FoldChange"] > 0 else "down")

    # Build signature table
    rows = []
    for gene, evidence in gene_evidence.items():
        unique_layers = list(set(evidence["layers"]))
        if len(unique_layers) < min_layers:
            continue

        # Fisher's combined p-value (meta-analysis)
        p_vals = [p for p in evidence["p_values"] if p > 0 and p < 1]
        if p_vals:
            fisher_stat = -2 * sum(np.log(p) for p in p_vals)
            fisher_p = stats.chi2.sf(fisher_stat, df=2 * len(p_vals))
        else:
            fisher_p = np.nan

        rows.append({
            "gene": gene,
            "n_layers": len(unique_layers),
            "layers": ";".join(sorted(unique_layers)),
            "fisher_p": fisher_p,
            "min_padj": min(evidence["p_values"]) if evidence["p_values"] else np.nan,
            "directions": ";".join(evidence["directions"]),
        })

    signature = pd.DataFrame(rows)

    if not signature.empty:
        _, fisher_padj, _, _ = multipletests(
            signature["fisher_p"].fillna(1), method="fdr_bh"
        )
        signature["fisher_padj"] = fisher_padj
        signature = signature.sort_values(["n_layers", "fisher_p"], ascending=[False, True])

    log.info(f"Resilience signature: {len(signature)} genes in >={min_layers} layers")
    return signature


def compare_with_published_sets(
    signature: pd.DataFrame,
    gene_col: str = "gene",
) -> pd.DataFrame:
    """Compare resilience signature with published gene sets.

    Tests overlap with known resilience/protection gene sets from literature.
    """
    # Published resilience gene sets (curated from key publications)
    published_sets = {
        "Mostafavi_2018_AD_modules": [
            "CLU", "APOE", "BIN1", "PICALM", "CR1", "CD33", "MS4A6A",
            "ABCA7", "EPHA1", "CD2AP", "INPP5D", "MEF2C", "HLA-DRB5",
        ],
        "Mathys_2019_microglia_DAM": [
            "TREM2", "TYROBP", "SPP1", "LPL", "CST7", "CD9",
            "ITGAX", "CLEC7A", "LILRB4", "TIMP2",
        ],
        "Perez_2022_resilience_genes": [
            "HSPA1A", "HSPA1B", "DNAJB1", "BAG3", "HSPB1",
            "SQSTM1", "MAP1LC3B", "BECN1",
        ],
        "synaptic_protection": [
            "SYT1", "SYN1", "DLG4", "SNAP25", "GRIA1", "GRIN1",
            "CAMK2A", "SLC17A7", "NRXN1", "NLGN1",
        ],
    }

    sig_genes = set(signature[gene_col].str.upper())

    results = []
    for set_name, set_genes in published_sets.items():
        set_upper = set(g.upper() for g in set_genes)
        overlap = sig_genes & set_upper

        # Fisher's exact test (background ~ 20000 genes)
        a = len(overlap)
        b = len(sig_genes - set_upper)
        c = len(set_upper - sig_genes)
        d = 20000 - a - b - c

        odds, pval = stats.fisher_exact([[a, b], [c, d]], alternative="greater")

        results.append({
            "reference_set": set_name,
            "n_reference": len(set_genes),
            "n_overlap": len(overlap),
            "overlap_genes": ", ".join(sorted(overlap)) if overlap else "",
            "odds_ratio": odds,
            "p_value": pval,
        })

    results_df = pd.DataFrame(results).sort_values("p_value")
    log.info(f"Compared with {len(published_sets)} published gene sets")
    return results_df
