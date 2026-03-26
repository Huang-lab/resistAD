"""Multi-layer network integration across omic layers.

Builds an integrated network connecting genes, proteins, CpG sites,
and transcription factors through different edge types (expression
correlation, PPI, eQTM, regulatory). Identifies convergent nodes
that appear across multiple omic layers.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.utils import log, save_df


def build_multilayer_network(
    de_genes: pd.DataFrame | None = None,
    de_proteins: pd.DataFrame | None = None,
    dmps: pd.DataFrame | None = None,
    eqtms: pd.DataFrame | None = None,
    regulons: dict[str, list[str]] | None = None,
    ppi_network: nx.Graph | None = None,
    padj_thresh: float = 0.05,
) -> nx.Graph:
    """Build integrated multi-layer network.

    Node types: gene, protein, cpg, tf
    Edge types: de_gene, de_protein, ppi, eqtm, regulatory

    Parameters
    ----------
    de_genes : transcriptomic DE results
    de_proteins : proteomic DE results
    dmps : differentially methylated positions
    eqtms : eQTM associations
    regulons : TF regulons from pySCENIC
    ppi_network : protein-protein interaction network

    Returns
    -------
    NetworkX graph with typed nodes and edges.
    """
    G = nx.Graph()

    # Add DE genes as nodes
    if de_genes is not None:
        sig_genes = de_genes[de_genes["padj"] < padj_thresh]
        for _, row in sig_genes.iterrows():
            G.add_node(
                row["gene"],
                node_type="gene",
                log2FC=row["log2FoldChange"],
                padj=row["padj"],
                layers={"transcriptomics"},
            )
        log.info(f"Added {len(sig_genes)} DE genes")

    # Add DE proteins
    if de_proteins is not None:
        sig_prots = de_proteins[de_proteins["padj"] < padj_thresh]
        for _, row in sig_prots.iterrows():
            name = row["protein"]
            if name in G.nodes:
                G.nodes[name]["layers"].add("proteomics")
                G.nodes[name]["protein_log2FC"] = row["log2FC"]
            else:
                G.add_node(
                    name,
                    node_type="protein",
                    log2FC=row["log2FC"],
                    padj=row["padj"],
                    layers={"proteomics"},
                )
        log.info(f"Added {len(sig_prots)} DE proteins")

    # Add PPI edges
    if ppi_network is not None:
        for u, v, data in ppi_network.edges(data=True):
            if u in G.nodes and v in G.nodes:
                G.add_edge(u, v, edge_type="ppi", weight=data.get("weight", 1))
        log.info(f"Added PPI edges: {sum(1 for _, _, d in G.edges(data=True) if d.get('edge_type') == 'ppi')}")

    # Add DMP-gene associations
    if dmps is not None and "UCSC_RefGene_Name" in dmps.columns:
        sig_dmps = dmps[dmps["padj"] < padj_thresh].head(1000)  # Top DMPs
        for _, row in sig_dmps.iterrows():
            genes = str(row["UCSC_RefGene_Name"]).split(";")
            cpg = row["cpg"]
            G.add_node(cpg, node_type="cpg", delta_beta=row["delta_beta"], layers={"epigenomics"})
            for gene in genes:
                gene = gene.strip()
                if gene and gene in G.nodes:
                    G.add_edge(cpg, gene, edge_type="methylation")
                    G.nodes[gene]["layers"].add("epigenomics")

    # Add eQTM edges
    if eqtms is not None:
        sig_eqtms = eqtms[eqtms["padj"] < padj_thresh]
        for _, row in sig_eqtms.iterrows():
            if row["gene"] in G.nodes:
                cpg = row["cpg"]
                if cpg not in G.nodes:
                    G.add_node(cpg, node_type="cpg", layers={"epigenomics"})
                G.add_edge(cpg, row["gene"], edge_type="eqtm", rho=row["rho"])

    # Add TF regulatory edges
    if regulons:
        for tf_name, targets in regulons.items():
            tf = tf_name.split("(")[0].strip()  # Clean regulon name
            if tf not in G.nodes:
                G.add_node(tf, node_type="tf", layers={"regulatory"})
            else:
                G.nodes[tf]["layers"].add("regulatory")

            for target in targets:
                if target in G.nodes:
                    G.add_edge(tf, target, edge_type="regulatory")

    log.info(f"Multi-layer network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Summarize edge types
    edge_types = {}
    for _, _, d in G.edges(data=True):
        et = d.get("edge_type", "unknown")
        edge_types[et] = edge_types.get(et, 0) + 1
    log.info(f"Edge types: {edge_types}")

    return G


def identify_convergent_nodes(
    G: nx.Graph,
    min_layers: int = 2,
) -> pd.DataFrame:
    """Identify genes/proteins appearing across multiple omic layers.

    These convergent nodes are high-confidence resilience candidates
    because they show consistent signal across independent data types.
    """
    convergent = []
    for node, data in G.nodes(data=True):
        layers = data.get("layers", set())
        if len(layers) >= min_layers:
            convergent.append({
                "node": node,
                "node_type": data.get("node_type", "unknown"),
                "n_layers": len(layers),
                "layers": ";".join(sorted(layers)),
                "degree": G.degree(node),
                "log2FC": data.get("log2FC", np.nan),
                "padj": data.get("padj", np.nan),
            })

    conv_df = pd.DataFrame(convergent).sort_values("n_layers", ascending=False)
    log.info(f"Convergent nodes (>={min_layers} layers): {len(conv_df)}")
    return conv_df


def detect_communities(G: nx.Graph) -> dict:
    """Detect communities in the integrated network.

    Returns dict mapping node -> community_id.
    """
    if G.number_of_nodes() == 0:
        return {}

    communities = nx.community.louvain_communities(G, seed=42)
    mapping = {}
    for i, comm in enumerate(communities):
        for node in comm:
            mapping[node] = i

    log.info(f"Network communities: {len(communities)} detected")
    return mapping
