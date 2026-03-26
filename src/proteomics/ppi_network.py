"""Protein-protein interaction (PPI) network analysis.

Builds PPI subnetworks from differentially abundant proteins using STRING
and identifies hub/bottleneck proteins that may be key resilience mediators.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.utils import log, save_df


def fetch_string_interactions(
    proteins: list[str],
    species: int = 9606,  # Human
    score_threshold: int = 700,
) -> pd.DataFrame:
    """Fetch PPI from STRING database via its API.

    Parameters
    ----------
    proteins : list of gene/protein names
    species : NCBI taxonomy ID (9606 = Homo sapiens)
    score_threshold : minimum combined score (0-1000)

    Returns
    -------
    DataFrame with: protein1, protein2, combined_score
    """
    import urllib.request
    import json

    # STRING API endpoint
    base_url = "https://string-db.org/api/json/network"
    proteins_str = "%0d".join(proteins[:2000])  # STRING limits to 2000
    url = f"{base_url}?identifiers={proteins_str}&species={species}&required_score={score_threshold}"

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        log.warning(f"STRING API request failed: {e}")
        return pd.DataFrame(columns=["protein1", "protein2", "combined_score"])

    if not data:
        log.warning("No interactions returned from STRING")
        return pd.DataFrame(columns=["protein1", "protein2", "combined_score"])

    interactions = pd.DataFrame([
        {
            "protein1": d["preferredName_A"],
            "protein2": d["preferredName_B"],
            "combined_score": d["score"],
        }
        for d in data
    ])

    log.info(f"STRING: {len(interactions)} interactions for {len(proteins)} proteins (score>={score_threshold})")
    return interactions


def build_ppi_network(
    de_results: pd.DataFrame,
    gene_col: str = "protein",
    padj_col: str = "padj",
    padj_thresh: float = 0.05,
    score_threshold: int = 700,
) -> nx.Graph:
    """Build PPI network from DE proteins.

    Parameters
    ----------
    de_results : protein DE results
    score_threshold : STRING combined score threshold

    Returns
    -------
    NetworkX graph with protein nodes and PPI edges.
    Node attributes include DE stats (log2FC, padj).
    """
    sig = de_results[de_results[padj_col] < padj_thresh]
    proteins = sig[gene_col].tolist()

    if not proteins:
        log.warning("No significant proteins for PPI network")
        return nx.Graph()

    log.info(f"Building PPI network for {len(proteins)} significant proteins")
    interactions = fetch_string_interactions(proteins, score_threshold=score_threshold)

    G = nx.Graph()

    # Add nodes with DE attributes
    for _, row in sig.iterrows():
        G.add_node(row[gene_col], log2FC=row.get("log2FC", 0), padj=row.get(padj_col, 1))

    # Add edges
    for _, row in interactions.iterrows():
        if row["protein1"] in G.nodes and row["protein2"] in G.nodes:
            G.add_edge(row["protein1"], row["protein2"], weight=row["combined_score"])

    # Remove isolated nodes
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    log.info(f"PPI network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
             f"({len(isolates)} isolated nodes removed)")
    return G


def identify_hub_proteins(G: nx.Graph, top_n: int = 20) -> pd.DataFrame:
    """Identify hub and bottleneck proteins in the PPI network.

    Computes degree centrality, betweenness centrality, and closeness centrality
    to find proteins that are structurally important in the resilience network.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)

    hub_df = pd.DataFrame({
        "protein": list(G.nodes),
        "degree": [G.degree(n) for n in G.nodes],
        "degree_centrality": [degree[n] for n in G.nodes],
        "betweenness_centrality": [betweenness[n] for n in G.nodes],
        "closeness_centrality": [closeness[n] for n in G.nodes],
        "log2FC": [G.nodes[n].get("log2FC", 0) for n in G.nodes],
        "padj": [G.nodes[n].get("padj", 1) for n in G.nodes],
    })

    # Composite hub score (normalized rank average)
    for col in ["degree_centrality", "betweenness_centrality", "closeness_centrality"]:
        hub_df[f"{col}_rank"] = hub_df[col].rank(ascending=False)

    rank_cols = [c for c in hub_df.columns if c.endswith("_rank")]
    hub_df["hub_score"] = hub_df[rank_cols].mean(axis=1)
    hub_df = hub_df.drop(columns=rank_cols).sort_values("hub_score")

    log.info(f"Top hub proteins: {hub_df.head(5)['protein'].tolist()}")
    return hub_df.head(top_n)


def network_communities(G: nx.Graph) -> dict[str, int]:
    """Detect communities in the PPI network using Louvain method."""
    if G.number_of_nodes() == 0:
        return {}

    communities = nx.community.louvain_communities(G, seed=42)
    node_to_community = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_community[node] = i

    log.info(f"PPI communities: {len(communities)} detected")
    return node_to_community
