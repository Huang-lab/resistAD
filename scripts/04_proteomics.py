#!/usr/bin/env python3
"""Olink proteomics differential abundance analysis.

Runs OLS-based differential protein abundance for:
  Resilient vs Vulnerable (primary contrast)
  Resilient vs Control    (secondary contrast)

Also runs STRING PPI network analysis on significant proteins.

Outputs:
  analysis/out/proteomics/protein_de_resilient_vs_vulnerable.csv
  analysis/out/proteomics/protein_de_resilient_vs_control.csv
  analysis/out/proteomics/ppi_network.graphml

Usage:
    python scripts/04_proteomics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.proteomics.differential_abundance import run_all_protein_contrasts, olink_qc
from src.proteomics.ppi_network import (
    fetch_string_interactions,
    build_ppi_network,
    identify_hub_proteins,
    network_communities,
)
from src.utils import setup_logging, load_df, save_df, output_path, ensure_dir

log = setup_logging("04_proteomics")


def main() -> None:
    cfg = load_config()
    raw_dir = cfg.resolve_path(cfg.raw_dir)
    out_dir = ensure_dir(cfg.resolve_path(cfg.output_dir) / "proteomics")

    # ── Load data ──────────────────────────────────────────────────────────
    cohort_path = cfg.resolve_path(cfg.output_dir) / "cohorts.parquet"
    if not cohort_path.exists():
        log.error("Cohort file not found. Run scripts/02_define_cohorts.py first.")
        sys.exit(1)

    olink_path = raw_dir / "ukb_olink.parquet"
    if not olink_path.exists():
        log.error("Olink file not found. Run scripts/01_download_data.py first.")
        sys.exit(1)

    log.info("Loading cohort data ...")
    cohort_df = load_df(cohort_path).set_index("eid")

    log.info("Loading Olink NPX data ...")
    olink_df = load_df(olink_path).set_index("eid")

    # ── QC ─────────────────────────────────────────────────────────────────
    olink_df = olink_qc(olink_df, lod_frac=0.50)
    log.info(f"Olink after QC: {olink_df.shape[1]} proteins × {len(olink_df)} participants")

    # ── Differential abundance ─────────────────────────────────────────────
    log.info("Running differential protein abundance ...")
    results = run_all_protein_contrasts(olink_df, cohort_df, cfg, out_dir)

    # Print top hits
    for contrast_name, res in results.items():
        if res.empty:
            continue
        sig = res[res["padj"] < cfg.thresholds.padj]
        log.info(f"\n{contrast_name}: {len(sig)} significant proteins (FDR < {cfg.thresholds.padj})")
        if not sig.empty:
            log.info(sig[["protein", "log2FC", "padj"]].head(10).to_string())

    # ── PPI network (significant proteins from primary contrast) ──────────
    de_primary = results.get("resilient_vs_vulnerable")
    if de_primary is not None and not de_primary.empty:
        sig_proteins = de_primary[
            de_primary["padj"] < cfg.thresholds.padj
        ]["protein"].tolist()

        if sig_proteins:
            log.info(f"\nBuilding PPI network for {len(sig_proteins)} significant proteins ...")
            try:
                interactions = fetch_string_interactions(
                    sig_proteins,
                    score_threshold=cfg.thresholds.ppi_score_threshold,
                )
                if not interactions.empty:
                    G = build_ppi_network(interactions)
                    hubs = identify_hub_proteins(G)
                    communities = network_communities(G)

                    save_df(hubs.reset_index(), out_dir / "ppi_hub_proteins.csv")
                    save_df(communities.reset_index(), out_dir / "ppi_communities.csv")

                    import networkx as nx
                    nx.write_graphml(G, str(out_dir / "ppi_network.graphml"))
                    log.info(f"PPI network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            except Exception as exc:
                log.warning(f"PPI analysis failed: {exc}")

    log.info(f"\nProteomics results saved to: {out_dir}")
    log.info("Step 04 complete.")


if __name__ == "__main__":
    main()
