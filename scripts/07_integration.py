#!/usr/bin/env python3
"""Multi-omic integration for AD resilience — UK Biobank.

Integrates available omic layers using MOFA+ and network methods:
  - Proteomics  : Olink differential abundance (required)
  - Imaging     : Brain MRI IDP results (required)
  - Methylation : EPIC DMP results (optional — only if step 06 ran)
  - Metabolomics: NMR metabolomics (optional)

Outputs:
  analysis/out/integration/mofa_factor_associations.csv
  analysis/out/integration/mofa_loadings_*.csv          (per significant factor)
  analysis/out/integration/convergent_nodes.csv
  analysis/out/integration/resilience_signature.csv
  analysis/out/integration/signature_vs_published.csv
  analysis/out/integration/multilayer_network.edgelist

Usage:
    python scripts/07_integration.py
    python scripts/07_integration.py --skip-mofa
    python scripts/07_integration.py --n-factors 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import pandas as pd

from src.config import load_config
from src.integration.mofa import interpret_factors, prepare_mofa_input, run_mofa
from src.integration.network_integration import (
    build_multilayer_network,
    detect_communities,
    identify_convergent_nodes,
)
from src.integration.signatures import compare_with_published_sets, extract_resilience_signature
from src.integration.visualization import plot_multi_omic_summary
from src.utils import ensure_dir, load_df, log, save_df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--skip-mofa", action="store_true",
        help="Skip MOFA+ integration (requires muon; memory-intensive)",
    )
    p.add_argument(
        "--n-factors", type=int, default=None,
        help="Number of MOFA+ factors (overrides config value)",
    )
    return p.parse_args()


def _load_result(path: Path, name: str) -> pd.DataFrame | None:
    """Load a CSV result file silently returning None if absent or empty."""
    if not path.exists():
        log.warning(f"  {name} not found at {path} — skipping")
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            log.warning(f"  {name} is empty (0 rows) — skipping")
            return None
        log.info(f"  Loaded {name}: {len(df)} rows")
        return df
    except Exception as exc:
        log.warning(f"  {name} could not be loaded ({exc}) — skipping")
        return None


def main() -> None:
    args = parse_args()
    cfg = load_config()
    out_base = cfg.resolve_path(cfg.output_dir)
    out = ensure_dir(out_base / "integration")
    fig_dir = ensure_dir(out / "figures")

    log.info("=== Step 07: Multi-Omic Integration ===")

    # ── Cohort file ────────────────────────────────────────────────────────
    cohort_path = out_base / "cohorts.parquet"
    if not cohort_path.exists():
        log.error("cohorts.parquet not found. Run scripts/02_define_cohorts.py first.")
        sys.exit(1)
    cohort_df = load_df(cohort_path).set_index("eid")

    # ── Load per-step results (CSV level — for network/signature) ──────────
    log.info("Loading per-step analysis results...")
    de_proteins = _load_result(
        out_base / "proteomics" / "protein_de_resilient_vs_vulnerable.csv",
        "Olink DE resilient_vs_vulnerable",
    )
    idp_results = _load_result(
        out_base / "imaging" / "idp_differences.csv",
        "MRI IDP differences",
    )
    dmps = _load_result(
        out_base / "methylation" / "dmp_resilient_vs_vulnerable.csv",
        "EPIC DMPs",
    )
    apoe_stats = _load_result(
        out_base / "genetics" / "apoe_frequency.csv",
        "APOE frequency table",
    )
    prs_stats = _load_result(
        out_base / "genetics" / "prs_cohort_stats.csv",
        "PRS cohort statistics",
    )

    # ── MOFA+ Integration ──────────────────────────────────────────────────
    mofa_results: dict = {}

    if not args.skip_mofa:
        log.info("\n--- MOFA+ Integration ---")

        # MOFA+ needs raw omic matrices (samples × features), not summary tables.
        # These intermediate files are written by steps 04 / 05 / 06.
        prot_matrix_path  = out_base / "proteomics"   / "olink_qc.parquet"
        idp_matrix_path   = out_base / "imaging"      / "idp_normalised.parquet"
        meth_matrix_path  = Path("data/raw/ukb_epic_betas.parquet")
        metab_matrix_path = Path("data/raw/ukb_nmr_metabolomics.parquet")

        prot_matrix  = (load_df(prot_matrix_path).set_index("eid")
                        if prot_matrix_path.exists() else None)
        idp_matrix   = (load_df(idp_matrix_path).set_index("eid")
                        if idp_matrix_path.exists() else None)
        meth_matrix  = (load_df(meth_matrix_path).set_index("eid")
                        if meth_matrix_path.exists() else None)
        metab_matrix = (load_df(metab_matrix_path).set_index("eid")
                        if metab_matrix_path.exists() else None)

        if prot_matrix is not None and idp_matrix is not None:
            try:
                n_factors = args.n_factors or cfg.integration.n_factors
                mdata = prepare_mofa_input(
                    proteomics_df=prot_matrix,
                    imaging_df=idp_matrix,
                    methylation_df=meth_matrix,
                    metabolomics_df=metab_matrix,
                    cohort_df=cohort_df,
                )
                mdata = run_mofa(
                    mdata,
                    n_factors=n_factors,
                    output_path=out / "mofa_model.hdf5",
                )
                mofa_results = interpret_factors(mdata, cohort_df)

                assoc = mofa_results.get("factor_associations", pd.DataFrame())
                save_df(assoc, out / "mofa_factor_associations.csv", index=False)

                for name, loading_df in mofa_results.get("top_loadings", {}).items():
                    save_df(loading_df, out / f"mofa_loadings_{name}.csv", index=False)

            except Exception as exc:
                log.warning(f"MOFA+ integration failed: {exc}")
                mofa_results = {}
        else:
            missing = []
            if prot_matrix is None:
                missing.append("olink_qc.parquet (run step 04)")
            if idp_matrix is None:
                missing.append("idp_normalised.parquet (run step 05)")
            log.warning("Skipping MOFA+: missing " + " and ".join(missing))
    else:
        log.info("MOFA+ skipped via --skip-mofa")

    # Always write a placeholder so downstream rules don't fail
    if not (out / "mofa_factor_associations.csv").exists():
        save_df(pd.DataFrame(), out / "mofa_factor_associations.csv", index=False)

    # ── Multi-layer Network ────────────────────────────────────────────────
    log.info("\n--- Multi-layer Network ---")
    # UKB has no bulk RNA-seq or eQTMs; use proteins + DMP-gene links
    G = build_multilayer_network(
        de_genes=None,
        de_proteins=de_proteins,
        dmps=dmps,
        eqtms=None,
    )

    if G.number_of_nodes() > 0:
        convergent = identify_convergent_nodes(
            G, min_layers=cfg.integration.convergence_n_layers
        )
        save_df(convergent, out / "convergent_nodes.csv", index=False)

        detect_communities(G)  # logged internally
        nx.write_edgelist(G, str(out / "multilayer_network.edgelist"))
        log.info(
            f"Network saved: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges → {out/'multilayer_network.edgelist'}"
        )
    else:
        log.info("Network is empty (no significant hits in available layers)")

    # ── Resilience Signature ───────────────────────────────────────────────
    log.info("\n--- Resilience Signature ---")
    # UKB lacks matched bulk RNA-seq; min_layers=1 so protein-only hits are kept.
    # If methylation is available, min_layers=2 gives stronger cross-omic support.
    has_meth_hits = dmps is not None and not dmps.empty and "UCSC_RefGene_Name" in dmps.columns
    min_layers = 2 if has_meth_hits else 1

    signature = extract_resilience_signature(
        de_genes=None,
        de_proteins=de_proteins,
        dmps=dmps,
        celltype_de=None,
        mofa_loadings=mofa_results.get("top_loadings"),
        min_layers=min_layers,
    )
    save_df(signature, out / "resilience_signature.csv", index=False)
    log.info(f"Resilience signature: {len(signature)} features (min_layers={min_layers})")

    if not signature.empty:
        try:
            comparison = compare_with_published_sets(signature)
            save_df(comparison, out / "signature_vs_published.csv", index=False)
        except Exception as exc:
            log.warning(f"Published-set comparison failed: {exc}")

    # ── Visualisation ──────────────────────────────────────────────────────
    log.info("\n--- Summary Figures ---")
    if not signature.empty:
        try:
            plot_multi_omic_summary(signature, fig_dir / "multi_omic_convergence.png")
        except Exception as exc:
            log.warning(f"Summary plot failed: {exc}")

    # ── Print summary ──────────────────────────────────────────────────────
    log.info("\n── Integration summary ──────────────────────────────")
    log.info(f"  Olink DE proteins tested : "
             f"{len(de_proteins) if de_proteins is not None else 'N/A'}")
    log.info(f"  IDPs tested              : "
             f"{len(idp_results) if idp_results is not None else 'N/A'}")
    log.info(f"  DMPs                     : "
             f"{len(dmps) if dmps is not None else 'N/A (methylation not available)'}")
    log.info(f"  Network nodes            : {G.number_of_nodes()}")
    log.info(f"  Network edges            : {G.number_of_edges()}")
    log.info(f"  Resilience signature     : {len(signature)} features")

    fa = mofa_results.get("factor_associations")
    if fa is not None and not fa.empty and "padj_kruskal" in fa.columns:
        n_sig = (fa["padj_kruskal"] < 0.10).sum()
        log.info(f"  Significant MOFA factors : {n_sig} (FDR<0.10)")
    elif args.skip_mofa:
        log.info("  MOFA+                    : skipped")
    else:
        log.info("  MOFA+                    : not run (missing omic matrices)")

    log.info(f"\nResults saved to: {out}")
    log.info("Step 07 complete.")


if __name__ == "__main__":
    main()
