#!/usr/bin/env python3
"""Brain MRI IDP analysis for AD resilience.

Tests brain structural and diffusion MRI measures across cohort groups,
and computes brain age gap as a continuous resilience biomarker.

Outputs:
  analysis/out/imaging/idp_results.csv       — per-IDP OLS results
  analysis/out/imaging/idp_profile.png       — heatmap of effect sizes
  analysis/out/imaging/brain_age_gap.parquet — per-participant BAG
  analysis/out/imaging/brain_age_gap.png     — violin plot

Usage:
    python scripts/05_imaging.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.imaging.mri_idps import (
    qc_idps,
    normalise_idps,
    test_idp_differences,
    compute_brain_age_gap,
    plot_idp_profile,
    plot_brain_age_gap,
)
from src.utils import setup_logging, load_df, save_df, output_path, ensure_dir

log = setup_logging("05_imaging")


def main() -> None:
    cfg = load_config()
    raw_dir = cfg.resolve_path(cfg.raw_dir)
    out_dir = ensure_dir(cfg.resolve_path(cfg.output_dir) / "imaging")

    # ── Load data ──────────────────────────────────────────────────────────
    cohort_path = cfg.resolve_path(cfg.output_dir) / "cohorts.parquet"
    if not cohort_path.exists():
        log.error("Cohort file not found. Run scripts/02_define_cohorts.py first.")
        sys.exit(1)

    idp_path = raw_dir / "ukb_mri_idps.parquet"
    if not idp_path.exists():
        log.error("MRI IDP file not found. Run scripts/01_download_data.py first.")
        sys.exit(1)

    log.info("Loading cohort data ...")
    cohort_df = load_df(cohort_path).set_index("eid")

    log.info("Loading brain MRI IDPs ...")
    idp_df = load_df(idp_path).set_index("eid")

    # ── Filter to participants with imaging data ────────────────────────────
    # Most UKB participants (~90%) have NO imaging; drop them before QC
    import pandas as pd
    has_any_imaging = idp_df.dropna(how="all").index
    idp_df = idp_df.loc[has_any_imaging]
    log.info(f"Participants with any imaging data: {len(idp_df)}")

    # ── QC and normalisation ───────────────────────────────────────────────
    idp_df = qc_idps(idp_df)
    idp_df = normalise_idps(idp_df)

    # Merge with cohort (inner join = only participants in both)
    merged = cohort_df.join(idp_df, how="inner")
    log.info(f"Merged: {len(merged)} participants with both cohort and MRI data")

    # ── Differential IDP analysis ──────────────────────────────────────────
    idp_cols = [c for c in idp_df.columns if c in merged.columns]
    log.info(f"Testing {len(idp_cols)} IDPs ...")
    results = test_idp_differences(merged, idp_cols, cfg)

    save_df(results.reset_index(), out_dir / "idp_differences.csv")
    log.info(f"\nIDP results:\n{results.to_string()}")

    # ── Visualisations ─────────────────────────────────────────────────────
    plot_idp_profile(results, out_dir / "idp_profile.png", padj_threshold=cfg.thresholds.padj)

    # ── Brain age gap ──────────────────────────────────────────────────────
    try:
        bag = compute_brain_age_gap(merged, idp_cols)
        merged["brain_age_gap"] = bag
        save_df(
            merged[["cohort", "brain_age_gap"]].reset_index(),
            out_dir / "brain_age_gap.csv",
        )
        plot_brain_age_gap(merged, output_path=out_dir / "brain_age_gap.png")
    except Exception as exc:
        log.warning(f"Brain age gap failed: {exc}")

    log.info(f"\nImaging results saved to: {out_dir}")
    log.info("Step 05 complete.")


if __name__ == "__main__":
    main()
