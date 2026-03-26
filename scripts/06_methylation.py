#!/usr/bin/env python3
"""EPIC methylation analysis for AD resilience (UK Biobank subset).

Analyzes DNA methylation differences in the subset of ~5000 UKB participants
with EPIC array data. This is conditional on methylation data being dispensed
in the DNAnexus project.

Outputs:
  analysis/out/methylation/dmp_resilient_vs_vulnerable.csv
  analysis/out/methylation/dmr_resilient_vs_vulnerable.csv

Usage:
    python scripts/06_methylation.py
    python scripts/06_methylation.py --beta-path data/raw/ukb_epic_betas.parquet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.epigenomics.methylation import run_dmp_analysis, annotate_cpgs, run_dmr_analysis
from src.utils import setup_logging, load_df, save_df, ensure_dir

log = setup_logging("06_methylation")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--beta-path", type=Path,
        default=Path("data/raw/ukb_epic_betas.parquet"),
        help="Path to EPIC methylation beta matrix (eid x CpG)",
    )
    p.add_argument(
        "--annotation", type=Path,
        default=None,
        help="Path to EPIC manifest annotation CSV",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    out_dir = ensure_dir(cfg.resolve_path(cfg.output_dir) / "methylation")

    # ── Check availability ─────────────────────────────────────────────────
    if not args.beta_path.exists():
        log.warning(
            f"EPIC methylation file not found: {args.beta_path}\n"
            "This is expected if methylation data is not dispensed in your UKB application.\n"
            "Skipping methylation analysis."
        )
        return

    # ── Load data ──────────────────────────────────────────────────────────
    cohort_path = cfg.resolve_path(cfg.output_dir) / "cohorts.parquet"
    if not cohort_path.exists():
        log.error("Cohort file not found. Run scripts/02_define_cohorts.py first.")
        sys.exit(1)

    log.info("Loading cohort data ...")
    cohort_df = load_df(cohort_path).set_index("eid")

    log.info(f"Loading EPIC beta matrix from {args.beta_path} ...")
    beta_df = load_df(args.beta_path)
    beta_df = beta_df.set_index("eid") if "eid" in beta_df.columns else beta_df
    log.info(f"Beta matrix: {len(beta_df)} participants x {beta_df.shape[1]} CpGs")

    confounders = [c for c in cfg.cohort.confounders if c in cohort_df.columns]

    # ── Resilient vs Vulnerable ────────────────────────────────────────────
    for contrast in [("Resilient", "Vulnerable"), ("Resilient", "Control")]:
        contrast_name = f"{contrast[0].lower()}_vs_{contrast[1].lower()}"
        log.info(f"\n── DMP: {contrast_name} ──")

        dmp = run_dmp_analysis(
            beta_matrix=beta_df,
            cohort_df=cohort_df,
            group_col="cohort",
            contrast=contrast,
            confounders=confounders,
        )

        if args.annotation:
            dmp = annotate_cpgs(dmp, args.annotation)

        save_df(dmp, out_dir / f"dmp_{contrast_name}.csv", index=False)

        sig = dmp[dmp["padj"] < cfg.thresholds.methylation_padj] if not dmp.empty else dmp
        log.info(f"Significant DMPs: {len(sig)}")

        # ── DMR analysis ───────────────────────────────────────────────────
        if "CHR" in dmp.columns:
            dmr = run_dmr_analysis(dmp)
            save_df(dmr, out_dir / f"dmr_{contrast_name}.csv", index=False)
            log.info(f"DMRs identified: {len(dmr)}")

    log.info(f"\nMethylation results saved to: {out_dir}")
    log.info("Step 06 complete.")


if __name__ == "__main__":
    main()
