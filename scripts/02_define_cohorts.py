#!/usr/bin/env python3
"""Define AD resilience cohorts from UK Biobank phenotype + PRS data.

Loads phenotypes, APOE genotypes, HES dementia diagnoses, and AD-PRS,
then assigns participants to Resilient / Vulnerable / Control groups.

Outputs:
  analysis/out/cohorts.parquet  — full cohort DataFrame
  analysis/out/cohort_summary.csv — demographic balance table

Usage:
    python scripts/02_define_cohorts.py
    python scripts/02_define_cohorts.py --prs data/raw/prs_scores.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.cohort import assign_cohort, cohort_summary, load_prs
from src.utils import setup_logging, load_df, save_df, output_path

log = setup_logging("02_define_cohorts")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prs", type=Path,
        default=Path("data/raw/prs_scores.parquet"),
        help="Path to pre-computed PRS scores (eid, prs_score). "
             "Run scripts/03_genetics.py first to generate this file.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    raw_dir = cfg.resolve_path(cfg.raw_dir)

    # ── Load phenotypes ────────────────────────────────────────────────────
    pheno_path = raw_dir / "ukb_phenotypes.parquet"
    if not pheno_path.exists():
        log.error(f"Phenotype file not found: {pheno_path}. Run script 01 first.")
        sys.exit(1)

    log.info(f"Loading phenotypes from {pheno_path}")
    pheno = load_df(pheno_path)
    pheno = pheno.set_index("eid") if "eid" in pheno.columns else pheno
    log.info(f"Phenotypes: {len(pheno)} participants")

    # ── Load HES dementia diagnoses ────────────────────────────────────────
    hes_path = raw_dir / "ukb_hes_dementia.parquet"
    if hes_path.exists():
        import pandas as pd
        hes_df = pd.read_parquet(hes_path)
        hes_df["eid"] = hes_df["eid"].astype(str)
        # Column may be named hes_diagnosis or hes_dx depending on extraction
        dx_col = "hes_diagnosis" if "hes_diagnosis" in hes_df.columns else "hes_dx"
        hes = hes_df.set_index("eid")[dx_col]
        log.info(f"HES: {(hes == 'dementia').sum()} dementia, {(hes == 'mci').sum()} MCI")
    else:
        import pandas as pd
        log.warning("HES dementia file not found. All participants will be treated as no dementia.")
        hes = pd.Series(dtype=str, name="hes_diagnosis")

    # ── Load AD-PRS ────────────────────────────────────────────────────────
    if args.prs.exists():
        log.info(f"Loading PRS from {args.prs}")
        prs = load_prs(str(args.prs))
    else:
        log.warning(
            f"PRS file not found: {args.prs}. "
            "Run scripts/03_genetics.py first. "
            "Defining cohorts using APOE only."
        )
        import pandas as pd
        prs = pd.Series(0.5, index=pheno.index, name="prs_score")

    # ── Assign cohorts ─────────────────────────────────────────────────────
    log.info("Assigning cohort groups ...")
    cohort_df = assign_cohort(pheno, prs, hes, cfg)
    log.info(f"Final cohort: {len(cohort_df)} participants assigned")

    # ── Save outputs ───────────────────────────────────────────────────────
    out_path = output_path(cfg, "cohorts.parquet")
    save_df(cohort_df.reset_index(), out_path)

    summary = cohort_summary(cohort_df)
    summary_path = output_path(cfg, "cohort_summary.csv")
    save_df(summary.reset_index(), summary_path)

    log.info("\n── Cohort summary ──────────────────────────────────")
    log.info(f"\n{summary.to_string()}")
    log.info(f"\nCohort file: {out_path}")
    log.info("Step 02 complete.")


if __name__ == "__main__":
    main()
