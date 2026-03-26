#!/usr/bin/env python3
"""Genetics analysis: AD-PRS computation, APOE stratification, rare variant burden.

Requires plink2 to be available (on DNAnexus: use Swiss Army Knife app,
or install via: sudo apt-get install plink2).

Outputs:
  data/raw/prs_scores.parquet     — per-participant AD-PRS (used by script 02)
  analysis/out/genetics/          — all genetics analysis results

Usage:
    python scripts/03_genetics.py --bfile /path/to/imputed/chr
    python scripts/03_genetics.py --skip-prs   # APOE analysis only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.genetics.apoe import apoe_frequency_table, apoe_e4_dose_effect
from src.genetics.prs import (
    download_gwas_weights,
    prepare_score_file,
    run_plink2_score,
    load_sscore,
    prs_distribution_plot,
    prs_cohort_stats,
)
from src.utils import setup_logging, load_df, save_df, output_path, ensure_dir

log = setup_logging("03_genetics")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bfile", type=str,
        default=None,
        help="Plink .bed/.bim/.fam prefix for imputed genotypes (on DNAnexus). "
             "e.g. 'Bulk/Imputation/UKB imputation from genotype/ukb22828_c1_b0_v3'",
    )
    p.add_argument("--plink2", type=str, default="plink2", help="Path to plink2 binary")
    p.add_argument("--skip-prs", action="store_true", help="Skip PRS computation")
    p.add_argument("--skip-rare", action="store_true", help="Skip rare variant analysis")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    out_dir = ensure_dir(cfg.resolve_path(cfg.output_dir) / "genetics")
    raw_dir = cfg.resolve_path(cfg.raw_dir)

    # ── Load cohort (if available) ─────────────────────────────────────────
    cohort_path = cfg.resolve_path(cfg.output_dir) / "cohorts.parquet"
    cohort_df = None
    if cohort_path.exists():
        cohort_df = load_df(cohort_path).set_index("eid")
        log.info(f"Loaded cohort: {len(cohort_df)} participants")

    # ── AD-PRS ────────────────────────────────────────────────────────────
    if not args.skip_prs:
        if args.bfile is None:
            log.error("--bfile is required for PRS computation. Use --skip-prs to skip.")
            sys.exit(1)

        log.info("Downloading GWAS summary statistics ...")
        gwas_path = download_gwas_weights(cfg)

        log.info("Preparing PRS score file ...")
        score_file = raw_dir / "prs_score_snps.tsv"
        prepare_score_file(
            gwas_path=gwas_path,
            output_path=score_file,
            p_threshold=cfg.prs.p_threshold,
            exclude_apoe=cfg.prs.exclude_apoe_region,
            apoe_chr=cfg.prs.apoe_chr,
            apoe_start_mb=cfg.prs.apoe_start_mb,
            apoe_end_mb=cfg.prs.apoe_end_mb,
        )

        log.info("Running plink2 scoring ...")
        sscore_path = run_plink2_score(
            bfile_prefix=args.bfile,
            score_file=score_file,
            output_prefix=raw_dir / "prs_output",
            plink2_bin=args.plink2,
        )

        prs = load_sscore(sscore_path)
        save_df(prs.reset_index(), raw_dir / "prs_scores.parquet")
        log.info(f"PRS saved: {raw_dir / 'prs_scores.parquet'}")

        if cohort_df is not None:
            cohort_df["prs_std"] = prs.reindex(cohort_df.index)

    # ── APOE analysis (needs cohort) ──────────────────────────────────────
    apoe_ok = (
        cohort_df is not None
        and "apoe_genotype" in cohort_df.columns
        and cohort_df["apoe_genotype"].nunique() > 1  # not all "unknown"
    )
    if apoe_ok:
        log.info("APOE genotype frequency table ...")
        freq_table = apoe_frequency_table(cohort_df)
        save_df(freq_table.reset_index(), out_dir / "apoe_frequency.csv")
        log.info(f"\n{freq_table.to_string()}")

        log.info("APOE e4 dose effect on cognition ...")
        dose_df = apoe_e4_dose_effect(cohort_df)
        save_df(dose_df.reset_index(), out_dir / "apoe_e4_dose_cognition.csv")
    else:
        log.warning(
            "APOE genotype not available (all 'unknown'). "
            "Producing placeholder apoe_frequency.csv. "
            "Re-run after extracting APOE from genotype data on DNAnexus."
        )
        import pandas as pd
        placeholder = pd.DataFrame({
            "APOE genotype": ["unknown"],
            "note": ["APOE SNPs not in phenotype dataset. Extract from imputed genotypes on DNAnexus."],
        })
        save_df(placeholder, out_dir / "apoe_frequency.csv", index=False)

    # ── PRS distribution plots (needs cohort + PRS) ───────────────────────
    prs_ok = cohort_df is not None and "prs_std" in cohort_df.columns and cohort_df["prs_std"].std() > 0
    if prs_ok:
        prs_distribution_plot(cohort_df, out_dir / "prs_distribution.png")

        stats_df = prs_cohort_stats(cohort_df)
        save_df(stats_df, out_dir / "prs_cohort_stats.csv", index=False)
        log.info(f"\nPRS cohort statistics:\n{stats_df.to_string()}")
    else:
        log.warning(
            "PRS not computed. Producing placeholder prs_cohort_stats.csv. "
            "Run with --bfile on DNAnexus to compute AD-PRS."
        )
        import pandas as pd
        placeholder = pd.DataFrame({
            "cohort": ["Resilient", "Vulnerable", "Control"],
            "mean_prs": [float("nan")] * 3,
            "note": ["PRS not yet computed"] * 3,
        })
        save_df(placeholder, out_dir / "prs_cohort_stats.csv", index=False)

    # ── Rare variants (WES) ───────────────────────────────────────────────
    if not args.skip_rare and args.bfile is not None:
        from src.genetics.rare_variants import run_burden_tests
        log.info("Running rare variant burden tests ...")
        # Use WES bfile (different from imputed)
        wes_bfile = args.bfile.replace("Imputation", "Exome sequences")
        burden_df = run_burden_tests(
            pheno_df=cohort_df,
            wes_bfile_prefix=wes_bfile,
            cfg=cfg,
            output_dir=out_dir / "rare_variants",
            plink2_bin=args.plink2,
        )
        save_df(burden_df.reset_index(), out_dir / "rare_variant_burden.csv")
        log.info(f"Rare variant results:\n{burden_df.to_string()}")

    log.info(f"Genetics results saved to: {out_dir}")
    log.info("Step 03 complete.")


if __name__ == "__main__":
    main()
