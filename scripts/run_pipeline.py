#!/usr/bin/env python3
"""Main orchestrator for the resistAD multi-omic resilience pipeline — UK Biobank.

Runs all analysis steps sequentially:
  0. Check data availability on DNAnexus
  1. Download UKB phenotypes & bulk omic data from DNAnexus
  2. Define resilience cohorts (APOE/PRS + cognition)
  3. Genetics    — APOE analysis, PRS computation, WES burden tests
  4. Proteomics  — Olink differential abundance, PPI network
  5. Imaging     — Brain MRI IDP analysis, brain age gap
  6. Methylation — EPIC DMP/DMR analysis (conditional on data availability)
  7. Integration — MOFA+, multi-layer network, resilience signature

Steps 3–6 are independent of each other and can be run in parallel.
Use the Snakefile for parallelised execution across multiple cores.
Step 7 requires all prior steps to have completed.

Usage:
    python scripts/run_pipeline.py                      # Run all steps (0–7)
    python scripts/run_pipeline.py --steps 2 3 4        # Run specific steps
    python scripts/run_pipeline.py --start-from 4       # Resume from step 4
    python scripts/run_pipeline.py --skip-mofa          # Skip MOFA+ in step 7
    python scripts/run_pipeline.py --skip-prs           # Use pre-computed PRS scores
    python scripts/run_pipeline.py --skip-rare          # Skip WES burden tests
    python scripts/run_pipeline.py --stop-on-error      # Halt on first failure
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import log


STEPS = {
    0: ("00_check_data_availability", "Check DNAnexus data availability"),
    1: ("01_download_data",           "Download data from DNAnexus / UK Biobank"),
    2: ("02_define_cohorts",          "Define resilience cohorts (APOE/PRS + cognition)"),
    3: ("03_genetics",                "APOE analysis, PRS computation, WES burden tests"),
    4: ("04_proteomics",              "Olink differential abundance & PPI network"),
    5: ("05_imaging",                 "Brain MRI IDP analysis & brain age gap"),
    6: ("06_methylation",             "EPIC methylation DMP/DMR analysis (conditional)"),
    7: ("07_integration",             "MOFA+ integration, network & resilience signature"),
}

# Steps that are independent (safe to run in any order after step 2)
PARALLEL_STEPS = {3, 4, 5, 6}


def run_step(step_num: int, extra_args: list[str] | None = None) -> bool:
    """Run a single pipeline step as a subprocess. Returns True on success."""
    module_name, description = STEPS[step_num]
    log.info(f"\n{'='*60}")
    log.info(f"STEP {step_num}: {description}")
    log.info(f"{'='*60}")

    script = Path(__file__).resolve().parent / f"{module_name}.py"
    cmd = [sys.executable, str(script)] + (extra_args or [])

    start = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        log.info(f"Step {step_num} completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as exc:
        elapsed = time.time() - start
        log.error(
            f"Step {step_num} FAILED after {elapsed:.1f}s "
            f"(exit code {exc.returncode})"
        )
        return False
    except Exception as exc:
        elapsed = time.time() - start
        log.error(f"Step {step_num} FAILED after {elapsed:.1f}s: {exc}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="resistAD UK Biobank multi-omic resilience pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--steps", nargs="+", type=int,
        metavar="N",
        help="Run only these step numbers (e.g., --steps 2 3 4)",
    )
    parser.add_argument(
        "--start-from", type=int, metavar="N",
        help="Start from step N and run all subsequent steps",
    )
    parser.add_argument(
        "--stop-on-error", action="store_true",
        help="Stop the pipeline if any step fails (default: continue)",
    )
    parser.add_argument(
        "--skip-mofa", action="store_true",
        help="Skip MOFA+ in step 7 (saves time / memory)",
    )
    parser.add_argument(
        "--skip-prs", action="store_true",
        help="Skip PRS computation in step 3 (requires pre-computed scores)",
    )
    parser.add_argument(
        "--skip-rare", action="store_true",
        help="Skip WES rare-variant burden tests in step 3",
    )
    args = parser.parse_args()

    # Determine steps to run
    if args.steps:
        steps_to_run = sorted(args.steps)
    elif args.start_from is not None:
        steps_to_run = list(range(args.start_from, max(STEPS.keys()) + 1))
    else:
        steps_to_run = sorted(STEPS.keys())

    # Per-step extra CLI arguments
    step_extra: dict[int, list[str]] = {}
    if args.skip_mofa:
        step_extra.setdefault(7, []).append("--skip-mofa")
    if args.skip_prs:
        step_extra.setdefault(3, []).append("--skip-prs")
    if args.skip_rare:
        step_extra.setdefault(3, []).append("--skip-rare")

    log.info("resistAD UK Biobank Pipeline")
    log.info(f"Steps to run: {steps_to_run}")
    if any(s in PARALLEL_STEPS for s in steps_to_run):
        log.info(
            "Tip: Steps 3–6 are independent. Use the Snakefile for "
            "parallel execution (snakemake --cores 4)."
        )
    log.info("=" * 60)

    total_start = time.time()
    results: dict[int, bool] = {}

    for step in steps_to_run:
        if step not in STEPS:
            log.warning(f"Unknown step {step}, skipping")
            continue

        success = run_step(step, extra_args=step_extra.get(step))
        results[step] = success

        if not success and args.stop_on_error:
            log.error(f"Stopping pipeline due to failure at step {step}")
            break

    # ── Summary ────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    log.info(f"\n{'='*60}")
    log.info("PIPELINE SUMMARY")
    log.info(f"{'='*60}")
    for step, success in results.items():
        status = "OK" if success else "FAILED"
        _, desc = STEPS[step]
        log.info(f"  Step {step:2d} ({desc}): {status}")
    log.info(f"Total time: {total_elapsed:.1f}s")

    if not all(results.values()):
        failed = [s for s, ok in results.items() if not ok]
        log.error(f"Failed steps: {failed}")
        sys.exit(1)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
