#!/usr/bin/env python3
"""Check which UK Biobank omic layers are available in the DNAnexus project.

Outputs data/raw/data_availability.yaml which gates subsequent pipeline steps.

Usage:
    python scripts/00_check_data_availability.py
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.dnanexus_access import login, check_data_availability, whoami
from src.utils import setup_logging, ensure_dir

log = setup_logging("00_check_availability")


def main() -> None:
    cfg = load_config()

    # Authenticate
    login(token_path=cfg.dnanexus.token_path)
    log.info(f"Authenticated as: {whoami()}")

    log.info(f"Checking data availability in project: {cfg.dnanexus.project_id}")
    availability = check_data_availability(
        project_id=cfg.dnanexus.project_id,
        bulk_paths=cfg.dnanexus.bulk,
    )

    # Always-available layers (from phenotype dataset)
    availability["phenotypes"] = True
    availability["genotypes_imputed"] = True  # Imputation folder confirmed present

    # Report
    log.info("\n── Data availability ──────────────────────────────────")
    for layer, present in sorted(availability.items()):
        status = "PRESENT" if present else "NOT FOUND"
        log.info(f"  {layer:<30} {status}")

    # Write availability YAML
    out_path = cfg.resolve_path(cfg.raw_dir) / "data_availability.yaml"
    ensure_dir(out_path.parent)
    with open(out_path, "w") as f:
        yaml.dump(availability, f, default_flow_style=False)
    log.info(f"\nAvailability written to: {out_path}")

    # Warn if key layers are missing
    required = ["phenotypes", "olink"]
    missing = [k for k in required if not availability.get(k, False)]
    if missing:
        log.warning(f"Required layers not found: {missing}")
        sys.exit(1)

    log.info("Data availability check complete.")


if __name__ == "__main__":
    main()
