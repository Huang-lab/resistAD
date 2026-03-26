#!/usr/bin/env python3
"""Download / extract UK Biobank phenotype data from DNAnexus.

Extracts core phenotype fields, Olink proteomics, brain MRI IDPs, and
optional metabolomics from the dispensed UKB dataset using the
`dx extract_dataset` CLI and saves them as local Parquet files.

Works both locally (after `dx login`) and inside DNAnexus RAP JupyterLab.

Usage:
    python scripts/01_download_data.py
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.dnanexus_access import login, whoami
from src.data.ukbiobank import (
    load_hes_dementia,
    load_metabolomics,
    load_mri_idps,
    load_olink,
    load_phenotypes,
)
from src.utils import ensure_dir, save_df, setup_logging

log = setup_logging("01_download_data")


def load_availability(cfg) -> dict:
    avail_path = cfg.resolve_path(cfg.raw_dir) / "data_availability.yaml"
    if not avail_path.exists():
        log.warning(
            "data_availability.yaml not found; run script 00 first. "
            "Assuming all layers available."
        )
        return {"phenotypes": True, "olink": True, "brain_mri": True}
    with open(avail_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()
    raw_dir = ensure_dir(cfg.resolve_path(cfg.raw_dir))

    login(token_path=cfg.dnanexus.token_path)
    log.info(f"Authenticated as: {whoami()}")

    availability = load_availability(cfg)

    dataset_id = cfg.dnanexus.phenotype_dataset_id
    project_id = cfg.dnanexus.project_id

    # ── Core phenotypes ──────────────────────────────────────────────────
    if availability.get("phenotypes", True):
        pheno_path = raw_dir / "ukb_phenotypes.parquet"
        if pheno_path.exists():
            log.info(f"Skipping phenotypes — {pheno_path} already exists")
        else:
            log.info("Extracting core phenotypes ...")
            pheno = load_phenotypes(dataset_id, project_id, cfg)
            save_df(pheno.reset_index(), pheno_path)

        hes_path = raw_dir / "ukb_hes_dementia.parquet"
        if hes_path.exists():
            log.info(f"Skipping HES — {hes_path} already exists")
        else:
            log.info("Extracting HES dementia diagnoses ...")
            hes = load_hes_dementia(dataset_id, project_id, cfg)
            hes.reset_index().to_parquet(hes_path, index=False)
            log.info(f"HES: {(hes == 'dementia').sum()} dementia, {(hes == 'mci').sum()} MCI cases")

    # ── Olink proteomics ─────────────────────────────────────────────────
    olink_path = raw_dir / "ukb_olink.parquet"
    if olink_path.exists():
        log.info(f"Skipping Olink — {olink_path} already exists")
    elif availability.get("olink", False):
        log.info("Extracting Olink proteomics (this may take ~15 min) ...")
        olink = load_olink(dataset_id, project_id, cfg)
        save_df(olink.reset_index(), olink_path)
    else:
        log.info("Olink not available — skipping.")

    # ── Brain MRI IDPs ───────────────────────────────────────────────────
    idp_path = raw_dir / "ukb_mri_idps.parquet"
    if idp_path.exists():
        log.info(f"Skipping MRI IDPs — {idp_path} already exists")
    elif availability.get("brain_mri", False):
        log.info("Extracting brain MRI IDPs ...")
        idps = load_mri_idps(dataset_id, project_id, cfg)
        save_df(idps.reset_index(), idp_path)
    else:
        log.info("Brain MRI not available — skipping.")

    # ── NMR Metabolomics (optional) ──────────────────────────────────────
    metab_path = raw_dir / "ukb_metabolomics.parquet"
    if metab_path.exists():
        log.info(f"Skipping metabolomics — {metab_path} already exists")
    else:
        log.info("Attempting to extract NMR metabolomics ...")
        metab = load_metabolomics(dataset_id, project_id)
        if metab is not None:
            save_df(metab.reset_index(), metab_path)

    log.info("Data extraction complete.")
    log.info(f"Files written to: {raw_dir}")


if __name__ == "__main__":
    main()
