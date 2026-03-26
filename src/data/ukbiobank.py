"""UK Biobank data loaders for the resistAD pipeline.

Loads phenotype, Olink proteomics, brain imaging IDPs, HES diagnoses,
and optional metabolomics from the UK Biobank dispensed dataset on DNAnexus.

Field naming conventions (dx extract_dataset):
  Non-instanced : participant.p{field}            (e.g. p31 = Sex)
  Instanced     : participant.p{field}_i{inst}    (e.g. p21003_i0 = Age baseline)
  Array-only    : participant.p{field}_a{idx}     (e.g. p22009_a1 = PC1)  [1-based]
  Both          : participant.p{field}_i{inst}_a{idx}

Separate entities:
  olink_instance_0.{gene}   — Olink NPX values (protein gene symbols)
  hesin_diag.diag_icd10     — Hospital Episode Statistics ICD-10 codes
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.config import Config

log = logging.getLogger("resistad.data.ukbiobank")

# ── Field name helpers ────────────────────────────────────────────────────────

def _p(
    field_id: int,
    instance: int | None = 0,
    array: int | None = None,
) -> str:
    """Format a UKB field into the dx extract_dataset field-name convention.

    Parameters
    ----------
    field_id : UKB field number
    instance : assessment instance (0–3); None for non-instanced fields
    array    : array index (1-based for array-only fields); None if not array

    Examples
    --------
    >>> _p(31)                       # Sex (non-instanced)
    'p31'
    >>> _p(21003, instance=0)        # Age at baseline
    'p21003_i0'
    >>> _p(22009, instance=None, array=1)  # PC1 (array-only)
    'p22009_a1'
    >>> _p(399, instance=0, array=1) # Pairs matching (instanced + array)
    'p399_i0_a1'
    """
    name = f"p{field_id}"
    if instance is not None:
        name += f"_i{instance}"
    if array is not None:
        name += f"_a{array}"
    return name


# ── Core phenotype loader ─────────────────────────────────────────────────────

def load_phenotypes(dataset_id: str, project_id: str, cfg: "Config") -> pd.DataFrame:
    """Load core phenotype fields for all participants.

    Parameters
    ----------
    dataset_id : UKB dispensed dataset record ID
    project_id : DNAnexus project ID
    cfg : pipeline Config

    Returns
    -------
    DataFrame indexed by eid (string).
    """
    f = cfg.fields
    non_instanced = set(f.get("non_instanced", [31, 22000, 22006, 22009]))

    fields = ["eid"]

    # ── Demographics ──────────────────────────────────────────────────────
    # Sex (non-instanced)
    fields.append(_p(f["sex"], instance=None))

    # Age at assessment, assessment centre (instanced 0–3)
    for fid in [f["age_at_assessment"], f["assessment_centre"]]:
        fields += [_p(fid, i) for i in range(4)]

    # Education, ethnicity (instanced 0–3)
    for fid in [f["education"], f["ethnicity"]]:
        fields += [_p(fid, i) for i in range(4)]

    # Genotype batch (non-instanced)
    fields.append(_p(f["genotype_batch"], instance=None))

    # Genetic ethnic grouping (non-instanced)
    fields.append(_p(f["genetic_ethnic_grouping"], instance=None))

    # ── Cognition (instanced 0–3) ────────────────────────────────────────
    for fid_key in ["fluid_intelligence", "reaction_time", "numeric_memory",
                    "trail_making_b", "symbol_digit"]:
        fid = f.get(fid_key)
        if fid:
            fields += [_p(fid, i) for i in range(4)]

    # Pairs matching is instanced + array (p399_i{N}_a{M}); skip for now,
    # use only the simpler cognitive tests.

    # ── Genetic PCs 1–10 (array-only: p22009_a1 .. p22009_a10) ──────────
    pc_fid = f.get("genetic_pcs", 22009)
    fields += [_p(pc_fid, instance=None, array=i) for i in range(1, 11)]

    # ── Extract ──────────────────────────────────────────────────────────
    from src.data.dnanexus_access import extract_fields
    raw = extract_fields(dataset_id, fields, project_id, entity="participant")

    # ── Rename to friendly column names ──────────────────────────────────
    rename = {
        _p(f["sex"], instance=None): "sex",
        _p(f["genotype_batch"], instance=None): "genotype_batch",
        _p(f["genetic_ethnic_grouping"], instance=None): "genetic_ethnic_grouping",
    }
    for i in range(4):
        rename[_p(f["age_at_assessment"], i)] = f"age_i{i}"
        rename[_p(f["assessment_centre"], i)] = f"centre_i{i}"
        rename[_p(f["education"], i)] = f"education_i{i}"
        rename[_p(f["ethnicity"], i)] = f"ethnicity_i{i}"
        fi = f.get("fluid_intelligence")
        if fi:
            rename[_p(fi, i)] = f"fluid_intel_i{i}"
        rt = f.get("reaction_time")
        if rt:
            rename[_p(rt, i)] = f"reaction_time_i{i}"
    for i in range(1, 11):
        rename[_p(pc_fid, instance=None, array=i)] = f"pc{i}"

    raw = raw.rename(columns=rename)

    # Primary convenience columns
    if "age_i0" in raw.columns:
        raw["age_at_assessment"] = raw["age_i0"]
    if "centre_i0" in raw.columns:
        raw["assessment_centre"] = raw["centre_i0"]
    if "education_i0" in raw.columns:
        raw["education"] = raw["education_i0"]
    if "ethnicity_i0" in raw.columns:
        raw["ethnicity"] = raw["ethnicity_i0"]

    raw["eid"] = raw["eid"].astype(str)
    log.info(f"Loaded phenotypes: {len(raw)} participants × {raw.shape[1]} columns")
    return raw.set_index("eid")


# ── HES dementia diagnoses ───────────────────────────────────────────────────

def load_hes_dementia(dataset_id: str, project_id: str, cfg: "Config") -> pd.Series:
    """Extract dementia/MCI flags from participant-level ICD-10 field.

    Uses participant.p41270 (Diagnoses - ICD10) which stores a JSON array
    of ICD-10 codes per participant. This avoids extracting the massive
    hesin_diag entity.

    Returns
    -------
    Series indexed by eid with values: 'dementia', 'mci', or NaN
    """
    import json as _json

    dementia_codes = set(cfg.fields.get("dementia_codes", ["F00", "F01", "F03", "G30", "G31"]))
    mci_codes = set(cfg.fields.get("mci_codes", ["F06.7", "R41.3"]))

    from src.data.dnanexus_access import extract_fields

    raw = extract_fields(
        dataset_id,
        field_names=["eid", "p41270"],
        project_id=project_id,
        entity="participant",
    )

    def classify(icd_str) -> str | None:
        """Classify a JSON array string of ICD-10 codes."""
        if pd.isna(icd_str) or icd_str == "":
            return None
        try:
            codes = _json.loads(icd_str) if isinstance(icd_str, str) else []
        except (_json.JSONDecodeError, TypeError):
            return None

        codes_3 = {str(c)[:3] for c in codes}
        codes_full = {str(c).strip() for c in codes}

        if codes_3 & dementia_codes:
            return "dementia"
        if codes_full & mci_codes or codes_3 & {"F06", "R41"}:
            return "mci"
        return None

    raw["hes_diagnosis"] = raw["p41270"].apply(classify)
    raw["eid"] = raw["eid"].astype(str)
    result = raw.set_index("eid")["hes_diagnosis"]

    n_dementia = (result == "dementia").sum()
    n_mci = (result == "mci").sum()
    log.info(f"HES diagnoses: {n_dementia} dementia, {n_mci} MCI out of {len(result)} participants")
    return result


# ── Olink proteomics ──────────────────────────────────────────────────────────

def load_olink(dataset_id: str, project_id: str, cfg: "Config") -> pd.DataFrame:
    """Load Olink Explore 3072 plasma proteomics (olink_instance_0 entity).

    Returns
    -------
    DataFrame (eid × protein) with NPX log2 values.
    """
    from src.data.dnanexus_access import list_fields, extract_fields

    # Discover all protein fields in the olink entity
    olink_fields_df = list_fields(dataset_id, project_id, entity="olink_instance_0")
    protein_fields = [
        f for f in olink_fields_df["field_name"].tolist()
        if not f.startswith("olink_instance_0.eid")  # skip eid from field list
    ]
    # Strip entity prefix for the field_names argument
    protein_names = [f.split(".", 1)[-1] for f in protein_fields]

    log.info(f"Discovered {len(protein_names)} Olink protein fields")

    # Extract eid + all protein fields
    raw = extract_fields(
        dataset_id,
        field_names=["eid"] + protein_names,
        project_id=project_id,
        entity="olink_instance_0",
    )

    raw["eid"] = raw["eid"].astype(str)
    raw = raw.set_index("eid")

    # LOD filter: drop proteins with >50% missing
    if cfg.olink.lod_filter:
        n_before = raw.shape[1]
        frac_missing = raw.isna().mean(axis=0)
        keep = frac_missing[frac_missing <= 0.50].index
        raw = raw[keep]
        log.info(f"Olink LOD filter: removed {n_before - len(keep)} proteins (>50% missing)")

    log.info(f"Loaded Olink: {len(raw)} participants × {raw.shape[1]} proteins")
    return raw


# ── Brain MRI IDPs ────────────────────────────────────────────────────────────

def load_mri_idps(dataset_id: str, project_id: str, cfg: "Config") -> pd.DataFrame:
    """Load brain MRI imaging derived phenotypes (IDPs).

    Instance 2 = imaging visit.

    Returns
    -------
    DataFrame (eid × IDP) with volumetric and diffusion measures.
    """
    f = cfg.fields
    idp_field_ids = {
        "hippocampus_L":           f.get("hippocampus_L", 26562),
        "hippocampus_R":           f.get("hippocampus_R", 26563),
        "total_brain_vol":         f.get("total_brain_vol", 25010),
        "wmh_volume":              f.get("wmh_volume", 25781),
        "fa_corticospinal_L":      f.get("fa_corticospinal_L", 25056),
        "fa_corticospinal_R":      f.get("fa_corticospinal_R", 25057),
        "md_corticospinal_L":      f.get("md_corticospinal_L", 25452),
        "md_corticospinal_R":      f.get("md_corticospinal_R", 25453),
        "cortical_thickness_mean": f.get("cortical_thickness_mean", 26755),
    }

    # Instance 2 = imaging visit
    field_list = ["eid"] + [_p(fid, instance=2) for fid in idp_field_ids.values()]

    from src.data.dnanexus_access import extract_fields
    raw = extract_fields(dataset_id, field_list, project_id, entity="participant")

    rename = {_p(fid, instance=2): name for name, fid in idp_field_ids.items()}
    raw = raw.rename(columns=rename)
    raw["eid"] = raw["eid"].astype(str)
    raw = raw.set_index("eid")

    log.info(f"Loaded MRI IDPs: {len(raw)} participants × {raw.shape[1]} IDPs")
    return raw


# ── Methylation ──────────────────────────────────────────────────────────────

def load_methylation_paths(project_id: str, cfg: "Config") -> list[dict]:
    """Find EPIC methylation array files in the DNAnexus project."""
    from src.data.dnanexus_access import find_files

    files = find_files(
        project_id=project_id,
        folder="Bulk",
        name_pattern="*methylation*",
        recurse=True,
    )
    log.info(f"Found {len(files)} methylation files")
    return files


# ── Metabolomics (NMR) ────────────────────────────────────────────────────────

def load_metabolomics(dataset_id: str, project_id: str) -> pd.DataFrame | None:
    """Load NMR metabolomics data (field 23400 series, instanced).

    Returns None if metabolomics data is not in the dispensed dataset.
    """
    nmr_base = 23400
    # NMR metabolomics: ~250 measures, instanced
    fields = ["eid"] + [_p(nmr_base + i, instance=0) for i in range(250)]

    try:
        from src.data.dnanexus_access import extract_fields
        raw = extract_fields(dataset_id, fields, project_id, entity="participant")
        raw["eid"] = raw["eid"].astype(str)
        raw = raw.set_index("eid")
        data_cols = [c for c in raw.columns if c != "eid"]
        raw = raw[data_cols].dropna(axis=1, how="all")
        if raw.shape[1] == 0:
            log.info("NMR metabolomics fields are empty — not dispensed.")
            return None
        log.info(f"Loaded NMR metabolomics: {len(raw)} × {raw.shape[1]} metabolites")
        return raw
    except Exception as exc:
        log.warning(f"Could not load metabolomics: {exc}")
        return None
