"""DNAnexus platform access layer for the resistAD pipeline.

All participant-level data remains on the DNAnexus RAP environment.
This module provides authenticated access to UK Biobank phenotype datasets
and bulk data files.

Field extraction uses the `dx extract_dataset` CLI, which works both locally
(with `dxpy` installed and `dx login` done) and on the RAP JupyterLab.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("resistad.data.dnanexus")


# ── Authentication ────────────────────────────────────────────────────────────

def login(token: str | None = None, token_path: str | Path | None = None) -> None:
    """Authenticate with the DNAnexus platform.

    Credentials are checked in this order:
    1. `token` argument
    2. `token_path` file
    3. DX_SECURITY_CONTEXT environment variable
    4. Existing ~/.dnanexus_config (written by `dx login`)
    """
    import dxpy

    if token is None and token_path is not None:
        token_path = Path(token_path)
        if token_path.exists():
            token = token_path.read_text().strip()

    if token:
        dxpy.set_security_context({"auth_token_type": "Bearer", "auth_token": token})
        log.info("Authenticated with DNAnexus using provided token.")
    else:
        try:
            whoami_info = dxpy.api.system_whoami()
            log.info(f"Using existing DNAnexus session: {whoami_info.get('username', '?')}")
        except Exception as exc:
            raise RuntimeError(
                "DNAnexus authentication failed. Run `dx login` or provide a token."
            ) from exc


def whoami() -> str:
    """Return the currently authenticated DNAnexus username."""
    import dxpy
    return dxpy.api.system_whoami().get("username", "unknown")


# ── dx CLI helper ────────────────────────────────────────────────────────────

def _dx_bin() -> str:
    """Locate the dx CLI binary."""
    dx = shutil.which("dx")
    if dx:
        return dx
    # Fallback: look in the same venv as the running Python
    venv_dx = Path(os.sys.executable).parent / "dx"
    if venv_dx.exists():
        return str(venv_dx)
    raise FileNotFoundError(
        "Cannot find `dx` CLI. Install dxpy (`pip install dxpy`) "
        "or ensure it is on PATH."
    )


# ── Dataset / phenotype access via CLI ───────────────────────────────────────

def _dataset_path(dataset_id: str, project_id: str | None = None) -> str:
    """Build the dataset path accepted by dx extract_dataset.

    Format: project-xxx:record-name  OR  just record-name (uses current project).
    """
    if project_id:
        return f"{project_id}:{dataset_id}"
    return dataset_id


def list_entities(dataset_id: str, project_id: str | None = None) -> list[str]:
    """List entity names in the dataset."""
    path = _dataset_path(dataset_id, project_id)
    result = subprocess.run(
        [_dx_bin(), "extract_dataset", path, "--list-entities"],
        capture_output=True, text=True, check=True,
    )
    entities = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if parts:
            entities.append(parts[0])
    return entities


def list_fields(
    dataset_id: str,
    project_id: str | None = None,
    entity: str = "participant",
) -> pd.DataFrame:
    """List available fields for an entity in the dataset.

    Returns DataFrame with columns [field_name, field_title].
    """
    path = _dataset_path(dataset_id, project_id)
    result = subprocess.run(
        [_dx_bin(), "extract_dataset", path,
         "--list-fields", "--entities", entity],
        capture_output=True, text=True, check=True,
    )
    rows = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t", maxsplit=1)
        if len(parts) == 2:
            rows.append({"field_name": parts[0], "field_title": parts[1]})
    return pd.DataFrame(rows)


def extract_fields(
    dataset_id: str,
    field_names: list[str],
    project_id: str | None = None,
    entity: str = "participant",
) -> pd.DataFrame:
    """Extract phenotype fields from a UKB dispensed dataset.

    Uses `dx extract_dataset` CLI — works locally and on RAP.

    Parameters
    ----------
    dataset_id : the dataset record ID (e.g. 'app151418_20260115143845')
    field_names : list of field names in UKB p<field>_i<instance>_a<array> format
                  e.g. ['eid', 'p20016_i0', 'p21003_i0']
    project_id : DNAnexus project ID (optional; uses current project if None)
    entity : dataset entity name (default 'participant')

    Returns
    -------
    pd.DataFrame with one row per participant
    """
    path = _dataset_path(dataset_id, project_id)
    dx = _dx_bin()

    # Validate fields against what the dataset actually has
    available = list_fields(dataset_id, project_id, entity=entity)
    available_set = set(available["field_name"].tolist())  # entity.field format

    qualified = []
    dropped = []
    for f in field_names:
        qf = f"{entity}.{f}"
        if qf in available_set:
            qualified.append(qf)
        else:
            dropped.append(f)
    if dropped:
        log.warning(
            f"Dropped {len(dropped)} fields not in dataset: "
            + ", ".join(dropped[:10])
            + ("..." if len(dropped) > 10 else "")
        )
    if not qualified:
        raise ValueError("No valid fields to extract after validation.")

    # Ensure eid is always the first qualified field
    eid_field = f"{entity}.eid"
    if eid_field in qualified:
        qualified.remove(eid_field)
    non_eid = qualified  # fields other than eid

    # Extract in batches to avoid server-side response size limits.
    # For the 'participant' entity (~500K rows): 15 fields per batch is safe.
    # For smaller entities (olink ~54K, hesin_diag): use larger batches.
    BATCH_SIZE = 15 if entity == "participant" else 100
    log.info(
        f"Extracting {len(non_eid)+1} fields from {dataset_id} "
        f"({entity}) in batches of {BATCH_SIZE} ..."
    )

    batches = [
        non_eid[i : i + BATCH_SIZE]
        for i in range(0, len(non_eid), BATCH_SIZE)
    ]
    dfs: list[pd.DataFrame] = []

    for batch_idx, batch in enumerate(batches, 1):
        batch_fields = [eid_field] + batch
        fields_file = tempfile.mktemp(suffix=".txt")
        out_path = tempfile.mktemp(suffix=".csv")

        MAX_RETRIES = 3
        for attempt in range(1, MAX_RETRIES + 1):
            # Ensure clean paths for each attempt
            Path(fields_file).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)

            try:
                Path(fields_file).write_text("\n".join(batch_fields))
                cmd = [
                    dx, "extract_dataset", path,
                    "--fields-file", fields_file,
                    "--delim", ",",
                    "-o", out_path,
                ]
                log.info(
                    f"  Batch {batch_idx}/{len(batches)}: "
                    f"{len(batch)} fields"
                    + (f" (attempt {attempt})" if attempt > 1 else "")
                )
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=False,
                )
                if result.returncode == 0:
                    batch_df = pd.read_csv(out_path)
                    batch_df.columns = [
                        c.split(".", 1)[-1] if "." in c else c
                        for c in batch_df.columns
                    ]
                    dfs.append(batch_df)
                    break  # success

                # Retriable server errors
                stderr = result.stderr
                if attempt < MAX_RETRIES and (
                    "BadJSONInReply" in stderr
                    or "Internal Server Error" in stderr
                    or "timed out" in stderr.lower()
                ):
                    import time as _time
                    wait = 5 * attempt
                    log.warning(
                        f"  Batch {batch_idx} failed (attempt {attempt}), "
                        f"retrying in {wait}s ..."
                    )
                    _time.sleep(wait)
                    continue

                # Non-retriable error
                log.error(f"dx extract_dataset stderr:\n{stderr}")
                raise RuntimeError(
                    f"dx extract_dataset batch {batch_idx} failed "
                    f"(exit {result.returncode}): {stderr[:500]}"
                )

            finally:
                Path(fields_file).unlink(missing_ok=True)
                Path(out_path).unlink(missing_ok=True)

    # Merge all batches on eid
    if len(dfs) == 1:
        merged = dfs[0]
    else:
        merged = dfs[0]
        for extra_df in dfs[1:]:
            merged = merged.merge(extra_df, on="eid", how="outer")

    log.info(f"Extracted {len(merged)} rows × {len(merged.columns)} columns")
    return merged


# ── File access ───────────────────────────────────────────────────────────────

def find_files(
    project_id: str,
    folder: str,
    name_pattern: str | None = None,
    recurse: bool = False,
) -> list[dict[str, Any]]:
    """List files in a DNAnexus project folder.

    Parameters
    ----------
    project_id : DNAnexus project ID
    folder : folder path within the project (e.g. 'Bulk/Protein biomarkers/Olink')
    name_pattern : optional glob-style name filter
    recurse : search subdirectories

    Returns
    -------
    list of dicts with keys: id, name, folder, size
    """
    import dxpy

    results = []
    kwargs: dict[str, Any] = {"project": project_id, "folder": f"/{folder}"}
    if name_pattern:
        kwargs["name"] = name_pattern
        kwargs["name_mode"] = "glob"
    if recurse:
        kwargs["recurse"] = True

    for item in dxpy.find_data_objects(classname="file", **kwargs):
        desc = dxpy.describe(item["id"])
        results.append({
            "id": item["id"],
            "name": desc["name"],
            "folder": desc["folder"],
            "size": desc.get("size", 0),
        })

    log.info(f"Found {len(results)} files in {folder}")
    return results


def download_file(file_id: str, dest: Path, overwrite: bool = False) -> Path:
    """Download a DNAnexus file to a local path.

    NOTE: Only use this for non-participant-level files (e.g. reference files,
    GWAS summary statistics). Never download individual-level data locally
    unless inside the DNAnexus RAP secure environment.
    """
    import dxpy

    dest = Path(dest)
    if dest.exists() and not overwrite:
        log.info(f"Using cached file: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    dxpy.DXFile(file_id).download(str(dest))
    log.info(f"Downloaded {file_id} → {dest}")
    return dest


# ── Availability probe ────────────────────────────────────────────────────────

def check_data_availability(project_id: str, bulk_paths: dict[str, str]) -> dict[str, bool]:
    """Check which omic data layers are present in the DNAnexus project.

    Parameters
    ----------
    project_id : DNAnexus project ID
    bulk_paths : mapping of layer name → folder path from config

    Returns
    -------
    dict of layer_name → bool (True if at least one file found)
    """
    import dxpy

    availability: dict[str, bool] = {}
    for layer, folder in bulk_paths.items():
        try:
            hits = list(dxpy.find_data_objects(
                classname="file",
                project=project_id,
                folder=f"/{folder}",
                limit=1,
            ))
            availability[layer] = len(hits) > 0
        except Exception:
            availability[layer] = False

    for layer, present in availability.items():
        status = "PRESENT" if present else "NOT FOUND"
        log.info(f"  {layer:<25} {status}")

    return availability
