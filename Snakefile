"""Snakemake workflow for resistAD UK Biobank multi-omic resilience pipeline.

DAG
───
  00_check_availability → 01_download → 02_cohorts
                                             │
                         ┌───────────────────┼────────────────────┐
                         ▼                   ▼                    ▼                    ▼
                    03_genetics        04_proteomics         05_imaging        06_methylation
                         └───────────────────┬────────────────────┘
                                             ▼
                                       07_integration

Steps 3–6 are fully independent and run in parallel when --cores > 1.

Usage
─────
    snakemake --cores 4                     # Full pipeline (parallelise steps 3–6)
    snakemake integration                   # Integration (triggers all deps)
    snakemake proteomics                    # Proteomics only (+ deps)
    snakemake -n                            # Dry run: show what would execute
    snakemake --forcerun cohorts            # Re-run cohort step + all downstream
    snakemake --config skip_mofa=True       # Skip MOFA+ in step 07
    snakemake --config skip_prs=True        # Use pre-computed PRS scores
"""

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT = config.get("output_dir", "analysis/out")
DATA   = config.get("data_dir",   "data/raw")

SKIP_MOFA = config.get("skip_mofa", False)
SKIP_PRS  = config.get("skip_prs",  False)
SKIP_RARE = config.get("skip_rare", False)

MOFA_FLAG = "--skip-mofa" if SKIP_MOFA else ""
PRS_FLAG  = "--skip-prs"  if SKIP_PRS  else ""
RARE_FLAG = "--skip-rare" if SKIP_RARE else ""


# ── Final target ──────────────────────────────────────────────────────────────

rule all:
    """Default target: produce the resilience signature from all omic layers."""
    input:
        f"{OUTPUT}/integration/resilience_signature.csv",


# ── Step 00: Check data availability on DNAnexus ─────────────────────────────

rule check_availability:
    """Probe DNAnexus project for dispensed omic layers → data_availability.yaml."""
    output:
        yaml  = f"{DATA}/data_availability.yaml",
        stamp = touch(f"{OUTPUT}/.availability_checked"),
    log:
        f"{OUTPUT}/logs/00_check_availability.log",
    shell:
        "python scripts/00_check_data_availability.py 2>&1 | tee {log}"


# ── Step 01: Download data from DNAnexus ─────────────────────────────────────

rule download:
    """Extract UKB fields and download bulk omic files from DNAnexus."""
    input:
        f"{DATA}/data_availability.yaml",
    output:
        phenotypes = f"{DATA}/ukb_phenotypes.parquet",
        stamp      = touch(f"{OUTPUT}/.download_done"),
    log:
        f"{OUTPUT}/logs/01_download.log",
    shell:
        "python scripts/01_download_data.py 2>&1 | tee {log}"


# ── Step 02: Define resilience cohorts ───────────────────────────────────────

rule cohorts:
    """Assign participants to Resilient / Vulnerable / Control based on APOE/PRS."""
    input:
        f"{DATA}/ukb_phenotypes.parquet",
    output:
        cohorts = f"{OUTPUT}/cohorts.parquet",
        summary = f"{OUTPUT}/cohort_summary.csv",
    log:
        f"{OUTPUT}/logs/02_cohorts.log",
    shell:
        "python scripts/02_define_cohorts.py 2>&1 | tee {log}"


# ── Step 03: Genetics ─────────────────────────────────────────────────────────

rule genetics:
    """Compute AD-PRS, APOE frequency tables, and optional WES burden tests."""
    input:
        f"{OUTPUT}/cohorts.parquet",
    output:
        apoe = f"{OUTPUT}/genetics/apoe_frequency.csv",
        prs  = f"{OUTPUT}/genetics/prs_cohort_stats.csv",
    log:
        f"{OUTPUT}/logs/03_genetics.log",
    shell:
        "python scripts/03_genetics.py "
        + f"{PRS_FLAG} {RARE_FLAG}".strip()
        + " 2>&1 | tee {log}"


# ── Step 04: Proteomics ───────────────────────────────────────────────────────

rule proteomics:
    """Olink differential abundance analysis for all cohort contrasts."""
    input:
        f"{OUTPUT}/cohorts.parquet",
    output:
        de_rv = f"{OUTPUT}/proteomics/protein_de_resilient_vs_vulnerable.csv",
        de_rc = f"{OUTPUT}/proteomics/protein_de_resilient_vs_control.csv",
    log:
        f"{OUTPUT}/logs/04_proteomics.log",
    shell:
        "python scripts/04_proteomics.py 2>&1 | tee {log}"


# ── Step 05: Imaging ──────────────────────────────────────────────────────────

rule imaging:
    """MRI IDP cohort differences and brain age gap computation."""
    input:
        f"{OUTPUT}/cohorts.parquet",
    output:
        idp = f"{OUTPUT}/imaging/idp_differences.csv",
        bag = f"{OUTPUT}/imaging/brain_age_gap.csv",
    log:
        f"{OUTPUT}/logs/05_imaging.log",
    shell:
        "python scripts/05_imaging.py 2>&1 | tee {log}"


# ── Step 06: Methylation (conditional) ───────────────────────────────────────

rule methylation:
    """EPIC DMP/DMR analysis (exits cleanly if beta matrix is not dispensed)."""
    input:
        f"{OUTPUT}/cohorts.parquet",
    output:
        stamp = touch(f"{OUTPUT}/.methylation_done"),
    log:
        f"{OUTPUT}/logs/06_methylation.log",
    shell:
        "python scripts/06_methylation.py 2>&1 | tee {log}"


# ── Step 07: Integration ──────────────────────────────────────────────────────

rule integration:
    """MOFA+ factor analysis, multi-layer network, and resilience signature."""
    input:
        cohorts     = f"{OUTPUT}/cohorts.parquet",
        genetics    = f"{OUTPUT}/genetics/apoe_frequency.csv",
        proteomics  = f"{OUTPUT}/proteomics/protein_de_resilient_vs_vulnerable.csv",
        imaging     = f"{OUTPUT}/imaging/idp_differences.csv",
        methylation = f"{OUTPUT}/.methylation_done",
    output:
        signature   = f"{OUTPUT}/integration/resilience_signature.csv",
        mofa_assoc  = f"{OUTPUT}/integration/mofa_factor_associations.csv",
    log:
        f"{OUTPUT}/logs/07_integration.log",
    shell:
        "python scripts/07_integration.py "
        + f"{MOFA_FLAG}".strip()
        + " 2>&1 | tee {log}"


# ── Convenience aliases ───────────────────────────────────────────────────────

rule download_only:
    """Download data without running any analysis."""
    input:
        rules.download.output,

rule define_cohorts:
    """Run cohort definition without downstream analyses."""
    input:
        rules.cohorts.output,
