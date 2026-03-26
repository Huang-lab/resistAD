"""Configuration loader for resistAD pipeline (UK Biobank / DNAnexus)."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DNAnexusConfig:
    project_id: str = ""
    phenotype_dataset_id: str = ""
    token_path: str = "data/dnanexus_token.txt"
    bulk: dict = field(default_factory=dict)


@dataclass
class CohortConfig:
    apoe_e4_as_risk: bool = True
    prs_risk_percentile: float = 0.90
    prs_control_max_percentile: float = 0.30
    min_age: int = 60
    cognition_field: int = 20016
    cognition_resilient_percentile: float = 0.75
    compute_resilience_index: bool = True
    confounders: list[str] = field(
        default_factory=lambda: [
            "age_at_assessment", "sex", "education",
            "assessment_centre", "genotype_array",
            "pc1", "pc2", "pc3", "pc4", "pc5",
            "pc6", "pc7", "pc8", "pc9", "pc10",
        ]
    )


@dataclass
class PRSConfig:
    gwas_weights_url: str = ""
    gwas_weights_file: str = "data/raw/bellenguez2022_ad_gwas.tsv.gz"
    clump_r2: float = 0.1
    clump_kb: int = 250
    p_threshold: float = 5e-8
    exclude_apoe_region: bool = True
    apoe_chr: int = 19
    apoe_start_mb: float = 44.4
    apoe_end_mb: float = 46.5


@dataclass
class Thresholds:
    padj: float = 0.05
    effect_size_min: float = 0.2
    ppi_score_threshold: int = 700
    mofa_n_factors: int = 15
    mofa_min_r2: float = 0.01
    methylation_delta_beta: float = 0.05
    methylation_padj: float = 0.05
    meta_padj: float = 0.05
    convergence_n_layers: int = 2


@dataclass
class OlinkConfig:
    lod_filter: bool = True
    bridge_normalise: bool = False
    panels: list[str] = field(
        default_factory=lambda: ["Cardiometabolic", "Inflammation", "Neurology", "Oncology"]
    )
    neurological_proteins: list[str] = field(
        default_factory=lambda: ["GFAP", "NEFL", "NRGN", "BDNF", "CLU", "CR1", "BIN1"]
    )


@dataclass
class ImagingConfig:
    idp_covariates: list[str] = field(default_factory=lambda: ["total_brain_vol", "assessment_centre"])
    idp_groups: dict = field(default_factory=dict)


@dataclass
class RareVariantConfig:
    gene_list: list[str] = field(
        default_factory=lambda: ["TREM2", "PLCG2", "ABI3", "ABCA7", "SORL1", "APP", "PSEN1", "PSEN2", "APOE"]
    )
    maf_threshold: float = 0.01
    cadd_threshold: int = 20


@dataclass
class IntegrationConfig:
    n_factors: int = 15
    convergence_n_layers: int = 2


@dataclass
class Config:
    project_root: Path = field(default_factory=Path.cwd)
    output_dir: Path = field(default_factory=lambda: Path("analysis/out"))
    data_dir: Path = field(default_factory=lambda: Path("data"))
    raw_dir: Path = field(default_factory=lambda: Path("data/raw"))
    dnanexus: DNAnexusConfig = field(default_factory=DNAnexusConfig)
    fields: dict = field(default_factory=dict)
    cohort: CohortConfig = field(default_factory=CohortConfig)
    prs: PRSConfig = field(default_factory=PRSConfig)
    thresholds: Thresholds = field(default_factory=Thresholds)
    olink: OlinkConfig = field(default_factory=OlinkConfig)
    imaging: ImagingConfig = field(default_factory=ImagingConfig)
    rare_variants: RareVariantConfig = field(default_factory=RareVariantConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)

    def resolve_path(self, p: Path | str) -> Path:
        p = Path(p)
        if p.is_absolute():
            return p
        return self.project_root / p


def load_config(config_path: str | Path | None = None, project_root: str | Path | None = None) -> Config:
    """Load pipeline configuration from YAML file.

    Parameters
    ----------
    config_path : path to config.yaml (default: <project_root>/config/config.yaml)
    project_root : project root directory (default: current working directory)
    """
    if project_root is None:
        project_root = Path.cwd()
    project_root = Path(project_root)

    if config_path is None:
        config_path = project_root / "config" / "config.yaml"
    config_path = Path(config_path)

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    proj = raw.get("project", {})
    dx_raw = raw.get("dnanexus", {})
    bulk = dx_raw.pop("bulk", {})

    cfg = Config(
        project_root=project_root,
        output_dir=Path(proj.get("output_dir", "analysis/out")),
        data_dir=Path(proj.get("data_dir", "data")),
        raw_dir=Path(proj.get("raw_dir", "data/raw")),
        dnanexus=DNAnexusConfig(**dx_raw, bulk=bulk),
        fields=raw.get("fields", {}),
        cohort=CohortConfig(**raw.get("cohort", {})),
        prs=PRSConfig(**raw.get("prs", {})),
        thresholds=Thresholds(**raw.get("thresholds", {})),
        olink=OlinkConfig(**raw.get("olink", {})),
        imaging=ImagingConfig(**raw.get("imaging", {})),
        rare_variants=RareVariantConfig(**raw.get("rare_variants", {})),
        integration=IntegrationConfig(**raw.get("integration", {})),
    )

    # Ensure output directories exist
    cfg.resolve_path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    cfg.resolve_path(cfg.raw_dir).mkdir(parents=True, exist_ok=True)

    return cfg
