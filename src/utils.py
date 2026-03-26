"""Shared utilities for the resistAD pipeline (UK Biobank / DNAnexus)."""

import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import Config


def setup_logging(name: str = "resistad", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with console output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


log = setup_logging()


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_path(config: Config, *parts: str) -> Path:
    """Build an output path relative to the configured output directory.

    Example: output_path(cfg, "proteomics", "de_resilient_vs_vulnerable.csv")
    """
    out = config.resolve_path(config.output_dir)
    for part in parts:
        out = out / part
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def save_df(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Save a DataFrame to CSV, TSV, or Parquet based on file extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, **kwargs)
    elif suffix == ".tsv":
        df.to_csv(path, sep="\t", **kwargs)
    else:
        df.to_csv(path, **kwargs)
    log.info(f"Saved {path.name} ({len(df)} rows)")


def load_df(path: Path, **kwargs) -> pd.DataFrame:
    """Load a DataFrame from CSV, TSV, or Parquet."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    elif suffix == ".tsv":
        return pd.read_csv(path, sep="\t", **kwargs)
    else:
        return pd.read_csv(path, **kwargs)


def clean_eid(eid_series: pd.Series) -> pd.Series:
    """Clean UK Biobank participant IDs (eid) to integer strings.

    UK Biobank eids are 7-digit positive integers. This normalises them
    for consistent merging across data sources.
    """
    return eid_series.astype(str).str.strip().str.lstrip("0").apply(
        lambda x: x if x else "0"
    )


def standardise(series: pd.Series) -> pd.Series:
    """Z-score standardise a numeric Series (mean=0, sd=1)."""
    return (series - series.mean()) / series.std(ddof=1)


def age_adjust(df: pd.DataFrame, value_col: str, age_col: str = "age_at_assessment") -> pd.Series:
    """Return residuals after regressing value_col on age.

    Used to produce age-adjusted cognitive scores and imaging IDPs.
    """
    import statsmodels.formula.api as smf

    sub = df[[value_col, age_col]].dropna()
    model = smf.ols(f"{value_col} ~ {age_col}", data=sub).fit()
    residuals = pd.Series(model.resid, index=sub.index, name=f"{value_col}_age_adj")
    return df.index.map(residuals)
