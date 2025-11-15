# src/uix/index.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd


@dataclass
class IsolationIndexConfig:
    """Config for combining metrics into a single index."""
    metrics: List[str] = None
    weights: Dict[str, float] = None
    index_col: str = "iso_index"

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["pct_age65p", "pct_single65p", "poverty_rate"]
        if self.weights is None:
            # must sum to 1.0
            self.weights = {
                "pct_age65p": 0.4,
                "pct_single65p": 0.3,
                "poverty_rate": 0.3,
            }


def _zscore(series: pd.Series) -> pd.Series:
    """Standardize a column: (x - mean) / std."""
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean()
    sigma = s.std(ddof=0)
    if sigma == 0 or pd.isna(sigma):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


def compute_isolation_index(df: pd.DataFrame,
                            config: IsolationIndexConfig | None = None
                            ) -> pd.DataFrame:
    """
    Add z-scored metrics and a combined isolation index.

    Returns a new DataFrame with extra columns:
      - <metric>_z
      - config.index_col (default: iso_index)
    """
    if config is None:
        config = IsolationIndexConfig()

    out = df.copy()

    # 1) z-score each metric
    z_cols = {}
    for metric in config.metrics:
        if metric not in out.columns:
            raise KeyError(f"Metric '{metric}' not found in DataFrame columns.")
        z_name = f"{metric}_z"
        out[z_name] = _zscore(out[metric])
        z_cols[metric] = z_name

    # 2) weighted sum
    idx = 0.0
    for metric, weight in config.weights.items():
        z_name = z_cols[metric]
        idx = idx + weight * out[z_name]

    out[config.index_col] = idx

    return out
