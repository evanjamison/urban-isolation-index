# src/cli/03_build_index.py

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from ..uix.index import IsolationIndexConfig, compute_isolation_index
import sys

# ---------------------------------------------------------------------
# Ensure "src" is on sys.path so we can import the uix package
# when running: python -m src.cli.03_build_index
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def default_paths_for_city(city: str) -> tuple[Path, Path, Path]:
    """
    Return (features_in, out_parquet, out_csv) defaults for a given city.
    """
    if city == "tokyo":
        features_in = PROJECT_ROOT / "data" / "interim" / "jp_tokyo_full_features.parquet"
        out_parquet = PROJECT_ROOT / "data" / "processed" / "jp_tokyo_index.parquet"
        out_csv = PROJECT_ROOT / "data" / "processed" / "jp_tokyo_index.csv"
    elif city == "osaka":
        features_in = PROJECT_ROOT / "data" / "interim" / "jp_osaka_full_features.parquet"
        out_parquet = PROJECT_ROOT / "data" / "processed" / "jp_osaka_index.parquet"
        out_csv = PROJECT_ROOT / "data" / "processed" / "jp_osaka_index.csv"
    else:
        raise ValueError(f"Unknown city: {city}")

    return features_in, out_parquet, out_csv




def isolation_config_for_city(city: str) -> IsolationIndexConfig:
    """
    Build the isolation index config for a given city.

    Currently both Tokyo and Osaka use the same metric set and weights:
      - pct_age65p
      - pct_single65p
      - poverty_rate

    You can extend this later (e.g., to add access metrics) by editing
    metrics/weights here.
    """
    metrics = ["pct_age65p", "pct_single65p", "poverty_rate"]
    weights = {
        "pct_age65p": 0.4,
        "pct_single65p": 0.3,
        "poverty_rate": 0.3,
    }

    # Keep a consistent column name across cities so downstream code still works
    index_col = "iso_index"

    return IsolationIndexConfig(
        metrics=metrics,
        weights=weights,
        index_col=index_col,
    )


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build isolation index (D-IRI style) for a Japanese city "
            "(Tokyo or Osaka) from pre-computed features."
        )
    )

    parser.add_argument(
        "--city",
        choices=["tokyo", "osaka"],
        default="tokyo",
        help="Target city to build index for. Default: tokyo",
    )

    parser.add_argument(
        "--features-in",
        type=Path,
        default=None,
        help=(
            "Input features parquet. If omitted, a city-specific default is used:\n"
            "  tokyo → data/interim/jp_tokyo_features.parquet\n"
            "  osaka → data/interim/jp_osaka_features.parquet"
        ),
    )
    parser.add_argument(
        "--out-parquet",
        type=Path,
        default=None,
        help=(
            "Output parquet path. If omitted, a city-specific default is used:\n"
            "  tokyo → data/processed/jp_tokyo_index.parquet\n"
            "  osaka → data/processed/jp_osaka_index.parquet"
        ),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help=(
                "Optional CSV export. If omitted, a city-specific default is used:\n"
                "  tokyo → data/processed/jp_tokyo_index.csv\n"
                "  osaka → data/processed/jp_osaka_index.csv"
        ),
    )

    args = parser.parse_args()

    # Fill in city-specific defaults if not provided
    def_feats, def_parquet, def_csv = default_paths_for_city(args.city)
    features_in = args.features_in or def_feats
    out_parquet = args.out_parquet or def_parquet
    out_csv = args.out_csv or def_csv

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] Building isolation index for city = {args.city}")
    print(f"[info] Reading features from: {features_in}")

    features_df = pd.read_parquet(features_in)

    cfg = isolation_config_for_city(args.city)
    result_df = compute_isolation_index(features_df, cfg)

    result_df.to_parquet(out_parquet, index=False)
    result_df.to_csv(out_csv, index=False)

    print(f"[ok] Wrote isolation index parquet → {out_parquet}")
    print(f"[ok] Wrote isolation index CSV     → {out_csv}")


if __name__ == "__main__":
    main()
