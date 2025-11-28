# src/cli/13_build_index_osaka.py

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from ..uix.index import IsolationIndexConfig, compute_isolation_index
import sys

# ---------------------------------------------------------------------
# Ensure "src" is on sys.path so we can import the uix package
# Usage: python -m src.cli.13_build_index_osaka
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main():
    parser = argparse.ArgumentParser(
        description="Build Osaka isolation index from jp_osaka_features."
    )
    parser.add_argument(
        "--features-in",
        default="data/interim/jp_osaka_features.parquet",
        help="Input features parquet (Osaka).",
    )
    parser.add_argument(
        "--out-parquet",
        default="data/processed/jp_osaka_index.parquet",
        help="Output parquet path (Osaka).",
    )
    parser.add_argument(
        "--out-csv",
        default="data/processed/jp_osaka_index.csv",
        help="Optional CSV export.",
    )
    args = parser.parse_args()

    in_path = Path(args.features_in)
    if not in_path.exists():
        raise SystemExit(f"Input features file not found: {in_path}")

    df = pd.read_parquet(in_path)

    # Same weights + logic as Tokyo (can change later if desired)
    config = IsolationIndexConfig()
    df_idx = compute_isolation_index(df, config=config)

    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df_idx.to_parquet(args.out_parquet, index=False)
    df_idx.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print(f"✅ Saved Osaka index parquet: {args.out_parquet}")
    print(f"✅ Saved Osaka index CSV:     {args.out_csv}")
    print(df_idx.head())


if __name__ == "__main__":
    main()

