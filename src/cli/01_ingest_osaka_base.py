from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest and clean Osaka ward-level demographic base data."
    )

    parser.add_argument(
        "--raw-in",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "jp_osaka_base_raw.csv",
        help="Raw Osaka demographics CSV (ward-level).",
    )
    parser.add_argument(
        "--out-raw",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "jp_osaka_base.csv",
        help="Normalized raw output CSV (optional).",
    )
    parser.add_argument(
        "--out-clean",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "jp_osaka_base_clean.csv",
        help="Cleaned Osaka base CSV (used downstream for features).",
    )

    args = parser.parse_args()
    args.out_raw.parent.mkdir(parents=True, exist_ok=True)
    args.out_clean.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.raw_in)

    # --- Standardize column names -----------------------------------------
    rename_map = {
        "ward_name": "ward_name_ja",
        "ward": "ward_name_ja",
        "total_population": "pop_total",
        "population_total": "pop_total",
        "population": "pop_total",
        "pop_65p": "pop_65plus",
        "population_65plus": "pop_65plus",
        "pop_65plus": "pop_65plus",
        "single_hh_65plus": "hh_single_65plus",
        "single65p": "hh_single_65plus",
        "poverty": "poverty_rate",
        "poverty_pct": "poverty_rate",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    required = ["ward_name_ja", "pop_total", "pop_65plus", "hh_single_65plus", "poverty_rate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {args.raw_in}: {missing}\n"
            "Update 01_ingest_osaka_base.py rename_map or your raw CSV."
        )

    # --- Compute percentages/proportions ----------------------------------
    df["pct_age65p"] = df["pop_65plus"] / df["pop_total"]
    df["pct_single65p"] = df["hh_single_65plus"] / df["pop_65plus"].replace(0, pd.NA)

    # Keep only the columns we care about in a clean, tidy table
    clean_cols = [
        "ward_name_ja",
        "pop_total",
        "pop_65plus",
        "hh_single_65plus",
        "poverty_rate",
        "pct_age65p",
        "pct_single65p",
    ]
    clean = df[clean_cols].copy()

    df.to_csv(args.out_raw, index=False)
    clean.to_csv(args.out_clean, index=False)

    print(f"[ok] Wrote normalized Osaka base → {args.out_raw}")
    print(f"[ok] Wrote cleaned Osaka base   → {args.out_clean}")


if __name__ == "__main__":
    main()
