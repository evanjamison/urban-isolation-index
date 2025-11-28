"""
10_merge_transit.py

Merge a city isolation index (Tokyo or Osaka) with transit metrics by ward.

This script does NOT require any access-related columns. It only assumes:

  - Index file has:  ward_jis  (as the ward identifier)
  - Transit file has: ward_jis  OR N03_007 (5-digit JIS code)

Examples (from project root, PowerShell):

  # Tokyo: jp_tokyo_index.csv + jp_tokyo_transit.csv → jp_tokyo_index_full.csv
  .\.venv\Scripts\python.exe -m src.cli.10_merge_transit `
      --city tokyo

  # Osaka: jp_osaka_index.csv + jp_osaka_transit.csv → jp_osaka_index_full.csv
  .\.venv\Scripts\python.exe -m src.cli.10_merge_transit `
      --city osaka

You can override the paths explicitly if desired:

  .\.venv\Scripts\python.exe -m src.cli.10_merge_transit `
      --index-path data/processed/jp_tokyo_index.csv `
      --transit-path data/interim/jp_tokyo_transit.csv `
      --out-path data/processed/jp_tokyo_index_full.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Merge transit metrics into a city index dataset (Tokyo or Osaka)."
    )

    p.add_argument(
        "--city",
        choices=["tokyo", "osaka"],
        default="tokyo",
        help="City to use for default paths (tokyo or osaka). Default: tokyo.",
    )
    p.add_argument(
        "--index-path",
        default=None,
        help=(
            "Index CSV (must contain ward_jis). "
            "If omitted, defaults are:\n"
            "  tokyo → data/processed/jp_tokyo_index.csv\n"
            "  osaka → data/processed/jp_osaka_index.csv"
        ),
    )
    p.add_argument(
        "--transit-path",
        default=None,
        help=(
            'Transit CSV (e.g., from 09_ingest_transit_alt.py). If omitted, defaults are:\n'
            "  tokyo → data/interim/jp_tokyo_transit.csv\n"
            "  osaka → data/interim/jp_osaka_transit.csv"
        ),
    )
    p.add_argument(
        "--out-path",
        default=None,
        help=(
            "Output merged CSV. If omitted, defaults are:\n"
            "  tokyo → data/processed/jp_tokyo_index_full.csv\n"
            "  osaka → data/processed/jp_osaka_index_full.csv"
        ),
    )

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ------------------------------------------------------------------
    # Resolve default paths by city if not explicitly provided
    # ------------------------------------------------------------------
    if args.city == "tokyo":
        default_index = Path("data/processed/jp_tokyo_index.csv")
        default_transit = Path("data/interim/jp_tokyo_transit.csv")
        default_out = Path("data/processed/jp_tokyo_index_full.csv")
    else:  # osaka
        default_index = Path("data/processed/jp_osaka_index.csv")
        default_transit = Path("data/interim/jp_osaka_transit.csv")
        default_out = Path("data/processed/jp_osaka_index_full.csv")

    index_path = Path(args.index_path) if args.index_path else default_index
    transit_path = Path(args.transit_path) if args.transit_path else default_transit
    out_path = Path(args.out_path) if args.out_path else default_out

    print(f"📂 City         : {args.city}")
    print(f"📂 Loading index from   {index_path} ...")
    idx = pd.read_csv(index_path)

    print(f"📂 Loading transit from {transit_path} ...")
    tr = pd.read_csv(transit_path)

    # ------------------------------------------------------------------
    # Handle ward ID columns
    # ------------------------------------------------------------------
    if "ward_jis" not in idx.columns:
        raise SystemExit("ERROR: ward_jis column not found in index dataset.")

    # Transit may use N03_007 (common in ward GeoJSON-derived tables)
    if "ward_jis" not in tr.columns:
        if "N03_007" in tr.columns:
            print("🔁 Renaming transit column N03_007 → ward_jis for merge.")
            tr = tr.rename(columns={"N03_007": "ward_jis"})
        else:
            raise SystemExit(
                "ERROR: Neither ward_jis nor N03_007 found in transit dataset; "
                "cannot merge."
            )

    # Coerce to common type
    idx["ward_jis"] = idx["ward_jis"].astype(str)
    tr["ward_jis"] = tr["ward_jis"].astype(str)

    # Avoid duplicate columns when merging
    dup_cols = [c for c in tr.columns if c in idx.columns and c != "ward_jis"]
    if dup_cols:
        print(f"⚠️  Dropping duplicate columns from transit before merge: {dup_cols}")
        tr = tr.drop(columns=dup_cols)

    print("🔗 Merging on ward_jis ...")
    merged = idx.merge(tr, on="ward_jis", how="left", validate="one_to_one")

    # ------------------------------------------------------------------
    # Ensure transit_z exists (if not already present)
    # ------------------------------------------------------------------
    if "transit_z" not in merged.columns:
        candidate_cols = [
            c
            for c in merged.columns
            if c.startswith("station_")
            or c.endswith("_count")
            or c == "station_density"
        ]
        if candidate_cols:
            base = candidate_cols[0]
            print(f"📏 Creating transit_z from {base} (z-score standardization).")
            x = merged[base].astype(float)
            std = x.std(ddof=0)
            if std == 0:
                merged["transit_z"] = 0.0
            else:
                merged["transit_z"] = (x - x.mean()) / std
        else:
            print("ℹ️ No obvious transit metric column found to standardize; "
                  "transit_z will be missing.")
    else:
        print("ℹ️ transit_z already present; leaving as-is.")

    print("\n✅ Merged dataset columns:")
    print(merged.columns.tolist())

    if "transit_z" in merged.columns:
        print("\n🧮 Summary of transit_z:")
        print(merged["transit_z"].describe())
    else:
        print("\nℹ️ transit_z column not present (no standardization performed).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"\n💾 Saved merged dataset with transit to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
