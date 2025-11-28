"""
build_designed_index.py

Compute a designed (weighted) isolation index (D-IRI) for ANY city
(Tokyo or Osaka) using ONLY:

  - pct_age65p_z
  - pct_single65p_z
  - poverty_rate_z
  - transit_z  (higher transit = lower isolation)

Usage:

  python scripts/build_designed_index.py \
      --in-path data/processed/jp_tokyo_index_full.csv \
      --out-path data/processed/jp_tokyo_with_designed.csv

  python scripts/build_designed_index.py \
      --in-path data/processed/jp_osaka_index_full.csv \
      --out-path data/processed/jp_osaka_with_designed.csv
"""

import pandas as pd
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-path", required=True)
    p.add_argument("--out-path", required=True)
    args = p.parse_args()

    print("📥 Loading dataset:", args.in_path)
    df = pd.read_csv(args.in_path)

    # Check required columns
    required = [
        "pct_age65p_z",
        "pct_single65p_z",
        "poverty_rate_z",
        "transit_z",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("🧮 Computing Designed Isolation Index (D-IRI)...")

    # Reverse-sign transit (more transit = *less* isolation)
    df["neg_transit_z"] = -df["transit_z"]

    df["iri_designed"] = (
        0.25 * df["pct_age65p_z"] +
        0.25 * df["pct_single65p_z"] +
        0.20 * df["poverty_rate_z"] +
        0.15 * df["neg_transit_z"]
    )

    print("\nSummary of iri_designed:")
    print(df["iri_designed"].describe())

    print("💾 Saving:", args.out_path)
    df.to_csv(args.out_path, index=False)


if __name__ == "__main__":
    main()
