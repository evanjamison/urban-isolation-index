"""
13_normalize_indices.py

Normalize Tokyo/Osaka isolation indices to a 0–100 scale.

Usage (Tokyo):
  python -m src.cli.13_normalize_indices ^
      --in-path data/processed/jp_tokyo_with_designed_pca.csv ^
      --out-path data/processed/jp_tokyo_indices_normalized.csv

Usage (Osaka):
  python -m src.cli.13_normalize_indices ^
      --in-path data/processed/jp_osaka_with_designed_pca.csv ^
      --out-path data/processed/jp_osaka_indices_normalized.csv
"""

import argparse
from pathlib import Path
import pandas as pd


def minmax_0_100(series: pd.Series) -> pd.Series:
    """Scale a numeric Series to a 0–100 range."""
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return pd.Series([50] * len(series), index=series.index)
    return 100 * (series - min_val) / (max_val - min_val)


def main():
    parser = argparse.ArgumentParser(description="Normalize isolation indices to 0–100.")
    parser.add_argument("--in-path", required=True,
                        help="CSV with iri_designed, iri_pca, and optional iso_index")
    parser.add_argument("--out-path", required=True,
                        help="Output CSV path")
    args = parser.parse_args()

    print(f"📥 Loading dataset from {args.in_path} ...")
    df = pd.read_csv(args.in_path)

    # --------------------------------------------
    # Required columns for *all* cities
    # --------------------------------------------
    required_base = ["iri_designed", "iri_pca"]
    missing_required = [c for c in required_base if c not in df.columns]

    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    # --------------------------------------------
    # Optional columns (normalize only if present)
    # --------------------------------------------
    optional_cols = [col for col in ["iso_index", "transit_z"] if col in df.columns]

    print("\n🧮 Scaling indices to 0–100 ...")
    df["iri_designed_100"] = minmax_0_100(df["iri_designed"])
    df["iri_pca_100"] = minmax_0_100(df["iri_pca"])

    for col in optional_cols:
        out_col = f"{col}_100"
        df[out_col] = minmax_0_100(df[col])
        print(f"   • Normalized: {col} → {out_col}")

    print("\n📊 Summary of normalized indices:")
    normalized_cols = [c for c in df.columns if c.endswith("_100")]
    print(df[normalized_cols].describe())

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n💾 Saved 0–100 normalized dataset → {out_path}")


if __name__ == "__main__":
    main()
