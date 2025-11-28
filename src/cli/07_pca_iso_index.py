"""
07_pca_iso_index.py

Compute a PCA-based isolation index for Tokyo and compare it to the
existing designed iso_index. Produces:

- data/processed/jp_tokyo_index_pca.csv
- out/maps/tokyo_iso_index_comparison.png
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def zscore(series: pd.Series) -> pd.Series:
    """Return population z-score (mean 0, std 1)."""
    mu = series.mean()
    sigma = series.std(ddof=0)
    if sigma == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mu) / sigma


def ensure_dirs(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------
def compute_pca_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add iso_index_pca column to df based on PCA of z-scored predictors."""
    needed = ["pct_age65p_z", "pct_single65p_z", "poverty_rate_z"]

    # If z columns are missing, create them from raw %
    if not all(col in df.columns for col in needed):
        raw_needed = ["pct_age65p", "pct_single65p", "poverty_rate"]
        missing_raw = [c for c in raw_needed if c not in df.columns]
        if missing_raw:
            raise ValueError(
                f"Missing needed columns for PCA: {missing_raw}. "
                f"Expected either z-scored columns {needed} "
                f"or raw columns {raw_needed}."
            )
        print("⚠ z-columns missing, computing z-scores from raw columns...")
        df["pct_age65p_z"] = zscore(df["pct_age65p"])
        df["pct_single65p_z"] = zscore(df["pct_single65p"])
        df["poverty_rate_z"] = zscore(df["poverty_rate"])

    X = df[needed].to_numpy()
    pca = PCA(n_components=1)
    scores = pca.fit_transform(X)[:, 0]

    # Standardize PC1 to mean 0, std 1
    scores_z = (scores - scores.mean()) / scores.std(ddof=0)

    # If existing iso_index is present, align sign with it
    if "iso_index" in df.columns:
        corr = np.corrcoef(scores_z, df["iso_index"])[0, 1]
        print(f"Correlation between PC1 and existing iso_index: {corr:.3f}")
        if corr < 0:
            print("🔁 Flipping sign of PC1 so that higher = more isolated (like iso_index).")
            scores_z = -scores_z

    df = df.copy()
    df["iso_index_pca"] = scores_z
    return df


def make_comparison_map(
    df: pd.DataFrame,
    wards_geojson: str | Path,
    out_path: str | Path,
) -> None:
    """Plot side-by-side choropleths: original iso_index vs PCA index."""
    wards = gpd.read_file(wards_geojson)

    # GeoJSON key for JIS ward code (from earlier work: N03_007)
    if "N03_007" not in wards.columns:
        raise KeyError(
            "Expected 'N03_007' column in wards GeoJSON. "
            "Inspect the file to confirm the correct code column."
        )

    wards = wards.copy()
    # Normalize join keys: ward_jis is numeric like 13101
    wards["ward_jis"] = wards["N03_007"].astype(str).astype(int)

    merged = wards.merge(df, on="ward_jis", how="inner")
    if merged.empty:
        raise ValueError(
            "Merge between wards GeoJSON and index data is empty. "
            "Check that ward_jis and N03_007 codes line up."
        )

    print(f"✅ Merged {len(merged)} wards for mapping.")

    # Common color scale for both maps
    vmin = float(
        np.nanmin(merged[["iso_index", "iso_index_pca"]].to_numpy())
    )
    vmax = float(
        np.nanmax(merged[["iso_index", "iso_index_pca"]].to_numpy())
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 8))

    ax1, ax2 = axes

    merged.plot(
        column="iso_index",
        ax=ax1,
        cmap="viridis",
        linewidth=0.5,
        edgecolor="black",
        vmin=vmin,
        vmax=vmax,
        legend=True,
    )
    ax1.set_title("Designed Isolation Index", fontsize=10)
    ax1.axis("off")

    merged.plot(
        column="iso_index_pca",
        ax=ax2,
        cmap="viridis",
        linewidth=0.5,
        edgecolor="black",
        vmin=vmin,
        vmax=vmax,
        legend=True,
    )
    ax2.set_title("PCA-based Isolation Index", fontsize=10)
    ax2.axis("off")

    plt.tight_layout()
    ensure_dirs(out_path)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"🗺 Saved comparison map to {out_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute PCA-based isolation index and comparison map."
    )
    parser.add_argument(
        "--index-path",
        default="data/processed/jp_tokyo_index.csv",
        help="Input CSV with iso_index and predictor columns.",
    )
    parser.add_argument(
        "--wards-geojson",
        default="data/external/jp_tokyo_wards.geojson",
        help="GeoJSON with 23-ward boundaries.",
    )
    parser.add_argument(
        "--out-csv",
        default="data/processed/jp_tokyo_index_pca.csv",
        help="Output CSV with iso_index_pca column.",
    )
    parser.add_argument(
        "--out-map",
        default="out/maps/tokyo_iso_index_comparison.png",
        help="Output PNG for side-by-side map.",
    )

    args = parser.parse_args()

    print(f"📥 Loading index from {args.index_path} ...")
    df = pd.read_csv(args.index_path)

    # Make sure ward_jis exists and is integer
    if "ward_jis" not in df.columns:
        raise KeyError("Expected 'ward_jis' column in index CSV.")
    df["ward_jis"] = df["ward_jis"].astype(int)

    df_pca = compute_pca_index(df)

    # Print quick top/bottom 5
    print("\nTop 5 wards by PCA isolation index:")
    top5 = (
        df_pca.sort_values("iso_index_pca", ascending=False)
        [["ward_jis", "ward_name", "iso_index_pca"]]
        .head(5)
    )
    for _, row in top5.iterrows():
        print(f"- {row['ward_jis']} {row['ward_name']}: iso_index_pca = {row['iso_index_pca']:.3f}")

    print("\nBottom 5 wards by PCA isolation index:")
    bottom5 = (
        df_pca.sort_values("iso_index_pca", ascending=True)
        [["ward_jis", "ward_name", "iso_index_pca"]]
        .head(5)
    )
    for _, row in bottom5.iterrows():
        print(f"- {row['ward_jis']} {row['ward_name']}: iso_index_pca = {row['iso_index_pca']:.3f}")

    # Save CSV
    ensure_dirs(args.out_csv)
    df_pca.to_csv(args.out_csv, index=False)
    print(f"\n💾 Saved PCA index CSV to {args.out_csv}")

    # Make map
    make_comparison_map(df_pca, args.wards_geojson, args.out_map)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
