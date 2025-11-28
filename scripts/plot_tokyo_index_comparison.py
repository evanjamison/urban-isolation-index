# scripts/plot_tokyo_index_comparison.py
"""
plot_tokyo_index_comparison.py

Compare the designed isolation index (iri_designed) and PCA-based index (iri_pca)
on side-by-side choropleth maps.

Usage (PowerShell example):

  .\.venv\Scripts\python.exe scripts/plot_tokyo_index_comparison.py `
      --index-path data/processed/jp_tokyo_with_designed_pca.csv `
      --wards-geojson data/external/jp_tokyo_wards.geojson `
      --out-path out/spatial_tokyo/tokyo_diri_vs_pca.png
"""

import argparse
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


def ensure_parent_dir(path: str) -> None:
    """Create parent directory for a file if it does not exist."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--index-path",
        default="data/processed/jp_tokyo_with_designed_pca.csv",
        help="CSV with iri_designed and iri_pca per ward",
    )
    # support both the correct spelling and the old typo for convenience
    p.add_argument(
        "--wards-geojson",
        "--wards-geosjon",
        dest="wards_geojson",
        default="data/external/jp_tokyo_wards.geojson",
        help="GeoJSON file with Tokyo ward boundaries",
    )
    p.add_argument(
        "--out-path",
        default="out/spatial_tokyo/tokyo_diri_vs_pca.png",
        help="Output PNG path for comparison map",
    )
    args = p.parse_args()

    # ------------------------------------------------------------------
    # 1) Load index data
    # ------------------------------------------------------------------
    print(f"📥 Loading index data from {args.index_path} ...")
    df = pd.read_csv(args.index_path)

    required_cols = ["ward_jis", "iri_designed", "iri_pca"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {args.index_path}")

    # standardize ward_jis type
    df["ward_jis"] = df["ward_jis"].astype(int)

    # ------------------------------------------------------------------
    # 2) Load GeoJSON and ensure ward ID column is present
    # ------------------------------------------------------------------
    print(f"📥 Loading wards GeoJSON from {args.wards_geojson} ...")
    wards = gpd.read_file(args.wards_geojson)

    # Your GeoJSON uses N03_007 as the ward code; rename if needed
    if "ward_jis" not in wards.columns:
        if "N03_007" in wards.columns:
            wards = wards.rename(columns={"N03_007": "ward_jis"})
            print("ℹ️  Renamed 'N03_007' → 'ward_jis' in GeoJSON.")
        else:
            raise ValueError(
                "GeoJSON must contain a ward identifier column "
                "('ward_jis' or 'N03_007')."
            )

    wards["ward_jis"] = wards["ward_jis"].astype(int)

    # ------------------------------------------------------------------
    # 3) Merge geometries with index data
    # ------------------------------------------------------------------
    print("🔗 Merging geometries with index data on ward_jis ...")
    gdf = wards.merge(df, on="ward_jis", how="left")

    if gdf["iri_designed"].isna().any() or gdf["iri_pca"].isna().any():
        missing = gdf[gdf["iri_designed"].isna() | gdf["iri_pca"].isna()][
            ["ward_jis"]
        ]
        print("⚠ Warning: some wards missing index values after merge:")
        print(missing)

    # ------------------------------------------------------------------
    # 4) Plot side-by-side comparison
    # ------------------------------------------------------------------
    vmin = min(gdf["iri_designed"].min(), gdf["iri_pca"].min())
    vmax = max(gdf["iri_designed"].max(), gdf["iri_pca"].max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)

    # Designed index map
    gdf.plot(
        column="iri_designed",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        linewidth=0.4,
        edgecolor="black",
        ax=axes[0],
    )
    axes[0].set_title("Tokyo Designed Isolation Index (D-IRI)")
    axes[0].axis("off")

    # PCA-based index map
    gdf.plot(
        column="iri_pca",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        linewidth=0.4,
        edgecolor="black",
        ax=axes[1],
    )
    axes[1].set_title("Tokyo PCA-based Isolation Index")
    axes[1].axis("off")

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_clim(vmin, vmax)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Standardized isolation score")

    # ------------------------------------------------------------------
    # 5) Save
    # ------------------------------------------------------------------
    ensure_parent_dir(args.out_path)
    plt.savefig(args.out_path, dpi=300)
    plt.close(fig)

    print(f"🗺️ Saved comparison map to {args.out_path}")
    print("✅ plot_tokyo_index_comparison completed.")


if __name__ == "__main__":
    main()

