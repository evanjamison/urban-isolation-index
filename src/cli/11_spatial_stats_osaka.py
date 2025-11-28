"""
11_spatial_stats_osaka.py

Compute spatial statistics for Osaka's Designed Isolation Index (D-IRI):
- Global Moran's I (spatial autocorrelation)
- Local Moran's I (LISA) with cluster classification
- Save results to CSV
- Save hotspot map PNG

Usage (from project root):

  .\.venv\Scripts\python.exe -m src.cli.11_spatial_stats_osaka `
    --index-path data/processed/jp_osaka_with_designed.csv `
    --wards-geojson data/external/jp_osaka_wards.geojson `
    --out-dir out/spatial_osaka
"""

import argparse
import os

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local


def load_and_merge(index_path: str, wards_geojson: str) -> gpd.GeoDataFrame:
    """Load D-IRI table and wards GeoJSON, merge on ward code."""
    print("📥 Loading index from", index_path, "...")
    df = pd.read_csv(index_path)

    # Expect columns: ward_jis, ward_name, iri_designed, ...
    if "ward_jis" not in df.columns:
        raise ValueError("Expected 'ward_jis' column in index dataset.")

    if "iri_designed" not in df.columns:
        raise ValueError("Expected 'iri_designed' column (D-IRI) in index dataset.")

    print("📥 Loading wards GeoJSON from", wards_geojson, "...")
    wards = gpd.read_file(wards_geojson)

    # GeoJSON uses N03_007 as the ward code (same pattern as Tokyo)
    ward_code_col = "N03_007"
    if ward_code_col not in wards.columns:
        raise ValueError(f"Expected '{ward_code_col}' column in GeoJSON.")

    # Ensure consistent types for join
    wards[ward_code_col] = wards[ward_code_col].astype(str)
    df["ward_jis"] = df["ward_jis"].astype(str)

    print("🔗 Merging index with geometries on ward_jis ...")
    gdf = wards.merge(df, left_on=ward_code_col, right_on="ward_jis", how="inner")

    if gdf.empty:
        raise ValueError("Merge produced no rows – check ward codes or files.")

    print(f"✅ Merged GeoDataFrame shape: {gdf.shape}")
    return gdf


def compute_global_moran(gdf: gpd.GeoDataFrame, value_col: str = "iri_designed"):
    """Compute global Moran's I for D-IRI."""
    print("\n🌍 Computing global Moran's I for", value_col, "...")
    w = Queen.from_dataframe(gdf)
    w.transform = "R"

    y = gdf[value_col].values
    moran = Moran(y, w)

    print("Global Moran's I results")
    print("-" * 40)
    print(f"I statistic      : {moran.I:.4f}")
    print(f"Expected I (E[I]): {moran.EI:.4f}")
    print(f"Z-score          : {moran.z_norm:.4f}")
    print(f"p-value (norm)   : {moran.p_norm:.4f}")
    print()

    return moran


def classify_lisa(gdf: gpd.GeoDataFrame, lisa: Moran_Local,
                  p_thresh: float = 0.05) -> gpd.GeoDataFrame:
    """Attach LISA results (local Moran) and cluster labels to GeoDataFrame."""
    print("🧩 Attaching Local Moran's I (LISA) results ...")

    gdf = gdf.copy()
    gdf["lisa_I"] = lisa.Is
    gdf["lisa_p"] = lisa.p_sim
    gdf["lisa_q"] = lisa.q

    # Cluster classification (mirrors your Tokyo script’s scheme)
    def _cluster_label(p, q):
        if p >= p_thresh:
            return "Not significant"
        if q == 1:
            return "High-High"
        if q == 2:
            return "Low-Low"
        if q == 3:
            return "Low-High"
        if q == 4:
            return "High-Low"
        return "Not significant"

    gdf["lisa_cluster"] = [
        _cluster_label(p, q) for p, q in zip(gdf["lisa_p"], gdf["lisa_q"])
    ]

    print("\nCluster counts (p <", p_thresh, ")")
    print("-" * 40)
    print(gdf["lisa_cluster"].value_counts())
    print()

    return gdf


def compute_local_moran(gdf: gpd.GeoDataFrame,
                        value_col: str = "iri_designed") -> gpd.GeoDataFrame:
    """Compute Local Moran (LISA) for D-IRI."""
    print("📍 Computing Local Moran's I (LISA) for", value_col, "...")
    w = Queen.from_dataframe(gdf)
    w.transform = "R"

    y = gdf[value_col].values
    lisa = Moran_Local(y, w)

    gdf_with_lisa = classify_lisa(gdf, lisa)
    return gdf_with_lisa


def save_results(gdf: gpd.GeoDataFrame, out_dir: str):
    """Save LISA table and cluster map for Osaka."""
    os.makedirs(out_dir, exist_ok=True)

    # 1) Save table
    out_csv = os.path.join(out_dir, "osaka_diri_lisa_results.csv")
    cols_to_save = [
        "ward_jis",
        "ward_name",
        "iri_designed",
        "station_count",
        "transit_z",
        "lisa_I",
        "lisa_p",
        "lisa_q",
        "lisa_cluster",
    ]
    cols_present = [c for c in cols_to_save if c in gdf.columns]
    gdf[cols_present].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"💾 Saved LISA results table to {out_csv}")

    # 2) Save hotspot map
    out_png = os.path.join(out_dir, "osaka_diri_lisa_clusters.png")
    print("🗺  Saving LISA cluster map to", out_png, "...")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_axis_off()
    ax.set_title("Osaka Designed Isolation Index (D-IRI)\nLISA Cluster Map", fontsize=14)

    # Define fixed colors for clusters
    cluster_colors = {
        "High-High": "#d7191c",       # red
        "Low-Low": "#2c7bb6",         # blue
        "Low-High": "#abd9e9",        # light blue
        "High-Low": "#fdae61",        # orange
        "Not significant": "#dddddd", # light gray
    }

    # Map cluster labels to colors
    gdf["cluster_color"] = gdf["lisa_cluster"].map(cluster_colors)

    gdf.plot(
        color=gdf["cluster_color"],
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
    )

    # Make a legend manually
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=label,
               markerfacecolor=color, markersize=10)
        for label, color in cluster_colors.items()
    ]
    ax.legend(handles=legend_elements, loc="lower left", title="Cluster type")

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print("✅ LISA cluster map saved.")


def main():
    p = argparse.ArgumentParser(
        description="Spatial statistics for Osaka D-IRI (Moran's I + LISA)."
    )
    p.add_argument(
        "--index-path",
        required=True,
        help="CSV file with ward_jis, ward_name, iri_designed, etc.",
    )
    p.add_argument(
        "--wards-geojson",
        required=True,
        help="GeoJSON file for Osaka wards (e.g., data/external/jp_osaka_wards.geojson).",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for spatial stats tables and maps.",
    )
    args = p.parse_args()

    gdf = load_and_merge(args.index_path, args.wards_geojson)

    # Global Moran's I
    _ = compute_global_moran(gdf, value_col="iri_designed")

    # Local Moran & clustering
    gdf_lisa = compute_local_moran(gdf, value_col="iri_designed")

    # Save CSV + map
    save_results(gdf_lisa, args.out_dir)

    print("\n🎉 Osaka spatial statistics pipeline completed.")


if __name__ == "__main__":
    main()
