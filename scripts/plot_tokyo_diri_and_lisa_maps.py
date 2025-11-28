"""
plot_tokyo_diri_and_lisa_maps.py

Create final maps for the designed isolation index:
- Left: continuous D-IRI choropleth
- Right: LISA clusters (High-High, Low-Low, etc.)

Usage (from repo root):

  .\.venv\Scripts\python.exe scripts/plot_tokyo_diri_and_lisa_maps.py ^
      --index-path data/processed/jp_tokyo_with_designed_pca.csv ^
      --wards-geojson data/external/jp_tokyo_wards.geojson ^
      --out-path out/spatial_tokyo/tokyo_diri_final_maps.png
"""

import argparse
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda.moran import Moran, Moran_Local
from libpysal.weights import Queen
from matplotlib.lines import Line2D


def ensure_parent_dir(path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def classify_lisa(local_moran: Moran_Local, p_thresh: float = 0.05) -> np.ndarray:
    """
    Return categorical labels for LISA clusters based on quadrant & p-value.

    Quadrants follow:
      1: High-High
      2: Low-High
      3: Low-Low
      4: High-Low
    """
    q = local_moran.q
    p = local_moran.p_sim

    labels = np.full_like(q, fill_value="Not significant", dtype=object)

    sig = p < p_thresh
    labels[(q == 1) & sig] = "High-High"
    labels[(q == 2) & sig] = "Low-High"
    labels[(q == 3) & sig] = "Low-Low"
    labels[(q == 4) & sig] = "High-Low"

    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Plot D-IRI choropleth and LISA cluster map for Tokyo."
    )
    parser.add_argument(
        "--index-path",
        default="data/processed/jp_tokyo_with_designed_pca.csv",
        help="CSV with ward_jis and iri_designed columns",
    )
    parser.add_argument(
        "--wards-geojson",
        default="data/external/jp_tokyo_wards.geojson",
        help="Tokyo wards GeoJSON (with N03_007 codes)",
    )
    parser.add_argument(
        "--out-path",
        default="out/spatial_tokyo/tokyo_diri_final_maps.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--p-thresh",
        type=float,
        default=0.05,
        help="Significance threshold for LISA (default 0.05)",
    )
    args = parser.parse_args()

    # --------------------------------------------------------------
    # Load and merge data
    # --------------------------------------------------------------
    print(f"📥 Loading index data from {args.index_path} ...")
    df = pd.read_csv(args.index_path)

    for col in ["ward_jis", "iri_designed"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {args.index_path}")

    # Harmonize ward IDs as 5-digit strings
    df["ward_jis"] = df["ward_jis"].astype(str).str.strip().str.zfill(5)

    print(f"📥 Loading wards GeoJSON from {args.wards_geojson} ...")
    wards = gpd.read_file(args.wards_geojson)

    # GeoJSON has N03_007 = municipality JIS code
    if "N03_007" not in wards.columns:
        raise ValueError("GeoJSON must contain 'N03_007' column (ward code).")

    wards["N03_007"] = wards["N03_007"].astype(str).str.strip().str.zfill(5)

    print("🔗 Merging geometries with index data (N03_007 ↔ ward_jis) ...")
    gdf = wards.merge(
        df[["ward_jis", "iri_designed"]],
        left_on="N03_007",
        right_on="ward_jis",
        how="left",
    )

    # Keep only polygons that actually have an index value
    missing_before = gdf["iri_designed"].isna().sum()
    if missing_before:
        print(f"⚠ Warning: {missing_before} polygons missing iri_designed; dropping them.")
        gdf = gdf[~gdf["iri_designed"].isna()].copy()

    gdf = gdf.set_index("ward_jis")

    # --------------------------------------------------------------
    # Global Moran's I
    # --------------------------------------------------------------
    print("\n🧪 Computing global Moran's I for iri_designed ...")
    w = Queen.from_dataframe(gdf)
    w.transform = "r"

    y = gdf["iri_designed"].values
    moran = Moran(y, w)

    print("Global Moran's I results")
    print("-" * 40)
    print(f"  I statistic  : {moran.I:.4f}")
    print(f"  E[I]         : {moran.EI:.4f}")
    print(f"  z-score      : {moran.z_norm:.3f}")
    print(f"  p-value(norm): {moran.p_norm:.4f}")

    # --------------------------------------------------------------
    # Local Moran's I (LISA)
    # --------------------------------------------------------------
    print("\n🧪 Computing Local Moran's I (LISA) for iri_designed ...")
    lisa = Moran_Local(y, w)
    labels = classify_lisa(lisa, p_thresh=args.p_thresh)
    gdf["lisa_cluster"] = labels

    cluster_counts = gdf["lisa_cluster"].value_counts()
    print("\nLISA cluster counts (p < {:.2f})".format(args.p_thresh))
    print("-" * 40)
    print(cluster_counts)

    # --------------------------------------------------------------
    # Plot: D-IRI map + LISA cluster map
    # --------------------------------------------------------------
    print("\n🎨 Building final maps ...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)

    # Continuous D-IRI choropleth
    vmin = gdf["iri_designed"].min()
    vmax = gdf["iri_designed"].max()
    gdf.plot(
        column="iri_designed",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        linewidth=0.4,
        edgecolor="black",
        ax=axes[0],
        legend=True,
    )
    axes[0].set_title("Tokyo Designed Isolation Index (D-IRI)")
    axes[0].axis("off")

    # LISA cluster map
    color_map = {
        "High-High": "#d7191c",
        "Low-Low": "#2c7bb6",
        "Low-High": "#fdae61",
        "High-Low": "#abdda4",
        "Not significant": "#e0e0e0",
    }
    gdf["lisa_color"] = gdf["lisa_cluster"].map(color_map)

    gdf.plot(
        color=gdf["lisa_color"],
        linewidth=0.4,
        edgecolor="black",
        ax=axes[1],
    )
    axes[1].set_title("LISA Clusters (iri_designed)")
    axes[1].axis("off")

    # Build custom legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=lab,
            markerfacecolor=color_map[lab],
            markersize=10,
        )
        for lab in ["High-High", "Low-Low", "Low-High", "High-Low", "Not significant"]
    ]
    axes[1].legend(
        handles=legend_elements,
        loc="lower left",
        title="Cluster type",
        frameon=True,
    )

    ensure_parent_dir(args.out_path)
    plt.savefig(args.out_path, dpi=300)
    plt.close(fig)

    print(f"🗺️ Final D-IRI + LISA maps saved to {args.out_path}")
    print("✅ plot_tokyo_diri_and_lisa_maps completed.")


if __name__ == "__main__":
    main()
