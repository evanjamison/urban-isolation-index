# scripts/plot_tokyo_difference_map.py
"""
Plot the difference between Designed IRI and PCA-based IRI for Tokyo wards.
D-IRI minus PCA-IRI (positive = designed > PCA).
"""

import argparse
import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


def ensure_parent(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index-path", default="data/processed/jp_tokyo_with_designed_pca.csv")
    p.add_argument("--wards-geojson", default="data/external/jp_tokyo_wards.geojson")
    p.add_argument("--out-path", default="out/spatial_tokyo/tokyo_diri_minus_pca.png")
    args = p.parse_args()

    print("📥 Loading index data…")
    df = pd.read_csv(args.index_path)
    df["ward_jis"] = df["ward_jis"].astype(str)

    print("📥 Loading wards GeoJSON…")
    wards = gpd.read_file(args.wards_geojson)
    wards["N03_007"] = wards["N03_007"].astype(str)
    wards = wards.rename(columns={"N03_007": "ward_jis"})

    print("🔗 Merging geometry with indices…")
    gdf = wards.merge(df, on="ward_jis", how="left")

    print("🧮 Computing D-IRI – PCA difference…")
    gdf["iri_diff"] = gdf["iri_designed"] - gdf["iri_pca"]

    v = max(abs(gdf["iri_diff"].min()), abs(gdf["iri_diff"].max()))

    print("🎨 Plotting…")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    gdf.plot(
        column="iri_diff",
        cmap="RdBu_r",
        vmin=-v,
        vmax=v,
        linewidth=0.4,
        edgecolor="black",
        legend=True,
        ax=ax,
    )
    ax.set_title("Tokyo: D-IRI minus PCA-IRI\n(Positive = Designed > PCA)", fontsize=14)
    ax.axis("off")

    ensure_parent(args.out_path)
    plt.savefig(args.out_path, dpi=300)
    plt.close()
    print(f"💾 Saved to {args.out_path}")


if __name__ == "__main__":
    main()
