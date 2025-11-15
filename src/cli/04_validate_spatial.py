"""
04_validate_spatial.py

Validate Tokyo spatial mapping by building a choropleth of the isolation index.
- Reads processed Tokyo isolation index CSV
- Reads Tokyo wards GeoJSON
- Harmonizes ward codes (5-digit strings)
- Produces choropleth and saves to out/maps/tokyo_iso_index.png
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


def load_data(index_path: str, wards_geojson: str):
    """Load the isolation index table and ward boundary GeoJSON."""
    # Isolation index CSV
    df = pd.read_csv(index_path)

    # GeoJSON
    wards = gpd.read_file(wards_geojson)

    print("GeoJSON columns:", list(wards.columns))

    # --- Harmonize ward codes to 5-digit strings ---
    # ward_jis from CSV
    if "ward_jis" not in df.columns:
        raise KeyError("Index CSV is missing 'ward_jis' column.")

    df["ward_jis"] = (
        df["ward_jis"]
        .astype(str)
        .str.strip()
        .str.zfill(5)
    )

    # N03_007 from GeoJSON = municipality JIS
    if "N03_007" not in wards.columns:
        raise KeyError("GeoJSON is missing 'N03_007' column.")

    wards["N03_007"] = (
        wards["N03_007"]
        .astype(str)
        .str.strip()
        .str.zfill(5)
    )

    print("\nSample ward_jis from CSV:", df["ward_jis"].head().tolist())
    print("Sample N03_007 from GeoJSON:", wards["N03_007"].head().tolist())

    return df, wards


def build_tokyo_choropleth(df: pd.DataFrame, wards: gpd.GeoDataFrame):
    """
    Merge isolation index with ward polygons and build choropleth.
    """

    # Ensure CRS is set and projected sensibly
    if wards.crs is None:
        # Most Japanese GeoJSON downloads are in WGS84
        wards = wards.set_crs("EPSG:4326")

    # Project to a Japan local CRS (Tokyo) for nicer geometry
    wards = wards.to_crs("EPSG:2443")

    # Merge on harmonized codes
    merged = wards.merge(df, left_on="N03_007", right_on="ward_jis", how="inner")

    print("\nMerged preview:")
    print(merged[["ward_jis", "ward_name", "iso_index"]].head())

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    merged.plot(
        column="iso_index",
        cmap="viridis",
        legend=True,
        linewidth=0.5,
        edgecolor="black",
        ax=ax,
    )

    ax.set_title("Tokyo Isolation Index (z-score)")
    ax.set_axis_off()
    plt.tight_layout()

    out_dir = Path("out/maps")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tokyo_iso_index.png"

    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"\n✅ Saved choropleth to {out_path}\n")
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Validate spatial mapping by plotting Tokyo isolation index."
    )
    parser.add_argument(
        "--index-path",
        default="data/processed/jp_tokyo_index.csv",
        help="CSV file containing ward_jis, iso_index, etc.",
    )
    parser.add_argument(
        "--wards-geojson",
        default="data/external/jp_tokyo_wards.geojson",
        help="GeoJSON boundary file for Tokyo wards.",
    )
    args = parser.parse_args()

    df, wards = load_data(args.index_path, args.wards_geojson)
    build_tokyo_choropleth(df, wards)


if __name__ == "__main__":
    main()
