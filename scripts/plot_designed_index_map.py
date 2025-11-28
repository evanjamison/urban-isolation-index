"""
plot_designed_index_map.py
Make choropleth maps of the Designed Isolation Index (D-IRI) for Tokyo wards:
  - z-score version (iri_designed)
  - 0–100 normalized version (iri_designed_100)

Usage (from repo root, Windows/PowerShell):

  .\.venv\Scripts\python.exe scripts/plot_designed_index_map.py `
    --index-path data/processed/jp_tokyo_indices_normalized.csv `
    --wards-geojson data/external/jp_tokyo_wards.geojson `
    --out-path out/plots/tokyo_designed_index_both.png
"""

import argparse
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    # ---------- CLI ----------
    p = argparse.ArgumentParser(
        description=(
            "Plot Designed Isolation Index (D-IRI) for Tokyo wards "
            "as both z-scores and 0–100 normalized values."
        )
    )
    p.add_argument(
        "--index-path",
        required=True,
        help="CSV with ward_jis, ward_name, iri_designed and iri_designed_100 columns",
    )
    p.add_argument(
        "--wards-geojson",
        required=True,
        help="GeoJSON with Tokyo ward boundaries (jp_tokyo_wards.geojson)",
    )
    p.add_argument(
        "--out-path",
        required=True,
        help="Output PNG path for the combined map",
    )
    args = p.parse_args()

    # ---------- Load index data ----------
    print("📥 Loading designed index from", args.index_path, "...")
    df = pd.read_csv(args.index_path)

    # Basic sanity checks
    for col in ("ward_jis", "ward_name", "iri_designed", "iri_designed_100"):
        if col not in df.columns:
            raise ValueError(f"Expected '{col}' column in index dataset.")

    # Make sure IDs are strings so they match the GeoJSON codes
    df["ward_jis"] = df["ward_jis"].astype(str)

    print("\n📊 Summary of iri_designed (z-score):")
    print(df["iri_designed"].describe())

    print("\n📊 Summary of iri_designed_100 (0–100):")
    print(df["iri_designed_100"].describe())

    # ---------- Load ward polygons ----------
    print("\n📥 Loading ward boundaries from", args.wards_geojson, "...")
    wards = gpd.read_file(args.wards_geojson)

    # The Tokyo wards GeoJSON built earlier has the code in N03_007
    if "N03_007" not in wards.columns:
        raise ValueError("Expected 'N03_007' column in wards GeoJSON for ward codes.")

    wards["N03_007"] = wards["N03_007"].astype(str)

    # ---------- Merge polygons with index ----------
    print("\n🔗 Merging wards with designed index on ward code (N03_007 ↔ ward_jis) ...")
    merged = wards.merge(
        df[["ward_jis", "ward_name", "iri_designed", "iri_designed_100"]],
        left_on="N03_007",
        right_on="ward_jis",
        how="inner",
    )

    if merged.empty:
        raise RuntimeError("Merge produced no rows – check that ward codes match.")

    print("✅ Merge complete. Rows:", len(merged))

    # ---------- Plotting ----------
    # Two subplots: left = z-score, right = 0–100
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Common plotting kwargs
    common_kwargs = dict(
        cmap="viridis",
        linewidth=0.5,
        edgecolor="black",
        legend=True,
    )

    # Left: z-score map
    merged.plot(
        column="iri_designed",
        ax=axes[0],
        **common_kwargs,
    )
    axes[0].set_title("Tokyo D-IRI (z-score)", fontsize=13)
    axes[0].axis("off")

    # Right: 0–100 map (force vmin/vmax for clean color scale)
    merged.plot(
        column="iri_designed_100",
        ax=axes[1],
        vmin=0,
        vmax=100,
        **common_kwargs,
    )
    axes[1].set_title("Tokyo D-IRI (0–100 normalized)", fontsize=13)
    axes[1].axis("off")

    plt.tight_layout()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(args.out_path, dpi=300)
    plt.close(fig)

    print(f"💾 Saved combined D-IRI maps (z-score + 0–100) to {args.out_path}")


if __name__ == "__main__":
    main()
