# scripts/plot_tokyo_pca_map.py
r"""
plot_tokyo_pca_map.py

Plot a choropleth map of the PCA-based isolation index for Tokyo wards (iri_pca),
with the SAME ward labeling style as plot_designed_index_map.py:

  新宿区 (Shinjuku-ku)

Also:
- Japanese font configured (Windows) so Kanji renders
- Bold labels + white halo for readability
- Manual centroid override for 江東区 / 江戸川区
- Manual nudges to separate 墨田区 / 台東区 labels

Usage (from project root, PowerShell):

  py scripts/plot_tokyo_pca_map.py `
    --index-path data/processed/jp_tokyo_with_designed_pca.csv `
    --wards-geojson data/external/jp_tokyo_wards.geojson `
    --out-path out/plots/tokyo_pca_index_labeled.png

Optional: match color scaling to Designed index (for consistent look):
  py scripts/plot_tokyo_pca_map.py `
    --index-path data/processed/jp_tokyo_with_designed_pca.csv `
    --designed-path data/processed/jp_tokyo_with_designed.csv `
    --wards-geojson data/external/jp_tokyo_wards.geojson `
    --out-path out/plots/tokyo_pca_index_labeled_matched.png
"""

import argparse
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import font_manager, rcParams
import matplotlib.patheffects as path_effects


# --------------------------------------------------
# Configure Japanese font (Windows-safe)
# --------------------------------------------------
JP_FONT_PATHS = [
    r"C:\Windows\Fonts\meiryo.ttc",     # preferred
    r"C:\Windows\Fonts\yugothib.ttf",
    r"C:\Windows\Fonts\msgothic.ttc",
]

jp_font = None
for fp in JP_FONT_PATHS:
    if os.path.exists(fp):
        jp_font = font_manager.FontProperties(fname=fp)
        break

if jp_font is None:
    raise RuntimeError("No Japanese font found. Install Meiryo or Yu Gothic.")

rcParams["font.family"] = jp_font.get_name()


# --------------------------------------------------
# Tokyo ward Romaji
# --------------------------------------------------
TOKYO_WARD_ROMAJI = {
    "千代田区": "Chiyoda-ku",
    "中央区": "Chuo-ku",
    "港区": "Minato-ku",
    "新宿区": "Shinjuku-ku",
    "文京区": "Bunkyo-ku",
    "台東区": "Taito-ku",
    "墨田区": "Sumida-ku",
    "江東区": "Koto-ku",
    "品川区": "Shinagawa-ku",
    "目黒区": "Meguro-ku",
    "大田区": "Ota-ku",
    "世田谷区": "Setagaya-ku",
    "渋谷区": "Shibuya-ku",
    "中野区": "Nakano-ku",
    "杉並区": "Suginami-ku",
    "豊島区": "Toshima-ku",
    "北区": "Kita-ku",
    "荒川区": "Arakawa-ku",
    "板橋区": "Itabashi-ku",
    "練馬区": "Nerima-ku",
    "足立区": "Adachi-ku",
    "葛飾区": "Katsushika-ku",
    "江戸川区": "Edogawa-ku",
}


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def extract_kanji_name(label: str) -> str:
    """From '江東区 (Koto-ku)' -> '江東区'."""
    return str(label).split(" (", 1)[0].strip()


def add_ward_labels(gdf: gpd.GeoDataFrame, ax, label_col: str, fontsize: int = 8) -> None:
    """
    Label wards using representative_point() by default,
    with centroids for Koto/Edogawa and small nudges for Sumida/Taito.
    """
    pts = gdf.geometry.representative_point()

    overrides = {
        "江東区": None,
        "江戸川区": None,
    }

    nudges = {
        "墨田区": (0.004, 0.002),    # Sumida-ku
        "台東区": (-0.004, -0.002),  # Taito-ku
    }

    # Centroids (projected) for better "visual center"
    gdf_proj = gdf.to_crs(3857)
    centroids = gdf_proj.geometry.centroid.to_crs(gdf.crs)

    for i, disp in enumerate(gdf[label_col].astype(str)):
        kanji = extract_kanji_name(disp)
        if kanji in overrides:
            overrides[kanji] = centroids.iloc[i]

    for i, disp in enumerate(gdf[label_col].astype(str)):
        kanji = extract_kanji_name(disp)

        p = overrides.get(kanji) or pts.iloc[i]
        x, y = p.x, p.y

        dx, dy = nudges.get(kanji, (0.0, 0.0))
        x += dx
        y += dy

        txt = ax.text(
            x,
            y,
            disp,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="black",
        )

        txt.set_path_effects(
            [
                path_effects.Stroke(linewidth=2.8, foreground="white"),
                path_effects.Normal(),
            ]
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--index-path",
        default="data/processed/jp_tokyo_with_designed_pca.csv",
        help="CSV with PCA index per ward (requires: ward_jis, iri_pca; optional: ward_name)",
    )
    p.add_argument(
        "--wards-geojson",
        "--wards-geosjon",
        dest="wards_geojson",
        default="data/external/jp_tokyo_wards.geojson",
        help="GeoJSON file with Tokyo ward boundaries",
    )
    p.add_argument(
        "--out-path",
        default="out/plots/tokyo_pca_index_labeled.png",
        help="Output PNG path for PCA map",
    )
    p.add_argument(
        "--title",
        default="Tokyo PCA-based Isolation Index",
        help="Plot title",
    )
    p.add_argument(
        "--label-fontsize",
        type=int,
        default=8,
        help="Ward label font size (default: 8)",
    )
    p.add_argument(
        "--designed-path",
        default=None,
        help="Optional Designed index CSV (with iri_designed) to share vmin/vmax with PCA for consistent color scaling.",
    )
    args = p.parse_args()

    # ------------------------------------------------------------
    # 1) Load PCA index data
    # ------------------------------------------------------------
    print(f"📥 Loading index data from {args.index_path} ...")
    df = pd.read_csv(args.index_path)

    for col in ("ward_jis", "iri_pca"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {args.index_path}. Found: {list(df.columns)}")

    # Use STRING join keys (more robust than int casting)
    df["ward_jis"] = df["ward_jis"].astype(str).str.strip()

    # ------------------------------------------------------------
    # 2) Load ward polygons
    # ------------------------------------------------------------
    print(f"📥 Loading wards GeoJSON from {args.wards_geojson} ...")
    wards = gpd.read_file(args.wards_geojson)

    # Normalize ward code column to "ward_jis"
    if "ward_jis" not in wards.columns:
        if "N03_007" in wards.columns:
            wards = wards.rename(columns={"N03_007": "ward_jis"})
            print("ℹ️  Renamed 'N03_007' → 'ward_jis' in GeoJSON.")
        else:
            raise ValueError(
                "GeoJSON must contain a ward identifier column ('ward_jis' or 'N03_007'). "
                f"Found: {list(wards.columns)}"
            )

    wards["ward_jis"] = wards["ward_jis"].astype(str).str.strip()

    # Try to find a ward-name column inside GeoJSON if PCA CSV doesn't have ward_name
    ward_name_col = None
    for cand in ("ward_name", "N03_004", "NAME", "name"):
        if cand in wards.columns:
            ward_name_col = cand
            break

    # ------------------------------------------------------------
    # 3) Merge
    # ------------------------------------------------------------
    print("🔗 Merging geometries with PCA index on ward_jis ...")
    gdf = wards.merge(df, on="ward_jis", how="left")

    if gdf["iri_pca"].isna().any():
        missing = gdf[gdf["iri_pca"].isna()][["ward_jis"]].sort_values("ward_jis")
        print("⚠️ Warning: some wards are missing iri_pca after merge:")
        print(missing.to_string(index=False))

    # ------------------------------------------------------------
    # 4) Build labels like: 新宿区 (Shinjuku-ku)
    # Prefer ward_name from CSV; else use GeoJSON ward name
    # ------------------------------------------------------------
    if "ward_name" in gdf.columns:
        wn = gdf["ward_name"].astype(str).str.strip()
    elif ward_name_col is not None:
        wn = gdf[ward_name_col].astype(str).str.strip()
    else:
        # Worst-case fallback: no Kanji available; just show ward_jis
        wn = gdf["ward_jis"].astype(str)

    gdf["_ward_display"] = wn.apply(
        lambda k: f"{k} ({TOKYO_WARD_ROMAJI[k]})" if k in TOKYO_WARD_ROMAJI else k
    )

    # ------------------------------------------------------------
    # 5) Color scaling (optional shared with Designed index)
    # ------------------------------------------------------------
    vmin = pd.to_numeric(gdf["iri_pca"], errors="coerce").min()
    vmax = pd.to_numeric(gdf["iri_pca"], errors="coerce").max()

    if args.designed_path:
        print(f"📥 Loading designed index for shared scale: {args.designed_path} ...")
        ddf = pd.read_csv(args.designed_path)
        if "iri_designed" not in ddf.columns:
            raise ValueError("Designed CSV missing 'iri_designed'.")
        dmin = pd.to_numeric(ddf["iri_designed"], errors="coerce").min()
        dmax = pd.to_numeric(ddf["iri_designed"], errors="coerce").max()
        vmin = float(min(vmin, dmin))
        vmax = float(max(vmax, dmax))
        print(f"🎨 Shared color scale: vmin={vmin:.3f}, vmax={vmax:.3f}")

    # ------------------------------------------------------------
    # 6) Plot PCA map (with labels)
    # ------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    gdf.plot(
        column="iri_pca",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        linewidth=0.6,
        edgecolor="black",
        legend=True,
        ax=ax,
    )

    add_ward_labels(gdf, ax, label_col="_ward_display", fontsize=args.label_fontsize)

    ax.set_title(args.title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    ensure_parent_dir(args.out_path)
    plt.savefig(args.out_path, dpi=300)
    plt.close(fig)

    print(f"🗺️ Saved labeled PCA map to {args.out_path}")
    print("✅ plot_tokyo_pca_map completed.")


if __name__ == "__main__":
    main()
