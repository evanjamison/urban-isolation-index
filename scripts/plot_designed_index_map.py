# scripts/plot_designed_index_map.py
r"""
plot_designed_index_map.py

Standalone choropleth map of the Designed Isolation Index (D-IRI) for Tokyo wards.

Features:
- Uses iri_designed (z-score composite)
- Optional shared color scale with PCA (to match comparison plots)
- Japanese ward labels with Romaji: 新宿区 (Shinjuku-ku)
- Bold labels with white halo for readability
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


# --------------------------------------------------
# Label helper (bold + halo)
# --------------------------------------------------
def extract_kanji_name(label: str) -> str:
    """From '江東区 (Koto-ku)' -> '江東区'."""
    return str(label).split(" (", 1)[0].strip()



def add_ward_labels(gdf, ax, label_col, fontsize=8):
    # Default points (always inside polygon)
    pts = gdf.geometry.representative_point()

    # Manual overrides for better centering on tricky wards
    # (You can add more wards here later if needed.)
    overrides = {
        "江東区": None,     # will be computed from polygon below
        "江戸川区": None,   # will be computed from polygon below
    }
    # Small manual nudges (in map coordinate units) to reduce label overlap
    # Tune values if needed.
    nudges = {
        "墨田区": (0.004, 0.002),   # Sumida-ku: slightly right + up
        "台東区": (-0.004, -0.002), # Taito-ku: slightly left + down
    }

    # Compute nicer points for overrides using centroid in a projected CRS
    # (centroid is more "center-ish" than representative_point; projection makes it geometrically meaningful)
    gdf_proj = gdf.to_crs(3857)  # Web Mercator, fine for centroids/label placement
    centroids = gdf_proj.geometry.centroid.to_crs(gdf.crs)

    # Fill override dict with computed centroids for just those wards
    for i, disp in enumerate(gdf[label_col].astype(str)):
        kanji = extract_kanji_name(disp)
        if kanji in overrides:
            overrides[kanji] = centroids.iloc[i]

    # Draw labels
    for i, disp in enumerate(gdf[label_col].astype(str)):
        kanji = extract_kanji_name(disp)

        # Use override centroid if we have it, else representative point
        p = overrides.get(kanji) or pts.iloc[i]
        x, y = p.x, p.y
        dx, dy = nudges.get(kanji, (0.0, 0.0))
        x += dx
        y += dy
        
        txt = ax.text(
            x, y, disp,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="black",
        )

        # White halo outline
        txt.set_path_effects([
            path_effects.Stroke(linewidth=2.8, foreground="white"),
            path_effects.Normal(),
        ])


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index-path", required=True)
    p.add_argument("--pca-path", default=None)
    p.add_argument("--wards-geojson", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--label-fontsize", type=int, default=8)
    args = p.parse_args()

    print("📥 Loading designed index:", args.index_path)
    df = pd.read_csv(args.index_path)

    for c in ("ward_jis", "ward_name", "iri_designed"):
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'")

    df["ward_jis"] = df["ward_jis"].astype(str)
    df["ward_name"] = df["ward_name"].astype(str)

    df["_ward_display"] = df["ward_name"].apply(
        lambda k: f"{k} ({TOKYO_WARD_ROMAJI[k]})" if k in TOKYO_WARD_ROMAJI else k
    )

    # --------------------------------------------------
    # Color normalization
    # --------------------------------------------------
    vmin = df["iri_designed"].min()
    vmax = df["iri_designed"].max()

    if args.pca_path:
        print("📥 Loading PCA index for shared scale:", args.pca_path)
        df_pca = pd.read_csv(args.pca_path)
        if "iri_pca" not in df_pca.columns:
            raise ValueError("PCA file missing iri_pca")
        vmin = min(vmin, df_pca["iri_pca"].min())
        vmax = max(vmax, df_pca["iri_pca"].max())

    print(f"🎨 Color scale: vmin={vmin:.3f}, vmax={vmax:.3f}")

    # --------------------------------------------------
    # Load geometry
    # --------------------------------------------------
    wards = gpd.read_file(args.wards_geojson)
    if "N03_007" not in wards.columns:
        raise ValueError("GeoJSON missing N03_007")
    wards["N03_007"] = wards["N03_007"].astype(str)

    gdf = wards.merge(
        df[["ward_jis", "_ward_display", "iri_designed"]],
        left_on="N03_007",
        right_on="ward_jis",
        how="inner",
    )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    gdf.plot(
        column="iri_designed",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        linewidth=0.6,
        edgecolor="black",
        legend=True,
        ax=ax,
    )

    add_ward_labels(
        gdf,
        ax,
        label_col="_ward_display",
        fontsize=args.label_fontsize,
    )

    ax.set_title("Tokyo D-IRI (z-score composite)", fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    plt.savefig(args.out_path, dpi=300)
    plt.close()

    print("💾 Saved map to:", args.out_path)


if __name__ == "__main__":
    main()
