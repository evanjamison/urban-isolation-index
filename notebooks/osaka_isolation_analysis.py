# notebooks/osaka_isolation_analysis.py

# %% [markdown]
# # 🏙 Osaka Urban Isolation Index – Exploration Notebook
#
# This notebook mirrors the Tokyo workflow, but for **Osaka**.
# It assumes you have already run:
#
# - `01_ingest_jp_estat.py` (with `--city osaka`)
# - `03_build_index.py` for Osaka
# - any additional scripts that build a final index file
#
# We try to load:
#   * `data/processed/jp_osaka_with_designed_pca.csv`  (full index)
#   * if not found, we fall back to `data/processed/jp_osaka_index.csv`.

# %% 0. Imports & paths

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import geopandas as gpd

# Optional: if you have PySAL installed for Moran's I / LISA
try:
    from esda.moran import Moran
    from esda.moran import Moran_Local
    import libpysal
    HAVE_PYSAL = True
except ImportError:
    HAVE_PYSAL = False
    print("⚠ PySAL not found; Moran's I / LISA cells will be skipped.")

# Configure plotting style
plt.rcParams["figure.dpi"] = 120
sns.set(style="whitegrid")

PROJECT_ROOT = Path("..").resolve()
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"

# %% 1. Load processed Osaka data

candidate_paths = [
    DATA_PROCESSED / "jp_osaka_with_designed_pca.csv",
    DATA_PROCESSED / "jp_osaka_index_with_access.csv",
    DATA_PROCESSED / "jp_osaka_index.csv",
]

index_path = None
for p in candidate_paths:
    if p.exists():
        index_path = p
        break

if index_path is None:
    raise FileNotFoundError(
        "Could not find an Osaka index file. "
        "Expected one of:\n"
        "  - data/processed/jp_osaka_with_designed_pca.csv\n"
        "  - data/processed/jp_osaka_index_with_access.csv\n"
        "  - data/processed/jp_osaka_index.csv\n"
        "Run your Osaka build scripts first (03_build_index, 13_build_index_osaka, etc.)."
    )

print(f"📥 Loading processed Osaka index data from: {index_path}")
df = pd.read_csv(index_path)
display(df.head())

print("\nColumns:\n", df.columns.tolist())

# %% 2. Basic sanity checks

print("✅ Number of wards:", len(df))
print("\nSummary stats for key columns:")
key_cols = [c for c in df.columns if c.startswith("pct_") or c in ["poverty_rate", "iri_designed", "iri_pca"]]
display(df[key_cols].describe().T)

# Check missing values
print("\nMissing values per column:")
display(df.isna().sum())

# %% 3. Correlation between designed and PCA index (if both present)

if "iri_designed" in df.columns and "iri_pca" in df.columns:
    corr = df[["iri_designed", "iri_pca"]].corr().iloc[0, 1]
    print(f"📈 Correlation between D-IRI and PCA index: {corr:.3f}")

    plt.figure(figsize=(5, 5))
    sns.scatterplot(data=df, x="iri_designed", y="iri_pca")
    plt.axline((0, 0), slope=1, linestyle="--", linewidth=1)
    plt.title("Osaka: Designed vs PCA Isolation Index")
    plt.xlabel("Designed Isolation Index (iri_designed)")
    plt.ylabel("PCA-based Index (iri_pca)")
    plt.tight_layout()
    plt.show()
else:
    print("ℹ Either 'iri_designed' or 'iri_pca' is missing; skipping index-index scatter.")

# %% 4. Join with Osaka ward geometries

# Adjust this if your GeoJSON has a different name/path.
wards_path = DATA_EXTERNAL / "jp_osaka_wards.geojson"
print(f"📥 Loading Osaka ward geometries from: {wards_path}")
wards_gdf = gpd.read_file(wards_path)

print("\nWard GeoJSON columns:\n", wards_gdf.columns.tolist())
display(wards_gdf.head())

# We expect a `ward_jis` column to match the Osaka index.
if "ward_jis" not in df.columns:
    raise ValueError("Expected 'ward_jis' column in Osaka index data.")

if "ward_jis" not in wards_gdf.columns:
    raise ValueError(
        "GeoJSON is missing 'ward_jis' column. "
        "If the code column is named something like 'N03_007', "
        "add a rename step in this notebook."
    )

# Make sure the join key types match
df["ward_jis"] = df["ward_jis"].astype(str)
wards_gdf["ward_jis"] = wards_gdf["ward_jis"].astype(str)

gdf = wards_gdf.merge(df, on="ward_jis", how="left")

print("✅ Joined GeoDataFrame (first few rows):")
display(gdf[["ward_jis", "ward_name"]].head())

# %% 5. Choropleth map – Designed Isolation Index (D-IRI)

if "iri_designed" not in gdf.columns:
    raise ValueError("No 'iri_designed' column found; cannot plot D-IRI map.")

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
gdf.plot(
    column="iri_designed",
    cmap="viridis",
    linewidth=0.4,
    edgecolor="black",
    legend=True,
    ax=ax,
)
ax.set_title("Osaka Designed Isolation Index (D-IRI)")
ax.axis("off")
plt.tight_layout()
plt.show()

# %% 6. Choropleth map – PCA Isolation Index (if available)

if "iri_pca" in gdf.columns:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    gdf.plot(
        column="iri_pca",
        cmap="viridis",
        linewidth=0.4,
        edgecolor="black",
        legend=True,
        ax=ax,
    )
    ax.set_title("Osaka PCA-based Isolation Index")
    ax.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("ℹ 'iri_pca' not present; skipping PCA map.")

# %% 7. Global Moran's I (optional, requires PySAL)

if HAVE_PYSAL and "iri_designed" in gdf.columns:
    print("🌐 Computing Global Moran's I for D-IRI (Osaka)...")

    # Build spatial weights (queen contiguity on ward polygons)
    w = libpysal.weights.Queen.from_dataframe(gdf)
    w.transform = "r"

    y = gdf["iri_designed"].values
    moran = Moran(y, w)

    print(f"Moran's I: {moran.I:.3f}")
    print(f"p-value (permutation): {moran.p_sim:.4f}")
else:
    print("ℹ PySAL not available or 'iri_designed' missing; skipping Global Moran's I.")

# %% 8. Local Moran's I (LISA) + cluster categories (optional)

if HAVE_PYSAL and "iri_designed" in gdf.columns:
    print("🧩 Computing Local Moran's I (LISA) for Osaka...")

    w = libpysal.weights.Queen.from_dataframe(gdf)
    w.transform = "r"
    y = gdf["iri_designed"].values
    lisa = Moran_Local(y, w)

    gdf["lisa_I"] = lisa.Is
    gdf["lisa_p"] = lisa.p_sim

    # Cluster classification (0=ns, 1=HH, 2=LH, 3=LL, 4=HL)
    sig = lisa.p_sim < 0.05
    gdf["lisa_cluster_raw"] = 0
    gdf.loc[sig, "lisa_cluster_raw"] = lisa.q[sig]

    cluster_map = {
        0: "Not significant",
        1: "High-High",
        2: "Low-High",
        3: "Low-Low",
        4: "High-Low",
    }
    gdf["lisa_cluster"] = gdf["lisa_cluster_raw"].map(cluster_map)

    print("Cluster counts:")
    display(gdf["lisa_cluster"].value_counts())

    # Simple LISA cluster map
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    gdf.plot(
        column="lisa_cluster",
        categorical=True,
        legend=True,
        linewidth=0.4,
        edgecolor="black",
        ax=ax,
    )
    ax.set_title("Osaka D-IRI – LISA Cluster Map")
    ax.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("ℹ Skipping LISA: PySAL not available or 'iri_designed' missing.")

# %% 9. Save enriched GeoDataFrame (optional)

out_path = DATA_PROCESSED / "jp_osaka_index_with_lisa.geojson"
print(f"💾 Saving Osaka GeoDataFrame with LISA (if computed) to: {out_path}")
gdf.to_file(out_path, driver="GeoJSON")
print("Done.")
