# File: notebooks/tokyo_isolation_analysis.py
# Run this as a Jupyter-style script (VS Code / Cursor supports # %% cells)

# %% [markdown]
# # Tokyo Urban Isolation Index – Analysis Notebook
#
# This notebook explores the Tokyo isolation index:
# - Loads the processed Tokyo dataset
# - Summarizes key features (aging, single living, poverty, access, transit)
# - Compares Designed Isolation Index (D-IRI) vs PCA-based index
# - Visualizes isolation patterns on maps
# - Inspects spatial clustering using precomputed LISA results
#
# Assumes you have already run the pipeline to produce:
# - data/processed/jp_tokyo_with_designed_pca.csv
# - data/external/jp_tokyo_wards.geojson
# - out/maps/tokyo_iso_index.png
# - out/spatial_tokyo/tokyo_diri_vs_pca.png
# - out/spatial_tokyo/tokyo_diri_lisa_clusters.png
# - out/spatial_tokyo/tokyo_diri_lisa_results.csv

# %% 
import os
os.chdir("C:/Users/user/OneDrive/Documents/urban-isolation-index")
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from IPython.display import display, Image

plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

PROJECT_ROOT = Path(".")  # adjust if running from a different working dir
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
SPATIAL_DIR = PROJECT_ROOT / "out" / "spatial_tokyo"
MAPS_DIR = PROJECT_ROOT / "out" / "maps"

# %% [markdown]
# ## 1. Load processed Tokyo data

# %%
index_path = Path("data/processed/jp_tokyo_with_designed_pca.csv")
df = pd.read_csv(index_path)
df.head()


# %% [markdown]
# ## 2. Basic info and summary statistics

# %%
print("DataFrame info:")
df.info()

# %%
# Select core numeric variables for summary
core_cols = [
    "pct_age65p",
    "pct_single65p",
    "poverty_rate",
    "access_z",
    "station_density",
    "iri_designed",
    "iri_pca",
]

print("\nSummary statistics for key variables:")
display(df[core_cols].describe().T)

# %% [markdown]
# ## 3. Correlation structure of isolation-related features

# %%
corr = df[core_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix – Tokyo Isolation Features")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Designed vs PCA Isolation Index
#
# We expect strong agreement if the designed index aligns well with the
# underlying data structure captured by PCA.

# %%
plt.figure(figsize=(7, 6))
sns.scatterplot(data=df, x="iri_designed", y="iri_pca")
sns.regplot(data=df, x="iri_designed", y="iri_pca", scatter=False, color="red")
plt.xlabel("Designed Isolation Index (D-IRI)")
plt.ylabel("PCA-based Isolation Index")
plt.title("D-IRI vs PCA Isolation Index – Tokyo Wards")
plt.axhline(0, color="grey", linewidth=0.5)
plt.axvline(0, color="grey", linewidth=0.5)
plt.tight_layout()
plt.show()

print("Correlation between D-IRI and PCA index:")
display(df[["iri_designed", "iri_pca"]].corr())

# %% [markdown]
# ## 5. Join with ward geometries

# %%
# 5. Join with ward geometries

print(df.columns.tolist())

wards_path = DATA_EXTERNAL / "jp_tokyo_wards.geojson"
print(f"Loading ward geometries from: {wards_path}")
gdf_wards = gpd.read_file(wards_path)

# Check what columns are in the GeoJSON
print(gdf_wards.columns.tolist())

# 🔧 Create a matching ward_jis column from N03_007
# Make both sides the same type (I like strings for safety)
df["ward_jis"] = df["ward_jis"].astype(str)
gdf_wards["ward_jis"] = gdf_wards["N03_007"].astype(str)

# Now the merge will work
gdf = gdf_wards.merge(df, on="ward_jis", how="left")

print("Joined GeoDataFrame (first few rows):")
display(gdf[["ward_jis", "ward_name", "iri_designed", "iri_pca"]].head())


# %% [markdown]
# ## 6. Choropleth map – Designed Isolation Index (D-IRI)

# %%
vmin = gdf["iri_designed"].min()
vmax = gdf["iri_designed"].max()

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
gdf.plot(
    column="iri_designed",
    cmap="viridis",
    linewidth=0.4,
    edgecolor="black",
    vmin=vmin,
    vmax=vmax,
    ax=ax,
)
ax.set_title("Tokyo Designed Isolation Index (D-IRI)")
ax.axis("off")

sm = plt.cm.ScalarMappable(cmap="viridis")
sm.set_clim(vmin, vmax)
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Standardized isolation score")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Choropleth map – PCA-based Isolation Index

# %%
vmin = gdf["iri_pca"].min()
vmax = gdf["iri_pca"].max()

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
gdf.plot(
    column="iri_pca",
    cmap="viridis",
    linewidth=0.4,
    edgecolor="black",
    vmin=vmin,
    vmax=vmax,
    ax=ax,
)
ax.set_title("Tokyo PCA-based Isolation Index")
ax.axis("off")

sm = plt.cm.ScalarMappable(cmap="viridis")
sm.set_clim(vmin, vmax)
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Standardized isolation score")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Display saved map outputs (for quick sanity check)

# %%
if (MAPS_DIR / "tokyo_iso_index.png").exists():
    print("Saved D-IRI map from pipeline:")
    display(Image(filename=str(MAPS_DIR / "tokyo_iso_index.png"), width=400))
else:
    print("tokyo_iso_index.png not found in out/maps/")

if (SPATIAL_DIR / "tokyo_diri_vs_pca.png").exists():
    print("Saved D-IRI vs PCA comparison map:")
    display(Image(filename=str(SPATIAL_DIR / "tokyo_diri_vs_pca.png"), width=400))
else:
    print("tokyo_diri_vs_pca.png not found in out/spatial_tokyo/")

if (SPATIAL_DIR / "tokyo_diri_lisa_clusters.png").exists():
    print("Saved LISA clusters map:")
    display(Image(filename=str(SPATIAL_DIR / "tokyo_diri_lisa_clusters.png"), width=400))
else:
    print("tokyo_diri_lisa_clusters.png not found in out/spatial_tokyo/")

# %% [markdown]
# ## 9. LISA results – cluster counts
#
# These results should come from a script like `11_spatial_stats_tokyo.py`
# which writes `tokyo_diri_lisa_results.csv`.

# %%
lisa_path = SPATIAL_DIR / "tokyo_diri_lisa_results.csv"

if lisa_path.exists():
    lisa_df = pd.read_csv(lisa_path)
    print(f"Loaded LISA results from: {lisa_path}")
    print("\nCluster frequency:")
    display(lisa_df["lisa_cluster"].value_counts())
else:
    print(f"No LISA results found at: {lisa_path}")
    lisa_df = None

# %% [markdown]
# ## 10. Merge LISA cluster labels onto geometry (if available)

# %%
# %% 10. Merge LISA cluster labels onto geometry (if available)

if lisa_df is not None:
    print("LISA columns:", lisa_df.columns.tolist())

    if "ward_jis" not in lisa_df.columns:
        raise ValueError("Expected 'ward_jis' column in LISA results.")

    # Merge LISA cluster labels onto the existing GeoDataFrame `gdf`
    gdf_lisa = gdf.merge(
        lisa_df[["ward_jis", "lisa_cluster"]],
        on="ward_jis",
        how="left",
    )

    print("Sample of wards with D-IRI and LISA cluster:")
    display(gdf_lisa[["ward_jis", "ward_name", "iri_designed", "lisa_cluster"]].head())

else:
    print("Skipping LISA merge; no LISA results loaded.")


# %% [markdown]
# ## 11. Isolation vs key predictors (scatter plots)
#
# These help interpret which features drive higher D-IRI values.

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.scatterplot(data=df, x="pct_age65p", y="iri_designed", ax=axes[0])
axes[0].set_title("D-IRI vs % Age 65+")
axes[0].set_xlabel("% age 65+")
axes[0].set_ylabel("D-IRI")

sns.scatterplot(data=df, x="pct_single65p", y="iri_designed", ax=axes[1])
axes[1].set_title("D-IRI vs % 65+ living alone")
axes[1].set_xlabel("% 65+ living alone")
axes[1].set_ylabel("D-IRI")

sns.scatterplot(data=df, x="poverty_rate", y="iri_designed", ax=axes[2])
axes[2].set_title("D-IRI vs Poverty rate")
axes[2].set_xlabel("Poverty rate")
axes[2].set_ylabel("D-IRI")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 12. Optional: OLS regression for interpretability
#
# Note: D-IRI is already built from these components, so this model is
# mainly descriptive, showing how each feature relates to variation
# in the index across wards.

# %%
try:
    import statsmodels.formula.api as smf

    formula = "iri_designed ~ pct_age65p + pct_single65p + poverty_rate + access_z + station_density"
    model = smf.ols(formula, data=df).fit()
    print(model.summary())
except ImportError:
    print("statsmodels not installed; skipping OLS regression.")
