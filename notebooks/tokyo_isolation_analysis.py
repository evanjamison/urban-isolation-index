# File: notebooks/tokyo_isolation_analysis.py
# Run this as a Jupyter-style script (VS Code / Cursor supports # %% cells)

# %% [markdown]
# # Tokyo Urban Isolation Index — Analytical Summary
#
# This notebook presents a structured analysis of social isolation risk across
# Tokyo’s 23 special wards using an integrated, data-driven index derived from
# the 2020 Population Census. The objective is to demonstrate how demographic,
# socioeconomic, and accessibility indicators can be combined to quantify
# spatial patterns of vulnerability in large urban environments.
#
# **Analytical Focus**
# The Tokyo Urban Isolation Index synthesizes several ward-level dimensions:
# - Aging composition (share of residents aged 65+)  
# - Household structure, with emphasis on single-living older adults  
# - Poverty prevalence  
# - (Optional) Public transit accessibility indicators  
#
# After cleaning and normalizing the inputs, the notebook evaluates two
# complementary formulations:
# 1. **Designed Isolation Index (D-IRI):** a weighting-guided composite measure.  
# 2. **PCA-based Index:** an unsupervised, dimension-reduction alternative.
#
# **Technical Contributions**
# - Implements full preprocessing, feature scaling, and index construction  
# - Produces diagnostic summaries for each risk dimension  
# - Generates spatial visualizations to highlight geographic concentration  
# - Computes spatial autocorrelation (Moran’s I and LISA) when PySAL is available  
#
# **Relevance**
# The workflow aligns with research in urban analytics, public policy,
# demography, and spatial data science. By operationalizing isolation risk into
# a reproducible index framework, this notebook contributes a transparent,
# extensible methodology for comparing neighborhood-level social vulnerability
# both within Tokyo and across cities.
#
# **Prerequisites**
# Expected pipeline outputs before running this notebook:
# - `data/processed/jp_tokyo_with_designed_pca.csv`  
# - `data/external/jp_tokyo_wards.geojson`  
# - (Optional) LISA results and map exports  
#
# The analyses contained here form the core reference for Tokyo in the broader
# Urban Isolation Index project and serve as a model for extending the study to
# Osaka and other metropolitan regions.



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
### 3. Interpretation of Correlation Structure
#The correlation matrix reveals a clear and interpretable structure among isolation-related features.
#Demographic pressures—aging, single-living, and poverty—are tightly interlinked, consistent with 
#Tokyo’s well-documented patterns of social vulnerability. Accessibility and station density behave as 
#distinct spatial components, capturing infrastructural differences rather than demographic conditions. 
#The strong alignment between the Designed Isolation Index (D-IRI) and the PCA-based index suggests 
#that the conceptual model successfully captures the dominant underlying structure in the data.


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
### 4. Interpretation of D-IRI vs PCA Index Relationship
#The near-linear relationship between the Designed Isolation Index and the PCA-derived isolation 
#factor demonstrates strong convergent validity. Despite being created from different modeling 
#philosophies—one theory-driven, one data-driven—both indices yield consistent ward-level patterns. 
#This agreement provides strong evidence that the core dimensions of isolation risk are being 
#systematically captured, independent of modeling choice.


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
# --- FIX ward_jis type mismatch before merging ---
# Convert both sources to string
df["ward_jis"] = df["ward_jis"].astype(str)

if "ward_jis" in lisa_df.columns:
    lisa_df["ward_jis"] = lisa_df["ward_jis"].astype(str)
elif "N03_007" in lisa_df.columns:
    # rename to expected column
    lisa_df = lisa_df.rename(columns={"N03_007": "ward_jis"})
    lisa_df["ward_jis"] = lisa_df["ward_jis"].astype(str)
else:
    raise ValueError("No ward code column found in LISA results.")

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
### 11. Interpretation of Predictor Scatter Plots
#The scatter plots show clear monotonic relationships between D-IRI and its core demographic features. 
#Higher proportions of older adults, single-living residents, and households in poverty are associated 
#with higher isolation scores. These relationships provide face validity for the index, demonstrating 
#that its behavior aligns with established theoretical expectations regarding social isolation in 
#urban contexts.


# %% [markdown]
### 12. Reliability Check — Cronbach’s Alpha
# We test whether isolation-related components form a consistent index.

# %%
# 12. Reliability Check — Cronbach's Alpha

import numpy as np
import pandas as pd

def cronbach_alpha(df_subset: pd.DataFrame) -> float:
    """
    Compute Cronbach's alpha for a DataFrame of features.
    Each column should be one component of the index.
    """
    # Ensure we operate on a DataFrame and drop rows with any missing values
    df_subset = df_subset.dropna()

    # Variance for each item
    item_vars = df_subset.var(axis=0, ddof=1)

    # Total variance of summed scale
    total_var = df_subset.sum(axis=1).var(ddof=1)

    n_items = df_subset.shape[1]
    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
    return alpha

# ---- Select variables for reliability test ----
# Use the z-scored versions where available; station_density is raw.
print("Available columns:", df.columns.tolist())

features_for_alpha = df[[
    "pct_age65p_z",
    "pct_single65p_z",
    "poverty_rate_z",
    "access_z",
    "station_density",   # <- use this (no '_z')
]]

alpha = cronbach_alpha(features_for_alpha)
print("Cronbach’s Alpha (internal consistency):", round(alpha, 3))
# %%
### Interpretation of Cronbach’s Alpha

# Cronbach’s α = **0.584**, which indicates moderate internal consistency among the components of the Tokyo Isolation Index.

# This level is **expected and appropriate** for a multidimensional social index:
# - The included variables (aging, single living, poverty, access, station density) describe **different dimensions** of isolation risk.
# - High alpha (≥0.8) would imply the variables are redundant.
# - Moderate alpha suggests **each feature contributes unique information**.

# This supports using these components together to construct the Designed Isolation Index (D-IRI).

# %% [markdown]
### 13. Robustness Check — Weight Sensitivity Analysis
# We perturb each feature's weight by ±20% and examine how much ward rankings change.

# %%
import pandas as pd

base_weights = {
    "pct_age65p_z": 1.0,
    "pct_single65p_z": 1.0,
    "poverty_rate_z": 1.0,
    "access_z": 1.0,
    "station_density": 1.0
}

def compute_index(df, weights):
    return sum(df[k] * w for k, w in weights.items())

# Compute baseline ranking
df["iso_index_baseline"] = compute_index(df, base_weights)
baseline_rank = df["iso_index_baseline"].rank()

results = []

for feature in base_weights.keys():
    for pct_change in [-0.2, 0.2]:   # ±20%
        new_weights = base_weights.copy()
        new_weights[feature] *= (1 + pct_change)

        df["iso_index_perturbed"] = compute_index(df, new_weights)
        perturbed_rank = df["iso_index_perturbed"].rank()

        # Rank correlation to measure sensitivity
        corr = baseline_rank.corr(perturbed_rank)

        results.append({
            "feature": feature,
            "perturbation": pct_change,
            "rank_correlation": corr
        })

sensitivity_df = pd.DataFrame(results)
display(sensitivity_df)

# %% [markdown]
### 13. Interpretation of Weight Sensitivity Analysis
#The robustness check shows that perturbing any feature’s weight by ±20% produces minimal changes in 
#ward rankings (rank-correlation consistently ≥ 0.98). This indicates that the index is highly stable 
#and not overly sensitive to subjective weighting decisions, strengthening confidence in its reliability 
#for comparative spatial analysis.


# %% [markdown]
### 14. Spatial Autocorrelation — Global Moran's I
# Tests whether isolation index values are spatially clustered.

# %%
# Tests whether isolation index values are spatially clustered.

from esda.moran import Moran
import libpysal
import geopandas as gpd

# 1. Load ward geometries (Tokyo)
wards_path = DATA_EXTERNAL / "jp_tokyo_wards.geojson"
print(f"Loading Tokyo ward geometries from: {wards_path}")
gdf_wards = gpd.read_file(wards_path)

print("Ward GeoJSON columns:", gdf_wards.columns.tolist())

# 2. Join geometries with the processed isolation dataframe
#    - gdf_wards uses N03_007 as the JIS code
#    - df uses ward_jis as the JIS code
gdf = gdf_wards.merge(
    df[["ward_jis", "ward_name", "iri_designed"]],
    left_on="N03_007",
    right_on="ward_jis",
    how="inner",
)

print("Joined GeoDataFrame shape:", gdf.shape)
display(gdf[["ward_jis", "ward_name", "iri_designed"]].head())

# 3. Build spatial weights (Queen contiguity)
w = libpysal.weights.Queen.from_dataframe(gdf, silence_warnings=True, ids=gdf.index.tolist())
w.transform = "r"

y = gdf["iri_designed"].values

mi = Moran(y, w)
print("Global Moran’s I:", round(mi.I, 4))
print("p-value:", round(mi.p_sim, 4))
# %% [markdown]
### 14. Interpretation of Global Moran’s I
#The Global Moran’s I statistic (I ≈ 0.307, p ≈ 0.006) demonstrates statistically significant spatial 
#clustering of isolation risk across Tokyo’s 23 wards. Rather than being randomly distributed, 
#high- and low-isolation areas tend to cluster geographically. This confirms that social isolation risk 
#is shaped by spatial context and that the index captures meaningful geographic patterns rather than 
#random variation.

# %% [markdown]
### 15. PCA Factor Loadings (Interpretability Check)
# Shows how each variable contributes to the PCA isolation factor.

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = df[["pct_age65p", "pct_single65p", "poverty_rate", "access_z", "station_density"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=1)
pca.fit(X_scaled)

loadings = pd.Series(pca.components_[0], index=X.columns)
display(loadings.to_frame("PCA Loading"))


# %% [markdown]
## Limitations and Future Work

#While the Tokyo Isolation Index provides a robust foundation for understanding spatial patterns of 
#isolation risk, several limitations suggest opportunities for advancement:

#**1. Limited dimensionality of available indicators.**  
#The analysis relies heavily on demographic and socioeconomic variables. Future iterations could benefit 
#from incorporating additional dimensions such as disability status, caregiver availability, neighborhood 
#land-use diversity, or subjective well-being indicators if accessible through Japanese survey data.

#**2. Station density as a coarse mobility proxy.**  
#Rail station density captures transit accessibility only indirectly. Incorporating train frequency, 
#bus routes, travel-time surfaces, or multimodal network accessibility could significantly refine the 
#mobility/access dimension.

#**3. Static snapshot of a dynamic urban process.**  
#The current pipeline uses 2020 census data. Recomputing the index using multiple census years—or 
#synthetic population projections—would enable longitudinal analysis of isolation trends.

#**4. Tokyo-only validation.**  
#Although the index performs well for Tokyo, generalizability has not yet been demonstrated.  
#The next major step is **replicating the full pipeline for Osaka**, which will allow:

#- external geographic validation,  
#- comparison of isolation profiles across major metropolitan regions,  
#- identification of structural similarities or differences between eastern and western Japan,  
#- testing whether the index’s statistical properties hold across distinct urban contexts.

#Building a parallel Osaka Isolation Index (O-IRI) will also enable future cross-city clustering, 
#regional PCA, or nationwide generalization of isolation risk modeling—aligning the project more closely 
#with the goals of quantitative social science and spatial data analysis programs in Japan.


