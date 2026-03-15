# src/app/main.py
from __future__ import annotations
import numpy as np

import json
import textwrap
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------
# Page config MUST come before other Streamlit calls
# --------------------------------------------------
st.set_page_config(page_title="Urban Isolation Index — Tokyo", layout="wide")

# Plotly is optional (we’ll degrade gracefully if missing)
try:
    import plotly.express as px  # type: ignore
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


# -----------------------------
# Helpers / paths
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parents[2]  # assumes src/app/main.py -> ROOT/src/app/main.py


def find_first(patterns: list[str], search_roots: list[Path]) -> Path | None:
    """Return the first existing file matching any glob pattern under any search root."""
    for base in search_roots:
        if not base.exists():
            continue
        for pat in patterns:
            matches = sorted(base.rglob(pat))
            if matches:
                return matches[0]
    return None


def safe_read_csv(path: Path) -> pd.DataFrame:
    """Read CSV with a few common encodings for JP datasets."""
    for enc in ("utf-8", "utf-8-sig", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin-1")


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert messy numeric strings to floats safely."""
    s = series
    if s.dtype == "object":
        s = (
            s.astype(str)
            .str.replace(",", "", regex=False)  # remove thousands separators
            .str.replace(r"[^\d\.\-\+eE]", "", regex=True)  # strip stray chars
        )
    return pd.to_numeric(s, errors="coerce")


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


def infer_ward_col(df: pd.DataFrame) -> str:
    ward_candidates = [c for c in df.columns if c.lower() in ("ward_name", "ward", "name", "wardname")]
    return ward_candidates[0] if ward_candidates else df.columns[0]


def pick_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def add_rankings_block(df: pd.DataFrame, *, ward_col: str, score_col: str, n_show: int) -> None:
    """Render top/bottom rankings for the chosen ward + score columns."""
    d = df.copy()

    # build display name: 新宿区 (Shinjuku-ku)
    ward_series = d[ward_col].astype(str).str.strip()
    d["_ward_display"] = ward_series.apply(
        lambda k: f"{k} ({TOKYO_WARD_ROMAJI[k]})" if k in TOKYO_WARD_ROMAJI else k
    )

    d[score_col] = coerce_numeric(d[score_col])

    st.markdown(f"**Ward column:** `{ward_col}`  \n**Index column:** `{score_col}`")

    base = d.dropna(subset=[score_col]).copy()
    if base.empty:
        st.error(f"No usable numeric values found in `{score_col}`.")
        st.write(d[[ward_col, score_col]].head(10))
        return

    top = (
        base.sort_values(score_col, ascending=False)
        .head(n_show)[["_ward_display", score_col]]
        .reset_index(drop=True)
    )
    bot = (
        base.sort_values(score_col, ascending=True)
        .head(n_show)[["_ward_display", score_col]]
        .reset_index(drop=True)
    )

    left, right = st.columns(2)
    with left:
        st.markdown(f"### Highest risk (top {n_show})")
        st.dataframe(top, use_container_width=True)
    with right:
        st.markdown(f"### Lowest risk (bottom {n_show})")
        st.dataframe(bot, use_container_width=True)


def add_rankings_block_custom(
    df: pd.DataFrame,
    *,
    ward_col: str,
    score_col: str,
    n_show: int,
    high_label: str = "Highest values",
    low_label: str = "Lowest values",
) -> None:
    """Render top/bottom rankings with customizable labels."""
    d = df.copy()

    ward_series = d[ward_col].astype(str).str.strip()
    d["_ward_display"] = ward_series.apply(
        lambda k: f"{k} ({TOKYO_WARD_ROMAJI[k]})" if k in TOKYO_WARD_ROMAJI else k
    )

    d[score_col] = coerce_numeric(d[score_col])

    st.markdown(f"**Ward column:** `{ward_col}`  \n**Index column:** `{score_col}`")

    base = d.dropna(subset=[score_col]).copy()
    if base.empty:
        st.error(f"No usable numeric values found in `{score_col}`.")
        st.write(d[[ward_col, score_col]].head(10))
        return

    top = (
        base.sort_values(score_col, ascending=False)
        .head(n_show)[["_ward_display", score_col]]
        .reset_index(drop=True)
    )
    bot = (
        base.sort_values(score_col, ascending=True)
        .head(n_show)[["_ward_display", score_col]]
        .reset_index(drop=True)
    )

    left, right = st.columns(2)
    with left:
        st.markdown(f"### {high_label}")
        st.dataframe(top, use_container_width=True)
    with right:
        st.markdown(f"### {low_label}")
        st.dataframe(bot, use_container_width=True)


def render_hover_map(
    *,
    wards_geojson_path: Path,
    df: pd.DataFrame,
    value_col: str,
    title: str,
    color_scale: str = "Viridis",
    diverging_midpoint: float | None = None,
    symmetric_range: bool = False,
) -> None:
    """
    Interactive hover choropleth using Plotly + GeoJSON.
    Uses join_key='N03_007' (Tokyo ward JIS code in your GeoJSON).
    """
    if not _HAS_PLOTLY:
        st.info("Interactive hover maps require Plotly. Install it with: `pip install plotly`.")
        return

    import geopandas as gpd

    # Shapely compatibility:
    try:
        from shapely.validation import make_valid as _make_valid  # shapely 1.8+
    except Exception:
        _make_valid = None

    # ✅ KEEP your working join key
    join_key = "N03_007"

    if "ward_jis" not in df.columns:
        st.error("Expected `ward_jis` column in dataset (needed for hover map join).")
        return
    if value_col not in df.columns:
        st.error(f"Expected `{value_col}` column in dataset (needed for hover map join).")
        return
    if not wards_geojson_path.exists():
        st.error(f"GeoJSON not found: {wards_geojson_path}")
        return

    # -------------------
    # Prep data (table)
    # -------------------
    ward_col = infer_ward_col(df)
    d = df.copy()
    d["ward_jis"] = d["ward_jis"].astype(str).str.strip()
    d[value_col] = coerce_numeric(d[value_col])

    d["_ward_display"] = d[ward_col].astype(str).str.strip().apply(
        lambda k: f"{k} ({TOKYO_WARD_ROMAJI[k]})" if k in TOKYO_WARD_ROMAJI else k
    )

    d = d.dropna(subset=[value_col]).drop_duplicates(subset=["ward_jis"]).copy()

    # -------------------
    # Load + clean shapes
    # -------------------
    gdf = gpd.read_file(wards_geojson_path)

    if join_key not in gdf.columns:
        st.error(f"GeoJSON missing '{join_key}' property column.")
        st.write("GeoJSON columns:", list(gdf.columns))
        return

    gdf[join_key] = gdf[join_key].astype(str).str.strip()

    # Filter to only wards present in dataset
    codes = set(d["ward_jis"].tolist())
    gdf = gdf[gdf[join_key].isin(codes)].copy()

    if gdf.empty:
        st.error("After filtering GeoJSON to ward_jis codes, no features remain.")
        st.write("Example ward_jis:", sorted(list(codes))[:10])
        return

    # Ensure CRS is lon/lat (EPSG:4326) if CRS is known
    try:
        if gdf.crs is not None:
            gdf = gdf.to_crs(4326)
    except Exception:
        pass

    # Keep only (Multi)Polygon geometries
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    if gdf.empty:
        st.error("No Polygon/MultiPolygon geometries left after filtering.")
        return

    # Fix invalid geometries
    if _make_valid is not None:
        try:
            gdf["geometry"] = gdf["geometry"].apply(_make_valid)
        except Exception:
            gdf["geometry"] = gdf["geometry"].buffer(0)
    else:
        gdf["geometry"] = gdf["geometry"].buffer(0)

    try:
        bad = (~gdf.is_valid).sum()
        if bad:
            gdf["geometry"] = gdf["geometry"].buffer(0)
    except Exception:
        pass

    gj = json.loads(gdf.to_json())

    # -------------------
    # Optional symmetric range for difference maps
    # -------------------
    range_color = None
    if symmetric_range:
        vals = d[value_col].to_numpy()
        vmax = float(np.nanmax(np.abs(vals))) if np.isfinite(np.nanmax(np.abs(vals))) else 1.0
        if vmax == 0:
            vmax = 1.0
        range_color = (-vmax, vmax)

    # -------------------
    # Plotly choropleth
    # -------------------
    fig = px.choropleth(
        d,
        geojson=gj,
        featureidkey=f"properties.{join_key}",
        locations="ward_jis",
        color=value_col,
        hover_name="_ward_display",
        hover_data={value_col: ":.3f"},
        color_continuous_scale=color_scale,
        range_color=range_color,
        color_continuous_midpoint=diverging_midpoint,
        title=title,
    )

    fig.update_geos(
        visible=False,
        projection_type="mercator",
        center={"lat": 35.68, "lon": 139.76},
        lataxis_range=[35.4, 36.0],
        lonaxis_range=[139.4, 140.1],
    )

    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))
    st.plotly_chart(fig, use_container_width=True)


def render_ml_notes() -> None:
    """Reusable ML notes block (used in top-level narrative + ML tab)."""
    with st.expander("🧠 Machine Learning: model notation & interpretation", expanded=False):
        st.markdown("### What the ML model is doing (high level)")
        st.markdown(
            """
The machine learning component is an **Elastic Net logistic regression** trained to predict a **binary high-risk label**:

where $y_i = 1$ means ward $i$ is classified as **high isolation risk** under the **Designed Isolation Index (D-IRI)**.

> **Key point:** ML is **not redefining isolation**.  
> It’s testing which indicators best reproduce the structural high-risk classification **out-of-sample**.
"""
        )
        st.latex(r"y_i \in \{0,1\}")

        st.markdown("---")
        st.markdown("### Model form")
        st.latex(r"\Pr(y_i = 1) = \sigma\left(\beta_0 + \sum_{k=1}^{p} \beta_k x_{ik}\right)")

        st.markdown(
            """
- $x_{ik}$: predictors for ward $i$ (typically standardized)
- $\beta_k$: learned coefficients
- $\sigma(\cdot)$: logistic sigmoid
"""
        )

        st.markdown("Elastic Net regularization combines:")
        st.markdown("- **L1 (lasso)** → feature selection (many coefficients become 0)")
        st.markdown("- **L2 (ridge)** → stability (shrinks coefficients smoothly)")

        st.markdown("Objective (cross-entropy + Elastic Net penalty):")
        st.latex(
            r"""
\mathcal{L}(\beta) =
-\sum_{i=1}^{n}\Big[y_i\log(\hat p_i)+(1-y_i)\log(1-\hat p_i)\Big]
+ \lambda\Big(\alpha\lVert\beta\rVert_1 + (1-\alpha)\lVert\beta\rVert_2^2\Big)
"""
        )
        st.markdown(
            """
- $\lambda$: overall shrinkage strength  
- $\alpha$: mix between L1 and L2  
"""
        )

        st.markdown("---")
        st.markdown("### Validation: LOOCV (Leave-One-Out CV)")
        st.markdown(
            """
Because $n=23$ wards is small, you used **LOOCV**:

For each ward $i$:
1. Train on the other $n-1$ wards  
2. Predict on the held-out ward $i$  
3. Repeat for all wards  

This gives **out-of-sample predictions for every ward** (no leakage).
"""
        )

        st.markdown("---")
        st.markdown("### Why z-scoring is appropriate here")
        st.markdown("Predictors are standardized as:")
        st.latex(r"z = \frac{x - \mu}{\sigma}")

        st.markdown(
            """
This is appropriate because:

- Regularization depends on coefficient scale → standardization makes penalties fair across variables  
- Coefficients become comparable: a 1-unit change means **1 SD within Tokyo**  
- Prevents “big-unit” variables (e.g., population) from dominating purely due to scale  

It **does not erase meaning** — it changes interpretation to *relative within Tokyo*.
"""
        )

        st.markdown("---")
        st.markdown("## How to read the ML maps")

        st.markdown("### Structural z map")
        st.markdown("A standardized version of the designed structural index:")
        st.latex(r"z(\mathrm{D\text{-}IRI}_i)")

        st.markdown("### ML z map")
        st.markdown("A standardized version of the ML risk signal (probability/score then standardized):")
        st.latex(r"z(\mathrm{ML}_i)")

        st.markdown("### Difference map (core diagnostic)")
        st.latex(r"\mathrm{Diff}_i = z(\mathrm{ML}_i) - z(\mathrm{D\text{-}IRI}_i)")

        st.markdown(
            """
- **Diff > 0:** ML flags higher risk than the structural index  
- **Diff < 0:** Structural index flags higher risk than ML  
- **Diff ≈ 0:** Agreement  
"""
        )

        st.markdown("---")
        st.markdown("### Why density / overall single-households may look weak")
        st.markdown(
            """
Population density and overall single-person households can be:

- low-variance across Tokyo wards (hard to discriminate)
- indirect correlates rather than direct mechanisms
- consistent with the idea that **physical proximity ≠ social integration**

Often the model prefers **age-conditioned household structure** (e.g., *elderly living alone*) over “living alone overall”.
"""
        )

        st.markdown("---")
        st.markdown("### Key takeaway")
        st.markdown("🧮 **D-IRI encodes theory**  \n🤖 **ML tests empirical consistency out-of-sample**  \nTheir contrast is the **result**, not a contradiction.")


def render_sii_fcfi_notes() -> None:
    """Explanatory notes for SII, FCFI, and the gap map."""
    with st.expander("📘 Social Isolation Index (SII)", expanded=False):
        st.markdown(
            r"""
### Definition
The **Social Isolation Index (SII)** is a ward-level composite measure designed to capture
**structural conditions that increase the likelihood of social disconnection among residents**.

### Formula
"""
        )

        st.markdown("### Formula")
        st.latex(
            r"""
\mathrm{SII} =
( \mathrm{Solo\ Household\ \%} \times 0.35 )
+
( \mathrm{Private\ Rental\ \%} \times 0.30 )
+
( \mathrm{Single\text{-}Parent\ Household\ \%} \times 0.15 )
+
( \mathrm{Commuter\ Outflow\ \%} \times 0.10 )
+
( \mathrm{Self\text{-}Inflicted\ Injury\ Calls/10k,\ normalized} \times 0.10 )
"""
        )

        st.markdown(
            """

### Components
- **Solo Household % (0.35)** — share of all private households consisting of a single person  
- **Private Rental % (0.30)** — share of all dwellings that are privately rented  
- **Single-Parent Household % (0.15)** — share of households headed by a lone father or lone mother with children  
- **Commuter Outflow % of Residents (0.10)** — residents who commute out of the ward daily as a percentage of nighttime population  
- **Self-Inflicted Injury Ambulance Calls per 10,000 Residents, normalized (0.10)** — outcome signal of acute distress

### Interpretation
- **Higher SII** = greater structural isolation risk relative to other wards
- Scores are **relative within Tokyo**, not absolute risk thresholds
"""
        )

    with st.expander("👨‍👩‍👧 Family & Child Friendliness Index (FCFI)", expanded=False):
        st.markdown(
            r"""
### Definition
The **Family and Child Friendliness Index (FCFI)** is a ward-level composite measure
designed to capture how well a ward’s demographic composition, green space,
civic infrastructure, and public safety environment support families with children.

"""
        )

        st.markdown("### Formula")
        st.latex(
            r"""
\mathrm{FCFI} =
( \mathrm{Married\text{-}with\text{-}Children\ Household\ \%} \times 0.25 )
+
( \mathrm{Average\ Household\ Size,\ normalized} \times 0.15 )
+
( \mathrm{Park\ Area\ per\ Resident,\ normalized} \times 0.15 )
+
( \mathrm{Community\ Education\ Programs/10k,\ normalized} \times 0.10 )
+
( \mathrm{Library\ Loans\ per\ Resident,\ normalized} \times 0.10 )
-
( \mathrm{Assault\ Ambulance\ Calls/10k,\ normalized} \times 0.05 )
"""
        )

        st.markdown(
            """

### Components
- **Married-with-Children Household % (0.25)** — direct demographic signal of family composition  
- **Average Household Size, normalized (0.15)** — proxy for multi-person/family households  
- **Park Area per Resident, normalized (0.15)** — child-accessible outdoor infrastructure  
- **Community Education Programs per 10,000 Residents, normalized (0.10)** — structured civic activity accessible to families  
- **Library Loans per Resident, normalized (0.10)** — proxy for civic and cultural infrastructure  
- **Assault Ambulance Calls per 10,000 Residents, normalized (−0.05)** — negative public safety component

### Methodological note
**Chiyoda** scores anomalously high on FCFI largely because of its unusually high
park area per resident relative to its small residential population.
Interpret that score with caution.

### Interpretation
- **Higher FCFI** = more family- and child-friendly relative to other wards
- Scores are **relative within Tokyo**, not subjective liveability ratings
"""
        )

    with st.expander("🧭 SII − FCFI Gap Map", expanded=False):
        st.markdown(
            r"""
### Definition
"""
        )

        st.markdown("### Formula")
        st.latex(
            r"""
\mathrm{Gap}_i = z(\mathrm{SII}_i) - z(\mathrm{FCFI}_i)
"""
        )

        st.markdown(
            """

### Interpretation
- **Positive / red** = isolation is higher relative to family support
- **Negative / blue** = family-friendliness is stronger relative to isolation
- **Near zero / white** = relatively balanced

This is useful for highlighting wards where social isolation pressures and family-supportive infrastructure diverge.
"""
        )


# -----------------------------
# Locate outputs
# -----------------------------
search_roots = [ROOT / "out", ROOT / "data", ROOT]

# Labeled PNGs (fallback)
designed_img = find_first(
    patterns=[
        "tokyo_designed_index_labeled.png",
        "*designed*index*labeled*.png",
        "tokyo_designed_index*.png",
        "*designed*index*.png",
    ],
    search_roots=search_roots,
)

pca_img = find_first(
    patterns=[
        "tokyo_pca_index_labeled.png",
        "*pca*index*labeled*.png",
        "tokyo_pca_index*.png",
        "*pca*index*.png",
    ],
    search_roots=search_roots,
)

# Spatial PNGs (fallback)
lisa_img = find_first(
    patterns=[
        "tokyo_diri_lisa_clusters.png",
        "*lisa*cluster*.png",
        "*moran*local*.png",
    ],
    search_roots=search_roots,
)

diff_img = find_first(
    patterns=[
        "tokyo_diri_minus_pca.png",
        "*minus*pca*.png",
        "*difference*map*.png",
        "*diri*minus*.png",
    ],
    search_roots=search_roots,
)

# -----------------------------
# ML spatial PNGs (ML LISA + ML−Structural Diff LISA)
# -----------------------------
ml_lisa_img = find_first(
    patterns=[
        "tokyo_ml_lisa_clusters.png",
        "*ml*lisa*cluster*.png",
        "*tokyo*ml*lisa*.png",
    ],
    search_roots=search_roots,
)

ml_diff_lisa_img = find_first(
    patterns=[
        "tokyo_diff_lisa_clusters.png",
        "*diff*lisa*cluster*.png",
        "*ml*structural*difference*lisa*.png",
        "*ml*minus*structural*lisa*.png",
    ],
    search_roots=search_roots,
)

# CSVs
designed_csv = find_first(
    patterns=[
        "jp_tokyo_with_designed.csv",
        "*tokyo*with*designed*.csv",
        "*tokyo*designed*.csv",
    ],
    search_roots=search_roots,
)

pca_csv = find_first(
    patterns=[
        "jp_tokyo_with_designed_pca.csv",
        "*tokyo*designed*pca*.csv",
        "*tokyo*pca*.csv",
    ],
    search_roots=search_roots,
)

# ML difference map CSV (produced by ml/scripts/06_make_difference_map.py)
ml_diff_csv = find_first(
    patterns=[
        "difference_map_data.csv",
        "*difference*map*data*.csv",
        "*ml*maps*difference*.csv",
    ],
    search_roots=search_roots,
)

# LISA results CSV (optional)
lisa_csv = find_first(
    patterns=[
        "tokyo_diri_lisa_results.csv",
        "*lisa*results*.csv",
        "*moran*local*.csv",
    ],
    search_roots=search_roots,
)

# Wards GeoJSON (needed for hover)
wards_geojson = find_first(
    patterns=[
        "jp_tokyo_wards.geojson",
        "*tokyo*wards*.geojson",
        "*tokyo*.geojson",
    ],
    search_roots=search_roots,
)

# -----------------------------
# SII / FCFI / GAP CSVs + PNGs
# -----------------------------
sii_csv = find_first(
    patterns=[
        "tokyo_sii.csv",
        "*sii*.csv",
    ],
    search_roots=search_roots,
)

fcfi_csv = find_first(
    patterns=[
        "tokyo_fcfi.csv",
        "*fcfi*.csv",
    ],
    search_roots=search_roots,
)

sii_fcfi_gap_csv = find_first(
    patterns=[
        "tokyo_sii_fcfi_gap.csv",
        "*sii*fcfi*gap*.csv",
        "*gap*.csv",
    ],
    search_roots=search_roots,
)

sii_img = find_first(
    patterns=[
        "tokyo_sii_map.png",
        "*sii*map*.png",
    ],
    search_roots=search_roots,
)

fcfi_img = find_first(
    patterns=[
        "tokyo_fcfi_map.png",
        "*fcfi*map*.png",
    ],
    search_roots=search_roots,
)

sii_fcfi_gap_img = find_first(
    patterns=[
        "tokyo_sii_fcfi_gap_map.png",
        "*sii*fcfi*gap*map*.png",
        "*gap*map*.png",
    ],
    search_roots=search_roots,
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Tokyo viewer")
n_show = st.sidebar.slider("Rows to display in rankings", min_value=5, max_value=23, value=10)

st.sidebar.markdown("---")
use_hover = st.sidebar.toggle(
    "Interactive hover map",
    value=True,
    help="Shows ward tooltips + IRI value (requires GeoJSON + Plotly).",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Detected outputs")
st.sidebar.write("Wards GeoJSON:", str(wards_geojson) if wards_geojson else "❌ not found")
st.sidebar.write("Designed CSV:", str(designed_csv) if designed_csv else "❌ not found")
st.sidebar.write("Designed map (PNG fallback):", str(designed_img) if designed_img else "❌ not found")
st.sidebar.write("PCA CSV:", str(pca_csv) if pca_csv else "❌ not found")
st.sidebar.write("PCA map (PNG fallback):", str(pca_img) if pca_img else "❌ not found")
st.sidebar.write("LISA map (PNG fallback):", str(lisa_img) if lisa_img else "❌ not found")
st.sidebar.write("Difference map (PNG fallback):", str(diff_img) if diff_img else "❌ not found")
st.sidebar.write("ML LISA map (PNG):", str(ml_lisa_img) if ml_lisa_img else "❌ not found")
st.sidebar.write("ML−Structural LISA map (PNG):", str(ml_diff_lisa_img) if ml_diff_lisa_img else "❌ not found")
st.sidebar.write("LISA results CSV:", str(lisa_csv) if lisa_csv else "❌ not found")
st.sidebar.write("ML difference CSV:", str(ml_diff_csv) if ml_diff_csv else "❌ not found")
st.sidebar.write("SII CSV:", str(sii_csv) if sii_csv else "❌ not found")
st.sidebar.write("FCFI CSV:", str(fcfi_csv) if fcfi_csv else "❌ not found")
st.sidebar.write("SII−FCFI gap CSV:", str(sii_fcfi_gap_csv) if sii_fcfi_gap_csv else "❌ not found")
st.sidebar.write("SII map (PNG):", str(sii_img) if sii_img else "❌ not found")
st.sidebar.write("FCFI map (PNG):", str(fcfi_img) if fcfi_img else "❌ not found")
st.sidebar.write("SII−FCFI gap map (PNG):", str(sii_fcfi_gap_img) if sii_fcfi_gap_img else "❌ not found")

if use_hover and not _HAS_PLOTLY:
    st.sidebar.warning("Plotly not installed. Run: `pip install plotly`")


# -----------------------------
# Header / write-up
# -----------------------------
st.title("Urban Social Isolation and Aging in Tokyo")
st.subheader("A Ward-Level Spatial Analysis Using Designed and PCA-Based Indices")

st.markdown(
    """
This interactive dashboard presents a spatial analysis of **social isolation risk**
across **Tokyo’s 23 Special Wards**, using two complementary composite indices:

- **Designed Isolation Index (D-IRI)** — a theory-driven weighted index  
- **PCA-Based Isolation Index** — a data-driven index derived from principal component analysis  

All values are standardized and interpreted **relative within Tokyo**.
"""
)

with st.expander("📘 Overview", expanded=True):
    st.markdown(
        """
Japan’s rapid population aging and persistent urbanization raise urgent questions
about **social isolation**, particularly among older adults in metropolitan regions.
Tokyo exhibits substantial variation in demographic structure, economic conditions,
and transportation access across its wards.

This project introduces and visualizes two complementary isolation indices to
identify **spatial patterns of relative isolation risk** and to support
research, teaching, and policy discussion.
"""
    )

with st.expander("📊 Data Sources"):
    st.markdown(
        """
### Primary Statistical Source
**Statistics Bureau of Japan (e-Stat Portal)**  
https://www.e-stat.go.jp/

Ward-level indicators were compiled from official census and administrative tables,
including population age structure, household composition, and economic indicators.

### Transit + Spatial Boundaries (GIS)
**National Land Numerical Information (NLNI) / MLIT, Japan**  
https://nlftp.mlit.go.jp/ksj/

Ward boundary geometries and supporting GIS layers were derived from official Japanese administrative GIS data.
Transit access metrics in this project are computed from these spatial sources (e.g., station layers joined to ward polygons).
"""
    )

with st.expander("🧠 Conceptualizing Social Isolation"):
    st.markdown(
        """
Social isolation is not directly observable. Instead, it is inferred from
**structural risk factors** that increase the likelihood of reduced mobility,
limited social contact, and economic vulnerability, especially among older adults.

This analysis focuses on four key dimensions:

1. **Population aging**
2. **Single-elderly household prevalence**
3. **Economic disadvantage**
4. **Access to public transportation**

Each captures a distinct pathway through which isolation risk may emerge in urban contexts.
"""
    )

with st.expander("🧮 Designed Isolation Index (D-IRI)"):
    st.markdown(
        r"""
### Motivation
The Designed Isolation Index (D-IRI) is a **theory-driven composite index**.
Weights are selected deliberately to reflect a transparent, policy-interpretable structure.

### Mathematical definition
Let each input be standardized across wards (z-scores). Because greater transit access
tends to *reduce* isolation risk, its standardized value is sign-reversed.

\[
\text{D-IRI}_i =
0.25\,z(\text{Age65+})_i
+ 0.25\,z(\text{Single Elderly})_i
+ 0.20\,z(\text{Poverty})_i
- 0.15\,z(\text{Transit})_i
\]

### Interpretation
- Higher values = **greater relative isolation risk** (within Tokyo)
- Lower values = **lower relative risk**
- Scores are comparative, not clinical diagnoses
"""
    )

with st.expander("📐 PCA-Based Isolation Index"):
    st.markdown(
        """
### Motivation
The PCA index provides a **data-driven** view of the same indicators by identifying the
dominant shared pattern across variables.

### Method
- Use standardized inputs
- Compute PCA (typically on the correlation matrix)
- Interpret **PC1** as a latent “isolation dimension”
- Standardize PC1 scores for interpretability

### Interpretation
- PCA weights emerge from covariance structure, not theory
- Comparing PCA vs D-IRI helps identify where “data structure” diverges from “theory weighting”
"""
    )

with st.expander("🗺 Spatial Patterns and Interpretation"):
    st.markdown(
        """
Mapping both indices often reveals clear spatial structure (i.e., not random noise):

- Some areas cluster into higher-risk patterns
- Some areas cluster into lower-risk patterns
- D-IRI and PCA can overlap, but do not have to match perfectly

Spatial statistics (next section) formalize whether the clustering is statistically meaningful.
"""
    )

with st.expander("🔍 Why Use Two Indices?"):
    st.markdown(
        """
Using both indices strengthens interpretability and robustness:

| Designed Index (D-IRI) | PCA Index |
|---|---|
| Theory-driven | Data-driven |
| Transparent weighting | Emergent weighting |
| Policy-interpretable | Diagnostic / exploratory |

Agreement increases confidence; divergence signals areas for deeper investigation.
"""
    )

# ML notes in main narrative
render_ml_notes()

with st.expander("🌍 Real-World Significance"):
    st.markdown(
        """
Understanding spatial variation in isolation risk supports:

- Public health and aging policy
- Transportation and accessibility planning
- Targeted social services and community programs
- Disaster preparedness for vulnerable populations

These indices function as **diagnostic tools** to support evidence-based, place-sensitive intervention.
"""
    )

with st.expander("⚠️ Limitations"):
    st.markdown(
        """
- Indices are relative measures (within Tokyo), not absolute “risk scores”
- Administrative indicators cannot fully capture lived social experience
- Structural indicators do not imply causality
- Results depend on available official statistics and GIS layers
"""
    )

with st.expander("✅ Conclusion"):
    st.markdown(
        """
By combining demographic theory, statistical standardization, dimensionality reduction,
and spatial visualization, this project highlights meaningful heterogeneity in social
isolation risk across Tokyo’s wards.
"""
    )

with st.expander("What these maps show (plain-language explanation)", expanded=False):
    st.markdown(
        textwrap.dedent(
            """
            These maps show the distribution of a **risk of social isolation index** across **Tokyo’s 23 Special Wards**.

            **Two index versions**
            - **Designed index (D-IRI):** a theory-driven weighted combination of standardized (z-score) indicators.
            - **PCA index:** a data-driven composite derived from the same standardized indicators.

            **How to read the colors**
            - Lighter / warmer colors indicate wards with **higher relative risk** in the index.
            - Darker / cooler colors indicate wards with **lower relative risk**.
            """
        )
    )

# -----------------------------
# Tabs
# -----------------------------
tab_designed, tab_pca, tab_context, tab_ml, tab_spatial = st.tabs(
    [
        "Designed index (D-IRI)",
        "PCA index",
        "Isolation & Family Context",
        "ML maps (risk + difference)",
        "Spatial analysis",
    ]
)


def render_index_tab(
    *,
    title: str,
    csv_path: Path | None,
    score_col_preferred: str,
    png_fallback: Path | None,
) -> None:
    st.subheader(title)

    c1, c2 = st.columns([1.35, 1])

    with c1:
        st.markdown("### Map")

        if csv_path and csv_path.exists():
            df = safe_read_csv(csv_path)

            if use_hover and wards_geojson and wards_geojson.exists():
                value_col = score_col_preferred if score_col_preferred in df.columns else df.columns[-1]
                render_hover_map(
                    wards_geojson_path=wards_geojson,
                    df=df,
                    value_col=value_col,
                    title=title,
                )
            else:
                if png_fallback and png_fallback.exists():
                    st.image(str(png_fallback), use_container_width=True)
                else:
                    st.warning("No hover map (missing GeoJSON / Plotly) and no PNG fallback found.")
        else:
            if png_fallback and png_fallback.exists():
                st.image(str(png_fallback), use_container_width=True)
            else:
                st.error("Dataset CSV not found and no PNG fallback found.")

    with c2:
        st.markdown("### Dataset (table)")
        if csv_path and csv_path.exists():
            df = safe_read_csv(csv_path)
            st.dataframe(df, use_container_width=True)
        else:
            st.error("Dataset CSV not found.")

    st.markdown("---")
    st.markdown("## Rankings")

    if not (csv_path and csv_path.exists()):
        st.error("Dataset CSV not found (rankings cannot be computed).")
        return

    df = safe_read_csv(csv_path)
    ward_col = infer_ward_col(df)
    score_col = score_col_preferred if score_col_preferred in df.columns else df.columns[-1]
    add_rankings_block(df, ward_col=ward_col, score_col=score_col, n_show=n_show)


def render_spatial_map_block(
    *,
    title: str,
    png_path: Path | None,
    table_csv: Path | None = None,
    table_title: str = "Dataset (table)",
) -> None:
    c1, c2 = st.columns([1.35, 1])

    with c1:
        st.markdown("### Map")
        if png_path and png_path.exists():
            st.image(str(png_path), use_container_width=True)
        else:
            st.error("Spatial map PNG not found. Re-generate it with your spatial scripts.")

    with c2:
        st.markdown(f"### {table_title}")
        if table_csv and table_csv.exists():
            df = safe_read_csv(table_csv)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No table found for this view (optional).")


with tab_designed:
    render_index_tab(
        title="Tokyo D-IRI (z-score composite)",
        csv_path=designed_csv,
        score_col_preferred="iri_designed",
        png_fallback=designed_img,
    )

with tab_pca:
    render_index_tab(
        title="Tokyo PCA-based Isolation Index",
        csv_path=pca_csv,
        score_col_preferred="iri_pca",
        png_fallback=pca_img,
    )

with tab_context:
    st.subheader("Isolation & Family Context")

    st.markdown(
        """
These exploratory indices add demographic and infrastructural context to the main isolation analysis.

- **SII** = Social Isolation Index
- **FCFI** = Family & Child Friendliness Index
- **Gap** = standardized difference between SII and FCFI
"""
    )

    render_sii_fcfi_notes()

    t1, t2, t3 = st.tabs(["SII", "FCFI", "Gap (SII − FCFI)"])

    # -----------------------------
    # SII
    # -----------------------------
    with t1:
        c1, c2 = st.columns([1.35, 1])

        with c1:
            st.markdown("### Map")
            if sii_csv and sii_csv.exists() and use_hover and wards_geojson and wards_geojson.exists():
                df = safe_read_csv(sii_csv)
                value_col = pick_first_existing_col(df, ["SII", "sii", "sii_score", "score"])
                if value_col is None:
                    st.error("Could not find an SII score column.")
                else:
                    render_hover_map(
                        wards_geojson_path=wards_geojson,
                        df=df,
                        value_col=value_col,
                        title="Tokyo Social Isolation Index (SII)",
                        color_scale="PuRd",
                    )
            elif sii_img and sii_img.exists():
                st.image(str(sii_img), use_container_width=True)
            else:
                st.error("SII map not found.")

        with c2:
            st.markdown("### Dataset (table)")
            if sii_csv and sii_csv.exists():
                df = safe_read_csv(sii_csv)
                st.dataframe(df, use_container_width=True)
            else:
                st.error("SII CSV not found.")

        if sii_csv and sii_csv.exists():
            df = safe_read_csv(sii_csv)
            ward_col = infer_ward_col(df)
            score_col = pick_first_existing_col(df, ["SII", "sii", "sii_score", "score"])
            if score_col is not None:
                st.markdown("---")
                st.markdown("## Rankings")
                add_rankings_block_custom(
                    df,
                    ward_col=ward_col,
                    score_col=score_col,
                    n_show=n_show,
                    high_label="Highest SII",
                    low_label="Lowest SII",
                )

    # -----------------------------
    # FCFI
    # -----------------------------
    with t2:
        c1, c2 = st.columns([1.35, 1])

        with c1:
            st.markdown("### Map")
            if fcfi_csv and fcfi_csv.exists() and use_hover and wards_geojson and wards_geojson.exists():
                df = safe_read_csv(fcfi_csv)
                value_col = pick_first_existing_col(df, ["FCFI", "fcfi", "fcfi_score", "score"])
                if value_col is None:
                    st.error("Could not find an FCFI score column.")
                else:
                    render_hover_map(
                        wards_geojson_path=wards_geojson,
                        df=df,
                        value_col=value_col,
                        title="Tokyo Family & Child Friendliness Index (FCFI)",
                        color_scale="Greens",
                    )
            elif fcfi_img and fcfi_img.exists():
                st.image(str(fcfi_img), use_container_width=True)
            else:
                st.error("FCFI map not found.")

        with c2:
            st.markdown("### Dataset (table)")
            if fcfi_csv and fcfi_csv.exists():
                df = safe_read_csv(fcfi_csv)
                st.dataframe(df, use_container_width=True)
            else:
                st.error("FCFI CSV not found.")

        if fcfi_csv and fcfi_csv.exists():
            df = safe_read_csv(fcfi_csv)
            ward_col = infer_ward_col(df)
            score_col = pick_first_existing_col(df, ["FCFI", "fcfi", "fcfi_score", "score"])
            if score_col is not None:
                st.markdown("---")
                st.markdown("## Rankings")
                add_rankings_block_custom(
                    df,
                    ward_col=ward_col,
                    score_col=score_col,
                    n_show=n_show,
                    high_label="Most family-friendly",
                    low_label="Least family-friendly",
                )

    # -----------------------------
    # GAP
    # -----------------------------
    with t3:
        c1, c2 = st.columns([1.35, 1])

        with c1:
            st.markdown("### Map")
            if sii_fcfi_gap_csv and sii_fcfi_gap_csv.exists() and use_hover and wards_geojson and wards_geojson.exists():
                df = safe_read_csv(sii_fcfi_gap_csv)
                value_col = pick_first_existing_col(df, ["sii_fcfi_gap", "gap", "sii_minus_fcfi"])
                if value_col is None:
                    st.error("Could not find a gap score column.")
                else:
                    render_hover_map(
                        wards_geojson_path=wards_geojson,
                        df=df,
                        value_col=value_col,
                        title="Tokyo Isolation vs Family-Friendliness Gap (SII − FCFI)",
                        color_scale="RdBu_r",
                        diverging_midpoint=0.0,
                        symmetric_range=True,
                    )
                    st.caption(
                        "Red = isolation higher relative to family support; "
                        "Blue = family-friendliness stronger relative to isolation; White ≈ balanced."
                    )
            elif sii_fcfi_gap_img and sii_fcfi_gap_img.exists():
                st.image(str(sii_fcfi_gap_img), use_container_width=True)
            else:
                st.error("SII−FCFI gap map not found.")

        with c2:
            st.markdown("### Dataset (table)")
            if sii_fcfi_gap_csv and sii_fcfi_gap_csv.exists():
                df = safe_read_csv(sii_fcfi_gap_csv)
                st.dataframe(df, use_container_width=True)
            else:
                st.error("SII−FCFI gap CSV not found.")

        if sii_fcfi_gap_csv and sii_fcfi_gap_csv.exists():
            df = safe_read_csv(sii_fcfi_gap_csv)
            ward_col = infer_ward_col(df)
            score_col = pick_first_existing_col(df, ["sii_fcfi_gap", "gap", "sii_minus_fcfi"])
            if score_col is not None:
                st.markdown("---")
                st.markdown("## Rankings")
                add_rankings_block_custom(
                    df,
                    ward_col=ward_col,
                    score_col=score_col,
                    n_show=n_show,
                    high_label="Largest positive gap",
                    low_label="Largest negative gap",
                )

with tab_ml:
    st.subheader("ML risk + Difference (ML − Structural)")

    if not (ml_diff_csv and ml_diff_csv.exists()):
        st.warning(
            "ML difference CSV not found. Run:\n"
            "  python ml/scripts/06_make_difference_map.py\n"
            "Then refresh this page."
        )
    else:
        dfm = safe_read_csv(ml_diff_csv)

        if "ward_jis" not in dfm.columns:
            st.error("Expected `ward_jis` column in ML difference CSV.")
        else:
            for c in ["iri_designed_z", "iri_ml_z", "iri_diff", "iri_designed", "iri_ml_score"]:
                if c in dfm.columns:
                    dfm[c] = coerce_numeric(dfm[c])

            t1, t2, t3 = st.tabs(["Structural z", "ML z", "Difference (ML − Structural)"])

            if use_hover and wards_geojson and wards_geojson.exists():
                with t1:
                    render_hover_map(
                        wards_geojson_path=wards_geojson,
                        df=dfm,
                        value_col="iri_designed_z",
                        title="Structural isolation risk (z-score)",
                        color_scale="Viridis",
                    )
                with t2:
                    render_hover_map(
                        wards_geojson_path=wards_geojson,
                        df=dfm,
                        value_col="iri_ml_z",
                        title="ML isolation risk (z-score)",
                        color_scale="Viridis",
                    )
                with t3:
                    render_hover_map(
                        wards_geojson_path=wards_geojson,
                        df=dfm,
                        value_col="iri_diff",
                        title="Difference in isolation risk (ML − Structural)",
                        color_scale="RdBu_r",
                        diverging_midpoint=0.0,
                        symmetric_range=True,
                    )
                    st.caption(
                        "Red = ML flags higher risk than the structural index; "
                        "Blue = structural index flags higher risk than ML; White ≈ agreement."
                    )
            else:
                st.info("Enable the interactive hover map toggle (and ensure GeoJSON is detected) to render these maps.")

            st.markdown("---")
            st.markdown("## Biggest disagreements (auto-annotated)")

            label_col = infer_ward_col(dfm)
            if label_col not in dfm.columns:
                label_col = "ward_jis"

            top_ml = dfm.sort_values("iri_diff", ascending=False).head(3)
            top_struct = dfm.sort_values("iri_diff", ascending=True).head(3)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### ML >> Structural (positive difference)")
                st.dataframe(
                    top_ml[[label_col, "ward_jis", "iri_diff", "iri_ml_score", "iri_designed"]],
                    use_container_width=True,
                )

            with c2:
                st.markdown("### Structural >> ML (negative difference)")
                st.dataframe(
                    top_struct[[label_col, "ward_jis", "iri_diff", "iri_ml_score", "iri_designed"]],
                    use_container_width=True,
                )

    render_ml_notes()


with tab_spatial:
    with st.expander("What these spatial maps mean (definitions + why it matters)", expanded=True):
        st.markdown("### Why spatial statistics?")
        st.markdown(
            "Spatial autocorrelation asks whether **similar values occur near each other** more than expected by chance."
        )

        st.markdown("---")
        st.markdown("## 1) Global Moran’s I (overall clustering)")

        st.markdown("**Definitions**")
        st.markdown(
            """
- **yᵢ**: index value in ward *i*  
- **ȳ**: mean index value across wards  
- **s**: standard deviation across wards  
- **zᵢ = (yᵢ − ȳ)/s**: standardized value (z-score)  
- **wᵢⱼ**: spatial weight between wards *i* and *j* (e.g., 1 if they share a border, else 0)  
- **n**: number of wards  
"""
        )

        st.markdown("A common form of **Global Moran’s I** is:")
        st.latex(
            r"""
I = \frac{n}{\sum_{i}\sum_{j} w_{ij}}
\cdot
\frac{\sum_{i}\sum_{j} w_{ij} (y_i - \bar{y})(y_j - \bar{y})}
{\sum_{i} (y_i - \bar{y})^2}
"""
        )

        st.markdown("Equivalently, in z-score form:")
        st.latex(
            r"""
I = \frac{n}{\sum_{i}\sum_{j} w_{ij}}
\cdot
\frac{\sum_{i}\sum_{j} w_{ij} z_i z_j}
{\sum_{i} z_i^2}
"""
        )

        st.markdown(
            """
**Interpretation**
- **I > 0**: similar values cluster together (high with high, low with low)  
- **I ≈ 0**: pattern resembles random spatial arrangement  
- **I < 0**: neighbors tend to be dissimilar (checkerboard pattern)  
"""
        )

        st.markdown("---")
        st.markdown("## 2) Local Moran’s I (LISA: where clustering occurs)")

        st.markdown("A common local statistic is:")
        st.latex(r""" I_i = z_i \sum_{j} w_{ij} z_j """)

        st.markdown(
            """
Local Moran’s I identifies *where* clustering/outliers occur.
Significance is usually assessed by permutation tests (p-values).

### LISA cluster types (when significant)
- **High–High (HH):** high ward surrounded by high neighbors (“hot spot”)  
- **Low–Low (LL):** low ward surrounded by low neighbors (“cold spot”)  
- **High–Low (HL):** high ward surrounded by low neighbors (outlier)  
- **Low–High (LH):** low ward surrounded by high neighbors (outlier)  
- **Not significant:** no strong evidence of local clustering/outlier behavior  
"""
        )

        st.markdown("---")
        st.markdown("## 3) D-IRI minus PCA (where theory vs data disagree)")
        st.latex(r""" \Delta_i = \mathrm{D\text{-}IRI}_i - \mathrm{PCA}_i """)

        st.markdown(
            """
- **Δ > 0:** Designed index rates the ward higher than PCA  
- **Δ < 0:** PCA rates the ward higher than Designed index  
"""
        )

        spatial_lisa_tab, spatial_diff_tab, spatial_ml_tab = st.tabs(
            ["LISA cluster map", "D-IRI minus PCA", "ML spatial diagnostics"]
        )

        with spatial_lisa_tab:
            st.markdown("### Tokyo Designed Isolation Index (D-IRI) — LISA Cluster Map")
            render_spatial_map_block(
                title="Tokyo D-IRI LISA cluster map",
                png_path=lisa_img,
                table_csv=lisa_csv,
                table_title="LISA results (table)",
            )

        with spatial_diff_tab:
            st.markdown("### Tokyo: D-IRI minus PCA (difference map)")
            render_spatial_map_block(
                title="Tokyo D-IRI minus PCA",
                png_path=diff_img,
                table_csv=None,
                table_title="Dataset (optional)",
            )

        with spatial_ml_tab:
            st.markdown("### ML Spatial Diagnostics (Local Moran’s I / LISA)")
            st.caption(
                "These maps replicate the same LISA cluster workflow, but computed on: "
                "(1) ML isolation risk z-scores, and (2) the difference (ML − Structural)."
            )

            spatial_ml_dir = ROOT / "out" / "spatial_ml"
            ml_png_expected = spatial_ml_dir / "tokyo_ml_lisa_clusters.png"
            diff_png_expected = spatial_ml_dir / "tokyo_diff_lisa_clusters.png"

            ml_png = ml_png_expected if ml_png_expected.exists() else ml_lisa_img
            diff_png = diff_png_expected if diff_png_expected.exists() else ml_diff_lisa_img

            def _show_png_with_download(path: Path | None, caption: str):
                if not (path and path.exists()):
                    st.error("ML spatial PNG not found.")
                    st.code(
                        "Run (PowerShell):\n"
                        ".\\.venv\\Scripts\\python.exe ml\\scripts\\ml_spatial.py `\n"
                        "  --ml-path out\\ml\\maps\\difference_map_data.csv `\n"
                        "  --wards-geojson data\\external\\jp_tokyo_wards.geojson `\n"
                        "  --out-dir out\\spatial_ml\n"
                        "\nThen refresh Streamlit."
                    )
                    return

                st.image(str(path), use_container_width=True, caption=caption)

                with open(path, "rb") as f:
                    st.download_button(
                        label=f"Download: {path.name}",
                        data=f,
                        file_name=path.name,
                        mime="image/png",
                        use_container_width=True,
                    )

            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("#### ML isolation risk (z-score) — LISA")
                _show_png_with_download(
                    ml_png,
                    caption="Local Moran’s I on ML z-scores. Red = High–High hotspots; gray = not significant.",
                )

            with c2:
                st.markdown("#### (ML − Structural) difference — LISA")
                _show_png_with_download(
                    diff_png,
                    caption="Local Moran’s I on (ML − Structural). If all gray, disagreements are not spatially clustered.",
                )

            with st.expander("How to interpret these (quick)", expanded=False):
                st.markdown(
                    """
- **High–High (HH):** high value surrounded by high neighbors (hotspot cluster)
- **Low–Low (LL):** low value surrounded by low neighbors (coldspot cluster)
- **High–Low / Low–High:** spatial outliers (ward differs from neighbors)
- **Not significant:** no strong evidence of local clustering/outlier behavior

**For the difference map** where `diff = ML_z − Structural_z`:
- **HH:** ML systematically higher than Structural in that region
- **LL:** Structural systematically higher than ML in that region
- **All gray:** disagreements are not spatially systematic (ward-specific)
"""
                )

st.caption("Tip: regenerate CSVs/plots with your pipeline scripts, then refresh this page.")
