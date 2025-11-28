# -*- coding: utf-8 -*-
"""
08_ingest_tokyo_parks.py

Ingest a Tokyo parks dataset and aggregate to the 23 special wards.

What this script does
---------------------
- Loads ward boundaries for Tokyo (23 wards) from a GeoJSON.
- Loads a parks dataset (points or polygons).
- Spatially joins parks to wards.
- Computes ward-level features:
    * ward_area_km2
    * n_parks
    * parks_per_km2
    * total_park_area_m2      (if parks are polygons)
    * park_area_per_km2       (if polygons)
- Saves a tidy table to:
    data/interim/jp_tokyo_parks_features.parquet

Usage (from project root)
-------------------------
Example with defaults:

  .\\.venv\\Scripts\\python.exe -m src.cli.08_ingest_tokyo_parks

Custom paths:

  .\\.venv\\Scripts\\python.exe -m src.cli.08_ingest_tokyo_parks ^
    --wards-geojson data/external/jp_tokyo_wards.geojson ^
    --parks-path data/raw/tokyo_parks.geojson ^
    --out-path data/interim/jp_tokyo_parks_features.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_wards(wards_path: Path) -> gpd.GeoDataFrame:
    if not wards_path.is_file():
        raise FileNotFoundError(f"Ward GeoJSON not found: {wards_path}")

    wards = gpd.read_file(wards_path)

    # Expect a ward code column; common in your project: N03_007
    if "N03_007" not in wards.columns:
        raise KeyError(
            "Expected column 'N03_007' in wards file. "
            "Inspect the file to confirm the correct ward-code column."
        )

    wards = wards.copy()
    wards["ward_jis"] = wards["N03_007"].astype(str).astype(int)

    if wards.crs is None:
        # Tokyo in EPSG:4326 is common; set default if missing
        wards = wards.set_crs("EPSG:4326")

    # Add area in km² using a projected CRS (Web Mercator is fine for relative comparisons)
    wards_merc = wards.to_crs(epsg=3857)
    wards["ward_area_km2"] = wards_merc.geometry.area / 1_000_000.0

    return wards[["ward_jis", "ward_area_km2", "geometry"]]


def load_parks(parks_path: Path) -> gpd.GeoDataFrame:
    if not parks_path.is_file():
        raise FileNotFoundError(f"Parks file not found: {parks_path}")

    # Try reading as vector data with geopandas
    parks = gpd.read_file(parks_path)

    if parks.crs is None:
        # Assume WGS84 if missing (common for open data)
        parks = parks.set_crs("EPSG:4326")

    # Require geometry
    if "geometry" not in parks.columns:
        raise ValueError(
            f"Parks file {parks_path} does not have a 'geometry' column. "
            "Ensure you're using a GIS vector format (GeoJSON, Shapefile, etc.) "
            "or add a conversion step from lat/lon."
        )

    # Drop rows with missing geometry
    parks = parks[~parks.geometry.is_empty & parks.geometry.notna()].copy()

    if parks.empty:
        raise ValueError("Parks dataset has no valid geometries after cleaning.")

    return parks


def aggregate_parks_to_wards(
    wards: gpd.GeoDataFrame, parks: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Spatially join parks to wards and compute ward-level features.

    Returns a pandas DataFrame indexed by ward_jis with:
      - ward_jis
      - ward_area_km2
      - n_parks
      - parks_per_km2
      - total_park_area_m2 (if polygons)
      - park_area_per_km2  (if polygons)
    """
    # Align CRS
    parks = parks.to_crs(wards.crs)

    # Determine if parks are points or polygons
    geom_types = parks.geometry.geom_type.unique()
    is_polygon = any(gt in ["Polygon", "MultiPolygon"] for gt in geom_types)

    # For area-based features, reproject to metric CRS
    wards_merc = wards.to_crs(epsg=3857)
    parks_merc = parks.to_crs(epsg=3857)

    # Spatial join: which ward each park belongs to
    # We'll use centroids for polygons to avoid sliver issues
    if is_polygon:
        parks_join_geom = parks_merc.geometry.centroid
    else:
        parks_join_geom = parks_merc.geometry

    parks_join = gpd.GeoDataFrame(
        parks_merc.drop(columns="geometry"),
        geometry=parks_join_geom,
        crs=parks_merc.crs,
    )

    joined = gpd.sjoin(
        parks_join,
        wards_merc[["ward_jis", "geometry"]],
        how="inner",
        predicate="within",
    )

    if joined.empty:
        raise ValueError(
            "No parks could be spatially joined to wards. "
            "Check that both datasets cover the same area and CRS."
        )

    # Basic counts
    grp = joined.groupby("ward_jis")
    n_parks = grp.size().rename("n_parks")

    # Start with ward area
    ward_area = wards.set_index("ward_jis")["ward_area_km2"]

    # Base frame
    out = pd.DataFrame({"ward_jis": ward_area.index})
    out["ward_area_km2"] = ward_area.values

    out = out.set_index("ward_jis", drop=False)

    # Attach park counts
    out["n_parks"] = n_parks
    out["n_parks"] = out["n_parks"].fillna(0).astype(int)

    # Density: parks per km²
    out["parks_per_km2"] = out["n_parks"] / out["ward_area_km2"].replace(0, pd.NA)

    # If polygons, compute park area features
    if is_polygon:
        # Use original polygon geometries for area
        parks_poly = parks_merc.loc[
            parks_merc.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ].copy()
        parks_poly["park_area_m2"] = parks_poly.geometry.area

        joined_poly = gpd.sjoin(
            parks_poly,
            wards_merc[["ward_jis", "geometry"]],
            how="inner",
            predicate="within",
        )

        park_area_sum = (
            joined_poly.groupby("ward_jis")["park_area_m2"].sum().rename("total_park_area_m2")
        )

        out["total_park_area_m2"] = park_area_sum
        out["total_park_area_m2"] = out["total_park_area_m2"].fillna(0.0)

        out["park_area_per_km2"] = (
            out["total_park_area_m2"] / (out["ward_area_km2"] * 1_000_000.0)
        )

    return out.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Tokyo parks data to ward-level features."
    )
    parser.add_argument(
        "--wards-geojson",
        default="data/external/jp_tokyo_wards.geojson",
        help="Path to Tokyo 23-ward GeoJSON.",
    )
    parser.add_argument(
        "--parks-path",
        default="data/raw/tokyo_parks.geojson",
        help="Path to parks dataset (GeoJSON/Shapefile/etc.).",
    )
    parser.add_argument(
        "--out-path",
        default="data/interim/jp_tokyo_parks_features.parquet",
        help="Output Parquet path for ward-level park features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    wards_path = Path(args.wards_geojson)
    parks_path = Path(args.parks_path)
    out_path = Path(args.out_path)

    print(f"📥 Loading wards from {wards_path} ...")
    wards = load_wards(wards_path)
    print(f"   Loaded {len(wards)} wards.")

    print(f"📥 Loading parks from {parks_path} ...")
    parks = load_parks(parks_path)
    print(f"   Loaded {len(parks)} parks with valid geometries.")

    print("🔗 Aggregating parks to wards ...")
    features = aggregate_parks_to_wards(wards, parks)

    print(features.head())

    ensure_parent_dir(out_path)
    features.to_parquet(out_path, index=False)
    print(f"💾 Saved ward-level park features to {out_path}")

    print("✅ Done.")


if __name__ == "__main__":
    main()
