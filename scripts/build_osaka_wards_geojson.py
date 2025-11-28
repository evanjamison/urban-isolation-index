# scripts/build_osaka_wards_geojson.py
"""
Build a single Osaka wards GeoJSON file from many small .json ward files.

Assumptions:
- Each file in data/external/osaka_wards_raw is a GeoJSON FeatureCollection
  for a single ward.
- The filename stem is the ward JIS code, e.g. 27102.json -> "27102".

Output:
- data/external/jp_osaka_wards.geojson with one feature per ward
  and a column N03_007 holding the ward JIS code (same as Tokyo file).
"""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

RAW_DIR = Path("data/external/osaka_wards_raw")
OUT_FILE = Path("data/external/jp_osaka_wards.geojson")


def load_single_json(path: Path) -> gpd.GeoDataFrame:
    """Load one ward JSON and tag it with its ward code from the filename."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "features" not in data:
        raise ValueError(f"{path} is not a FeatureCollection (no 'features' key)")

    gdf = gpd.GeoDataFrame.from_features(data["features"])

    # Use filename stem as ward code, e.g. 27102.json -> "27102"
    ward_code = path.stem
    gdf["N03_007"] = ward_code

    return gdf


def main() -> None:
    files = sorted(list(RAW_DIR.glob("*.json")) + list(RAW_DIR.glob("*.geojson")))

    if not files:
        raise FileNotFoundError(f"No .json/.geojson files found in {RAW_DIR}")

    frames = []

    for fp in files:
        try:
            gdf = load_single_json(fp)
            frames.append(gdf)
            print(f"Loaded {fp.name} with {len(gdf)} features, ward {gdf['N03_007'].iloc[0]}")
        except Exception as e:
            print(f"⚠️ Skipping {fp.name}: {e}")

    if not frames:
        raise RuntimeError("No valid ward geometries were loaded for Osaka.")

    combined = pd.concat(frames, ignore_index=True)
    wards = gpd.GeoDataFrame(combined, geometry="geometry")

    # Ensure CRS is set; most national datasets are WGS84
    if wards.crs is None:
        wards = wards.set_crs("EPSG:4326")

    print("Final GeoJSON columns:", list(wards.columns))

    wards.to_file(OUT_FILE, driver="GeoJSON")
    print(f"✅ Wrote {OUT_FILE} with {len(wards)} features")


if __name__ == "__main__":
    main()
