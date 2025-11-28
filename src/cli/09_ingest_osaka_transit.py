"""
09_ingest_osaka_transit.py
Build Osaka transit access metrics using spatial join (same method as Tokyo).
"""

import json
import argparse
import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.geometry import Point


def main(raw_json, wards_geojson, out_csv):
    print("📂 Loading stations JSON …")
    with open(raw_json, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for rec in data:
        for st in rec.get("stations", []):
            # Osaka prefecture = "27"
            if st.get("prefecture") == "27":
                rows.append({
                    "station_code": st.get("code"),
                    "name_kanji": st.get("name_kanji"),
                    "lon": st.get("lon"),
                    "lat": st.get("lat"),
                })

    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

    print("📍 Loading Osaka wards GeoJSON …")
    wards = gpd.read_file(wards_geojson)

    # filter prefecture = Osaka (大阪府)
    osaka_wards = wards[wards["N03_001"] == "大阪府"]

    print("🔗 Spatial join: assigning stations → wards …")
    joined = gpd.sjoin(gdf, osaka_wards, how="inner", predicate="within")

    counts = joined.groupby("N03_007").size().reset_index(name="station_count")

    merged = osaka_wards.merge(counts, on="N03_007", how="left")
    merged["station_count"] = merged["station_count"].fillna(0)

    merged["area_km2"] = merged.geometry.to_crs(epsg=3857).area / 1e6
    merged["station_density"] = merged["station_count"] / merged["area_km2"]

    merged["transit_z"] = (
        (merged["station_density"] - merged["station_density"].mean())
        / merged["station_density"].std()
    )

    out_df = merged[
        ["N03_007", "station_count", "area_km2", "station_density", "transit_z"]
    ]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"✅ Saved Osaka transit access metrics → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="stations.json path")
    parser.add_argument("--wards", required=True, help="Osaka wards geojson")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()
    main(args.raw, args.wards, args.out)

