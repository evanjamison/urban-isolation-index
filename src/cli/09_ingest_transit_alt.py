"""
09_ingest_transit_alt.py
Alternative ingest: use stations.json, filter to Tokyo, convert to GeoJSON
"""

import json
import geopandas as gpd
import pandas as pd
import argparse
from shapely.geometry import Point

def main(raw_json, wards_geojson, out_csv):
    print("📂 Loading stations json …")
    with open(raw_json, encoding="utf-8") as f:
        data = json.load(f)
    
    rows = []
    for rec in data:
        for st in rec.get("stations", []):
            if st.get("prefecture") == "13":  # Tokyo Prefecture code is 13
                rows.append({
                    "station_code": st.get("code"),
                    "name_kanji": st.get("name_kanji"),
                    "lon": st.get("lon"),
                    "lat": st.get("lat"),
                })
    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

    print("📍 Loading Tokyo wards …")
    wards = gpd.read_file(wards_geojson)

    tokyo_wards = wards[wards["N03_001"] == "東京都"]  # or whichever column you use to filter prefecture

    print("🔗 Spatial join: stations in wards …")
    joined = gpd.sjoin(gdf, tokyo_wards, how="inner", predicate="within")

    counts = joined.groupby("N03_007").size().reset_index(name="station_count")

    merged = tokyo_wards.merge(counts, on="N03_007", how="left")
    merged["station_count"] = merged["station_count"].fillna(0)
    merged["area_km2"] = merged.geometry.to_crs(epsg=3857).area / 1e6
    merged["station_density"] = merged["station_count"] / merged["area_km2"]
    merged["transit_z"] = (merged["station_density"] - merged["station_density"].mean()) / merged["station_density"].std()

    out_df = merged[["N03_007", "station_count", "area_km2", "station_density", "transit_z"]]
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved transit access data to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="stations.json path")
    parser.add_argument("--wards", required=True, help="Tokyo wards geojson")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()
    main(args.raw, args.wards, args.out)
