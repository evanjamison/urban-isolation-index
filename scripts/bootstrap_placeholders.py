import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path

Path("data/interim").mkdir(parents=True, exist_ok=True)

def square(x, y, s=0.01):
    return Polygon([(x, y), (x + s, y), (x + s, y + s), (x, y + s)])

tokyo = gpd.GeoDataFrame(
    {
        "ward": ["Chiyoda", "Shinjuku", "Taito"],
        "pct_age65p": [26.2, 24.5, 29.3],
        "pct_single65p": [14.1, 12.4, 15.9],
        "poverty_rate": [8.2, 9.0, 10.1],
        "lib_within_1km": [5, 8, 3],
        "dist_nearest_senior_m": [300, 180, 650],
        "geometry": [
            square(139.75, 35.68),
            square(139.70, 35.69),
            square(139.78, 35.71),
        ],
    },
    crs="EPSG:4326",
)

nyc = gpd.GeoDataFrame(
    {
        "tract": ["Manhattan_1", "Brooklyn_2", "Queens_3"],
        "pct_age65p": [17.0, 20.5, 19.2],
        "pct_single65p": [10.3, 12.8, 11.7],
        "poverty_rate": [15.4, 18.2, 13.6],
        "lib_within_1km": [6, 3, 4],
        "dist_nearest_senior_m": [500, 700, 450],
        "geometry": [
            square(-74.00, 40.71),
            square(-73.94, 40.68),
            square(-73.85, 40.74),
        ],
    },
    crs="EPSG:4326",
)

tokyo.to_parquet("data/interim/jp_tokyo_features.parquet")
nyc.to_parquet("data/interim/us_nyc_features.parquet")
print("✅ Wrote placeholder parquet files to data/interim/")
