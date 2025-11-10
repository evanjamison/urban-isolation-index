#!/usr/bin/env python
"""
verify_data.py — quick dataset inspector

Usage examples:
  python scripts/verify_data.py --file data/raw/us_acs_ny_2020.csv
  python scripts/verify_data.py -f data/interim/us_features.parquet --rows 10
  python scripts/verify_data.py -f data/interim/jp_tokyo_features.parquet --describe
  python scripts/verify_data.py -f data/interim/us_features.parquet --value-counts county
  python scripts/verify_data.py -f data/external/nyc_tracts_2020.geojson
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

# Optional geopandas support
def _try_import_geopandas():
    try:
        import geopandas as gpd  # type: ignore
        return gpd
    except Exception:
        return None

def _read_any(path: Path):
    ext = path.suffix.lower()
    gpd = _try_import_geopandas()
    # geospatial first if possible
    if gpd and ext in {".geojson", ".gpkg", ".shp"}:
        return gpd.read_file(path)
    # tabular
    import pandas as pd
    if ext in {".csv", ".txt"}:
        return pd.read_csv(path)
    if ext in {".parquet", ".pq"}:
        # if it’s a GeoParquet, geopandas can also read it
        if gpd:
            try:
                return gpd.read_parquet(path)
            except Exception:
                pass
        return pd.read_parquet(path)
    if ext in {".feather", ".ft"}:
        return pd.read_feather(path)
    # fallback: try geopandas then pandas
    if gpd:
        try:
            return gpd.read_file(path)
        except Exception:
            pass
    return pd.read_table(path)

def main():
    p = argparse.ArgumentParser(description="Quick dataset inspector")
    p.add_argument("-f", "--file", required=True, help="Path to CSV/Parquet/GeoJSON/etc.")
    p.add_argument("--rows", type=int, default=5, help="Number of head rows to show (default: 5)")
    p.add_argument("--describe", action="store_true", help="Also print numeric .describe()")
    p.add_argument("--columns", action="store_true", help="Print column list only and exit")
    p.add_argument("--value-counts", dest="vc", default=None, help="Column to run value_counts() on")
    args = p.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"❌ File not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = _read_any(path)

    # GeoDataFrame vs DataFrame
    is_geo = False
    crs = None
    try:
        import geopandas as gpd  # type: ignore
        is_geo = isinstance(df, gpd.GeoDataFrame)
        if is_geo:
            crs = df.crs
    except Exception:
        pass

    print(f"\n📄 File: {path}")
    print(f"Type: {type(df).__name__}")
    print(f"Shape: {getattr(df, 'shape', None)}")
    if is_geo:
        print(f"CRS: {crs}")
        if 'geometry' in df:
            geom_types = df.geometry.geom_type.value_counts().to_dict()
            print(f"Geometry types: {geom_types}")

    if args.columns:
        print("\n📋 Columns:")
        print(list(df.columns))
        return

    # Head
    print(f"\n🔎 head({args.rows}):")
    try:
        print(df.head(args.rows).to_string(index=False))
    except Exception:
        print(df.head(args.rows))

    # Missing values quick view
    try:
        na = df.isna().sum()
        nonzero_na = na[na > 0]
        print("\n🚩 Missing values (non-zero only):")
        if len(nonzero_na):
            print(nonzero_na.sort_values(ascending=False).to_string())
        else:
            print("None")
    except Exception:
        pass

    # value_counts
    if args.vc:
        col = args.vc
        if col in df.columns:
            print(f"\n📊 value_counts('{col}'):")
            print(df[col].value_counts(dropna=False).head(25).to_string())
        else:
            print(f"\n⚠️ Column not found for value_counts: {col}", file=sys.stderr)

    # describe
    if args.describe:
        try:
            print("\n📈 describe():")
            print(df.describe(include='all', percentiles=[0.05,0.5,0.95]).transpose().to_string())
        except Exception:
            pass

if __name__ == "__main__":
    main()
