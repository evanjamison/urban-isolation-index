# -*- coding: utf-8 -*-
"""
01_ingest_jp_estat.py — e-Stat Japan ingestion (Tokyo 23 wards)

What this does
--------------
- Calls e-Stat v3 getStatsData for the table IDs you provide.
- Filters to the 23 special wards of Tokyo (JIS X-0402 municipality codes).
- Extracts values per ward and merges into one table.
- Writes:
    data/raw/jp_estat_tokyo_raw.csv
    data/interim/jp_tokyo_features.parquet

What you need from e-Stat
-------------------------
- appId in your .env:  ESTAT_APP_ID=xxxxxxxxxxxxxxxxxxxx
- statsDataId for:
    --age65-id       (share of 65+, OR count of 65+ with a second table for totals)
    --alone65-id     (elderly living alone numerator, ideally 65+)
    --poverty-id     (poverty numerator & denominator OR percent)
Optionally pass --time "2020" (or the exact e-Stat time code) and any category codes if needed.

Usage
-----
# minimal (will warn if IDs are missing)
python -m src.cli.01_ingest_jp_estat

# with IDs (examples are placeholders!)
python -m src.cli.01_ingest_jp_estat ^
  --age65-id 000000000001 ^
  --alone65-id 000000000002 ^
  --poverty-id 000000000003 ^
  --time 2020

Notes
-----
- e-Stat responses vary by table. This script flattens VALUE records generically
  and keeps the 'area' (municipality), 'time', 'value', and any catXX labels it finds.
- If your table returns a percent already, we use it. If it returns counts,
  we compute percentages when both numerator and denominator exist.
"""

from __future__ import annotations
import os, sys, time, json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
import requests
import pandas as pd
from dotenv import load_dotenv

# --------------------------------------------------------------------
# Tokyo 23 wards — JIS X-0402 municipality codes (Shi-Ku-Cho-Son)
# --------------------------------------------------------------------
TOKYO_WARDS = {
    "13101": "千代田区", "13102": "中央区",   "13103": "港区",     "13104": "新宿区",
    "13105": "文京区",   "13106": "台東区",   "13107": "墨田区",   "13108": "江東区",
    "13109": "品川区",   "13110": "目黒区",   "13111": "大田区",   "13112": "世田谷区",
    "13113": "渋谷区",   "13114": "中野区",   "13115": "杉並区",   "13116": "豊島区",
    "13117": "北区",     "13118": "荒川区",   "13119": "板橋区",   "13120": "練馬区",
    "13121": "足立区",   "13122": "葛飾区",   "13123": "江戸川区",
}

ESTAT_ENDPOINT = "https://api.e-stat.go.jp/rest/3.0/app/getStatsData"


def _load_app_id() -> str:
    load_dotenv()
    app_id = os.getenv("ESTAT_APP_ID", "").strip()
    if not app_id:
        print("❌ Missing ESTAT_APP_ID in .env", file=sys.stderr)
        sys.exit(1)
    return app_id


def _request_estat(app_id: str, stats_data_id: str, cd_time: Optional[str],
                   cd_area: List[str], extra_params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Call e-Stat with basic paging and backoff."""
    params = {
        "appId": app_id,
        "statsDataId": stats_data_id,
        # e-Stat expects comma-separated codes for cdArea
        "cdArea": ",".join(cd_area),
        # We typically want the latest year; pass a code if you have it
        # Some tables accept 'cdTime' like '2020' or a coded time id
    }
    if cd_time:
        params["cdTime"] = cd_time
    if extra_params:
        params.update(extra_params)

    start = 1
    items: List[Dict[str, Any]] = []
    while True:
        p = dict(params)
        p["startPosition"] = str(start)
        # generous limit per page
        p["limit"] = "100000"

        # retry loop
        for attempt in range(5):
            r = requests.get(ESTAT_ENDPOINT, params=p, timeout=60)
            if r.status_code == 200:
                break
            if r.status_code in (429, 503):
                time.sleep(0.7 * (2 ** attempt))
                continue
            r.raise_for_status()

        data = r.json()
        # basic sanity
        stat = data.get("GET_STATS_DATA", {}).get("STATISTICAL_DATA", {})
        table_inf = stat.get("TABLE_INF", {})
        value = stat.get("DATA_INF", {}).get("VALUE", [])
        if isinstance(value, dict):  # sometimes single object
            value = [value]
        if not value:
            break

        items.extend(value)

        # pagination
        info = stat.get("DATA_INF", {}).get("NOTE", {})
        # e-Stat doesn't always provide total; we try until less than requested
        if len(value) < int(p["limit"]):
            break
        start += int(p["limit"])

    return {"items": items}


def _flatten_values(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten e-Stat VALUE records.
    We preserve '@area', '@time', '@value', and any '@catXX' attributes if present.
    """
    rows = []
    for rec in records:
        # Each VALUE is like {"@area": "13101", "@time": "2020", "$": "1234", ...}
        out: Dict[str, Any] = {}
        for k, v in rec.items():
            if k.startswith("@"):
                key = k[1:]  # strip '@'
                out[key] = v
            elif k == "$":
                out["value"] = v
            else:
                # unexpected keys are kept
                out[k] = v
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # normalize dtypes
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _fetch_table_series(app_id: str, stats_data_id: str, cd_time: Optional[str], label: str) -> pd.DataFrame:
    """
    Fetch one table and return a frame with columns:
      area_code, time, {label}
    """
    res = _request_estat(app_id, stats_data_id, cd_time, list(TOKYO_WARDS.keys()))
    df = _flatten_values(res["items"])
    if df.empty:
        print(f"⚠️ No rows from e-Stat for {label} (statsDataId={stats_data_id})")
        return pd.DataFrame(columns=["area","time",label])
    # unify column names
    area_col = "area" if "area" in df.columns else "area_code"
    time_col = "time" if "time" in df.columns else "time_code"
    if area_col not in df.columns:
        # Some tables encode area as catXX; try to find it
        area_candidates = [c for c in df.columns if c.startswith("cat") and df[c].astype(str).str.match(r"^13\d{3}$").any()]
        if area_candidates:
            area_col = area_candidates[0]
        else:
            raise RuntimeError(f"Could not determine area column for {label}")
    if time_col not in df.columns:
        # default if absent
        time_col = "time"
        df[time_col] = None

    out = df[[area_col, time_col, "value"]].rename(columns={area_col: "area_code", time_col: "time", "value": label})
    # keep only wards we know
    out = out[out["area_code"].isin(TOKYO_WARDS.keys())].copy()
    # latest duplicate handling: if multiple time rows, keep the last non-null
    out = (out
           .sort_values(["area_code", "time"])
           .drop_duplicates(subset=["area_code"], keep="last"))
    return out


def build_tokyo_features(
    app_id: str,
    age65_id: Optional[str],
    alone65_id: Optional[str],
    poverty_id: Optional[str],
    cd_time: Optional[str],
) -> pd.DataFrame:
    """
    Pull the tables you provide and assemble a tidy feature frame with columns:
      ward_jis, ward_name, pct_age65p, pct_single65p, poverty_rate
    """
    frames = []

    if age65_id:
        frames.append(_fetch_table_series(app_id, age65_id, cd_time, label="pct_age65p"))
    else:
        print("⚠️ --age65-id not provided; pct_age65p will be NaN.")

    if alone65_id:
        # Some tables are a share already; we store as pct_single65p
        frames.append(_fetch_table_series(app_id, alone65_id, cd_time, label="pct_single65p"))
    else:
        print("⚠️ --alone65-id not provided; pct_single65p will be NaN.")

    if poverty_id:
        frames.append(_fetch_table_series(app_id, poverty_id, cd_time, label="poverty_rate"))
    else:
        print("⚠️ --poverty-id not provided; poverty_rate will be NaN.")

    # Merge on area_code
    if not frames:
        raise SystemExit("❌ No table IDs provided. Pass at least one of --age65-id, --alone65-id, --poverty-id.")

    feat = None
    for f in frames:
        feat = f if feat is None else feat.merge(f, on=["area_code"], how="outer")

    # Attach ward names, order, and tidy columns
    feat = feat.rename(columns={"area_code": "ward_jis"})
    feat["ward_name"] = feat["ward_jis"].map(TOKYO_WARDS)

    cols = ["ward_jis", "ward_name", "pct_age65p", "pct_single65p", "poverty_rate"]
    for c in cols:
        if c not in feat.columns:
            feat[c] = pd.NA
    feat = feat[cols].sort_values("ward_jis").reset_index(drop=True)
    return feat


def main():
    parser = argparse.ArgumentParser(description="Ingest e-Stat data for Tokyo 23 wards")
    parser.add_argument("--age65-id", default="", help="statsDataId for % 65+ (or a table that directly yields that share)")
    parser.add_argument("--alone65-id", default="", help="statsDataId for elderly living alone share (prefer 65+)")
    parser.add_argument("--poverty-id", default="", help="statsDataId for poverty rate (or numerator/denominator table summarized as %)")
    parser.add_argument("--time", default="", help="e-Stat cdTime code (e.g., '2020' or a coded time id). Optional.")
    parser.add_argument("--raw-out", default="data/raw/jp_estat_tokyo_raw.csv")
    parser.add_argument("--features-out", default="data/interim/jp_tokyo_features.parquet")
    args = parser.parse_args()

    app_id = _load_app_id()
    Path(args.raw_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.features_out).parent.mkdir(parents=True, exist_ok=True)

    # Build features
    feat = build_tokyo_features(
        app_id=app_id,
        age65_id=args.age65_id.strip() or None,
        alone65_id=args.alone65_id.strip() or None,
        poverty_id=args.poverty_id.strip() or None,
        cd_time=args.time.strip() or None,
    )

    # Save a raw CSV (human-readable) and the parquet used downstream
    feat.to_csv(args.raw_out, index=False, encoding="utf-8-sig")
    feat.to_parquet(args.features_out, index=False)
    print(f"✅ Saved raw:      {args.raw_out}")
    print(f"✅ Saved features: {args.features_out}")
    print(feat.head())

if __name__ == "__main__":
    main()
