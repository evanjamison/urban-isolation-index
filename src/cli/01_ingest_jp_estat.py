# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
01_ingest_jp_estat.py — e-Stat Japan ingestion (Tokyo 23 wards)

Supports:
- A single table that already returns a *percent* for each ward, OR
- A (numerator, denominator) pair of tables from which we compute a percent.

Examples (PowerShell):
  # Using pre-computed-percent tables
  python -m src.cli.01_ingest_jp_estat --age65-id  <ID> --alone65-id <ID> --poverty-id <ID> --time 2020

  # Using num/den pairs + category filters (semi-colon separated key=val)
  python -m src.cli.01_ingest_jp_estat `
    --alone65-num-id <ID> --alone65-den-id <ID> --cats-alone65 "cdCat01=A;cdCat02=B" --time 2020
"""

from __future__ import annotations
import os, sys, time
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import requests
import pandas as pd
from dotenv import load_dotenv

# -------------------------------------------------------------
# Tokyo 23 wards — JIS X-0402 municipality codes
# -------------------------------------------------------------
TOKYO_WARDS = {
    "13101": "千代田区","13102":"中央区","13103":"港区","13104":"新宿区","13105":"文京区","13106":"台東区",
    "13107":"墨田区","13108":"江東区","13109":"品川区","13110":"目黒区","13111":"大田区","13112":"世田谷区",
    "13113":"渋谷区","13114":"中野区","13115":"杉並区","13116":"豊島区","13117":"北区","13118":"荒川区",
    "13119":"板橋区","13120":"練馬区","13121":"足立区","13122":"葛飾区","13123":"江戸川区",
}

ESTAT_ENDPOINT = "https://api.e-stat.go.jp/rest/3.0/app/getStatsData"


# ----------------------- helpers -----------------------
def _load_app_id() -> str:
    load_dotenv()
    app_id = os.getenv("ESTAT_APP_ID", "").strip()
    if not app_id:
        print("❌ Missing ESTAT_APP_ID in .env", file=sys.stderr)
        sys.exit(1)
    return app_id

def _parse_cat_string(s: str | None) -> Dict[str, str]:
    """
    Parse "cdCat01=A;cdCat02=B" into {"cdCat01":"A","cdCat02":"B"}.
    """
    out: Dict[str, str] = {}
    if not s:
        return out
    for kv in s.split(";"):
        kv = kv.strip()
        if not kv:
            continue
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out

def _request_estat(app_id: str, stats_data_id: str, cd_time: Optional[str],
                   cd_area: List[str], extra_params: Optional[Dict[str, str]] = None,
                   tries: int = 5) -> Dict[str, Any]:
    params = {
        "appId": app_id,
        "statsDataId": stats_data_id,
        "cdArea": ",".join(cd_area),
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
        p["limit"] = "100000"

        for attempt in range(tries):
            r = requests.get(ESTAT_ENDPOINT, params=p, timeout=60)
            if r.status_code == 200:
                break
            if r.status_code in (429, 503):
                time.sleep(0.7 * (2 ** attempt))
                continue
            r.raise_for_status()

        data = r.json()
        stat = data.get("GET_STATS_DATA", {}).get("STATISTICAL_DATA", {})
        value = stat.get("DATA_INF", {}).get("VALUE", [])
        if isinstance(value, dict):
            value = [value]
        if not value:
            break

        items.extend(value)
        if len(value) < int(p["limit"]):
            break
        start += int(p["limit"])
    return {"items": items}

def _flatten_values(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        out: Dict[str, Any] = {}
        for k, v in rec.items():
            if k == "$":
                out["value"] = v
            elif k.startswith("@"):
                out[k[1:]] = v  # strip '@'
            else:
                out[k] = v
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df

def _find_area_col(df: pd.DataFrame) -> str:
    if "area" in df.columns:
        return "area"
    if "area_code" in df.columns:
        return "area_code"
    # sometimes area appears as catXX with ward codes
    for c in df.columns:
        if c.startswith("cat") and df[c].astype(str).str.match(r"^13\d{3}$").any():
            return c
    raise RuntimeError("Could not determine area column from e-Stat response")

def _series_from_table(app_id: str, table_id: str, cd_time: Optional[str],
                       label: str, cat_kwargs: Dict[str, str]) -> pd.DataFrame:
    res = _request_estat(app_id, table_id, cd_time, list(TOKYO_WARDS.keys()), cat_kwargs)
    df = _flatten_values(res["items"])
    if df.empty:
        print(f"⚠️ No rows for {label} (statsDataId={table_id})")
        return pd.DataFrame(columns=["area_code","time",label])

    area_col = _find_area_col(df)
    time_col = "time" if "time" in df.columns else ("time_code" if "time_code" in df.columns else None)
    if time_col is None:
        time_col = "time"; df[time_col] = None

    out = df[[area_col, time_col, "value"]].rename(
        columns={area_col: "area_code", time_col: "time", "value": label}
    )
    out = out[out["area_code"].isin(TOKYO_WARDS.keys())].copy()
    # keep last (latest) by time if multiple
    out = out.sort_values(["area_code", "time"]).drop_duplicates(["area_code"], keep="last")
    return out

def _percent_from_pair(app_id: str, num_id: str, den_id: str, cd_time: Optional[str],
                       label: str, num_cats: Dict[str,str], den_cats: Dict[str,str]) -> pd.DataFrame:
    num = _series_from_table(app_id, num_id, cd_time, "num", num_cats)
    den = _series_from_table(app_id, den_id, cd_time, "den", den_cats)
    if num.empty or den.empty:
        print(f"⚠️ Could not compute {label}: numerator or denominator empty.")
        base = pd.DataFrame({"area_code": list(TOKYO_WARDS.keys())})
        return base.assign(**{label: pd.NA})

    df = num.merge(den, on="area_code", how="outer")
    df[label] = (df["num"] / df["den"] * 100).round(2)
    return df[["area_code", label]]

# ----------------------- main builder -----------------------
def build_tokyo_features(
    app_id: str,
    # Either a direct percent table...
    age65_id: Optional[str], alone65_id: Optional[str], poverty_id: Optional[str],
    # ...or a numerator/denominator pair
    age65_num_id: Optional[str], age65_den_id: Optional[str],
    alone65_num_id: Optional[str], alone65_den_id: Optional[str],
    poverty_num_id: Optional[str], poverty_den_id: Optional[str],
    # category filters for each (semi-colon "k=v;k2=v2")
    cats_age65: Dict[str,str], cats_alone65: Dict[str,str], cats_poverty: Dict[str,str],
    cats_age65_num: Dict[str,str], cats_age65_den: Dict[str,str],
    cats_alone65_num: Dict[str,str], cats_alone65_den: Dict[str,str],
    cats_poverty_num: Dict[str,str], cats_poverty_den: Dict[str,str],
    cd_time: Optional[str],
) -> pd.DataFrame:

    frames: list[pd.DataFrame] = []

    # ---------- pct_age65p ----------
    if age65_id:
        frames.append(_series_from_table(app_id, age65_id, cd_time, "pct_age65p", cats_age65))
    elif age65_num_id and age65_den_id:
        frames.append(_percent_from_pair(app_id, age65_num_id, age65_den_id, cd_time,
                                         "pct_age65p", cats_age65_num, cats_age65_den))
    else:
        print("⚠️ No source for pct_age65p provided.")

    # ---------- pct_single65p ----------
    if alone65_id:
        frames.append(_series_from_table(app_id, alone65_id, cd_time, "pct_single65p", cats_alone65))
    elif alone65_num_id and alone65_den_id:
        frames.append(_percent_from_pair(app_id, alone65_num_id, alone65_den_id, cd_time,
                                         "pct_single65p", cats_alone65_num, cats_alone65_den))
    else:
        print("⚠️ No source for pct_single65p provided.")

    # ---------- poverty_rate ----------
    if poverty_id:
        frames.append(_series_from_table(app_id, poverty_id, cd_time, "poverty_rate", cats_poverty))
    elif poverty_num_id and poverty_den_id:
        frames.append(_percent_from_pair(app_id, poverty_num_id, poverty_den_id, cd_time,
                                         "poverty_rate", cats_poverty_num, cats_poverty_den))
    else:
        print("⚠️ No source for poverty_rate provided.")

    if not frames:
        raise SystemExit("❌ Provide at least one indicator (age65 / alone65 / poverty).")

    feat = None
    for f in frames:
        feat = f if feat is None else feat.merge(f, on="area_code", how="outer")

    feat = feat.rename(columns={"area_code": "ward_jis"})
    feat["ward_name"] = feat["ward_jis"].map(TOKYO_WARDS)

    cols = ["ward_jis","ward_name","pct_age65p","pct_single65p","poverty_rate"]
    for c in cols:
        if c not in feat.columns:
            feat[c] = pd.NA
    feat = feat[cols].sort_values("ward_jis").reset_index(drop=True)
    return feat

# ----------------------- CLI -----------------------
def main():
    p = argparse.ArgumentParser(description="Ingest e-Stat for Tokyo 23 wards (percent tables or num/den pairs).")
    # direct percent tables
    p.add_argument("--age65-id", default="")
    p.add_argument("--alone65-id", default="")
    p.add_argument("--poverty-id", default="")
    # pairs
    p.add_argument("--age65-num-id", default="")
    p.add_argument("--age65-den-id", default="")
    p.add_argument("--alone65-num-id", default="")
    p.add_argument("--alone65-den-id", default="")
    p.add_argument("--poverty-num-id", default="")
    p.add_argument("--poverty-den-id", default="")
    # category filters
    p.add_argument("--cats-age65", default="")
    p.add_argument("--cats-alone65", default="")
    p.add_argument("--cats-poverty", default="")
    p.add_argument("--cats-age65-num", default="")
    p.add_argument("--cats-age65-den", default="")
    p.add_argument("--cats-alone65-num", default="")
    p.add_argument("--cats-alone65-den", default="")
    p.add_argument("--cats-poverty-num", default="")
    p.add_argument("--cats-poverty-den", default="")
    # common
    p.add_argument("--time", default="")
    p.add_argument("--raw-out", default="data/raw/jp_estat_tokyo_raw.csv")
    p.add_argument("--features-out", default="data/interim/jp_tokyo_features.parquet")
    args = p.parse_args()

    app_id = _load_app_id()
    Path(args.raw_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.features_out).parent.mkdir(parents=True, exist_ok=True)

    feat = build_tokyo_features(
        app_id=app_id,
        age65_id=args.age65_id or None,
        alone65_id=args.alone65_id or None,
        poverty_id=args.poverty_id or None,
        age65_num_id=args.age65_num_id or None,
        age65_den_id=args.age65_den_id or None,
        alone65_num_id=args.alone65_num_id or None,
        alone65_den_id=args.alone65_den_id or None,
        poverty_num_id=args.poverty_num_id or None,
        poverty_den_id=args.poverty_den_id or None,
        cats_age65=_parse_cat_string(args.cats_age65),
        cats_alone65=_parse_cat_string(args.cats_alone65),
        cats_poverty=_parse_cat_string(args.cats_poverty),
        cats_age65_num=_parse_cat_string(args.cats_age65_num),
        cats_age65_den=_parse_cat_string(args.cats_age65_den),
        cats_alone65_num=_parse_cat_string(args.cats_alone65_num),
        cats_alone65_den=_parse_cat_string(args.cats_alone65_den),
        cats_poverty_num=_parse_cat_string(args.cats_poverty_num),
        cats_poverty_den=_parse_cat_string(args.cats_poverty_den),
        cd_time=(args.time or None),
    )

    # Save — human-readable CSV + parquet for pipeline
    feat.to_csv(args.raw_out, index=False, encoding="utf-8-sig")
    feat.to_parquet(args.features_out, index=False)
    print(f"✅ Saved raw:      {args.raw_out}")
    print(f"✅ Saved features: {args.features_out}")
    print(feat.head())

if __name__ == "__main__":
    main()
