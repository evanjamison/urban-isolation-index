# -*- coding: utf-8 -*-
"""
01_ingest_jp_estat.py — e-Stat Japan ingestion (Tokyo 23 wards + Osaka wards)

Supports two patterns for each indicator:
  A) A single table that already returns a *percent* for each ward, or
  B) A (numerator, denominator) pair of tables from which we compute a percent.

Indicators we build:
  - pct_age65p      : % of population aged 65+       (you chose: pattern B)
  - pct_single65p   : % of single-person 65+ hh      (you chose: pattern A)
  - poverty_rate    : % of households under poverty  (you chose: pattern A)

Examples (PowerShell):

  # Tokyo: age65 = num/den; single65 & poverty = direct percent tables
  python -m src.cli.01_ingest_jp_estat `
    --city tokyo `
    --age65-num-id   0000000000 --age65-den-id 0000000000 `
    --alone65-id     0000000000 `
    --poverty-id     0000000000 `
    --time 202010

  # Osaka: example with num/den for single65 too
  python -m src.cli.01_ingest_jp_estat `
    --city osaka `
    --age65-num-id   <ID> --age65-den-id <ID> `
    --alone65-num-id <ID> --alone65-den-id <ID> `
    --poverty-id     <ID> `
    --time 202010
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import argparse
import requests
import pandas as pd
from dotenv import load_dotenv

# -------------------------------------------------------------
# JIS X-0402 municipality codes
# -------------------------------------------------------------
# Tokyo 23 wards
TOKYO_WARDS: Dict[str, str] = {
    "13101": "千代田区",
    "13102": "中央区",
    "13103": "港区",
    "13104": "新宿区",
    "13105": "文京区",
    "13106": "台東区",
    "13107": "墨田区",
    "13108": "江東区",
    "13109": "品川区",
    "13110": "目黒区",
    "13111": "大田区",
    "13112": "世田谷区",
    "13113": "渋谷区",
    "13114": "中野区",
    "13115": "杉並区",
    "13116": "豊島区",
    "13117": "北区",
    "13118": "荒川区",
    "13119": "板橋区",
    "13120": "練馬区",
    "13121": "足立区",
    "13122": "葛飾区",
    "13123": "江戸川区",
}

# Osaka-shi wards
OSAKA_WARDS: Dict[str, str] = {
    "27102": "都島区",
    "27103": "福島区",
    "27104": "此花区",
    "27106": "西区",
    "27107": "港区",
    "27108": "大正区",
    "27109": "天王寺区",
    "27111": "浪速区",
    "27113": "西淀川区",
    "27114": "東淀川区",
    "27115": "東成区",
    "27116": "生野区",
    "27117": "旭区",
    "27118": "城東区",
    "27119": "阿倍野区",
    "27120": "住吉区",
    "27121": "東住吉区",
    "27122": "西成区",
    "27123": "淀川区",
    "27124": "鶴見区",
    "27125": "住之江区",
    "27126": "平野区",
    "27127": "北区",
    "27128": "中央区",
}

CITY_CONFIG = {
    "tokyo": {
        "wards": TOKYO_WARDS,
        "raw_default": "data/raw/jp_estat_tokyo_raw.csv",
        "features_default": "data/interim/jp_tokyo_features.parquet",
    },
    "osaka": {
        "wards": OSAKA_WARDS,
        "raw_default": "data/raw/jp_estat_osaka_raw.csv",
        "features_default": "data/interim/jp_osaka_features.parquet",
    },
}

ESTAT_ENDPOINT = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def _load_app_id() -> str:
    load_dotenv()
    app_id = os.getenv("ESTAT_APP_ID", "").strip()
    if not app_id:
        print("❌ Missing ESTAT_APP_ID in .env", file=sys.stderr)
        sys.exit(1)
    return app_id


def _parse_cat_string(s: str | None) -> Dict[str, str]:
    """
    Parse "cdCat01=A;cdCat02=B" into {"cdCat01": "A", "cdCat02": "B"}.
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


def _request_estat(
    app_id: str,
    stats_data_id: str,
    cd_time: Optional[str],
    cd_area: List[str],
    extra_params: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Call e-Stat getStatsData (JSON endpoint) with basic paging and backoff.
    Returns a dict {"items": [...]} where items is a list of VALUE records.
    """
    params = {
        "appId": app_id,
        "statsDataId": stats_data_id,
        "cdArea": ",".join(cd_area),
        "lang": "J",  # JSON still fine in Japanese; labels aren’t crucial here
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

        # Retry loop with simple backoff
        for attempt in range(5):
            r = requests.get(ESTAT_ENDPOINT, params=p, timeout=60)
            if r.status_code == 200:
                break
            if r.status_code in (429, 503):
                delay = 0.7 * (2**attempt)
                print(f"⏱ HTTP {r.status_code}, retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            print(f"❌ HTTP {r.status_code} from e-Stat", file=sys.stderr)
            print(r.text[:500])
            r.raise_for_status()

        try:
            data = r.json()
        except ValueError:
            print("❌ e-Stat did not return JSON. First 500 chars:", file=sys.stderr)
            print(r.text[:500])
            raise

        stat = data.get("GET_STATS_DATA", {}).get("STATISTICAL_DATA", {})
        value = stat.get("DATA_INF", {}).get("VALUE", [])
        if isinstance(value, dict):
            value = [value]
        if not value:
            break

        items.extend(value)
                # DEBUG: print one example record for the first page & first statsDataId
        if start == 1 and items:
            import json
            print("\n===== DEBUG FIRST VALUE RECORD =====")
            print(json.dumps(items[0], ensure_ascii=False)[:600])
            print("===== END DEBUG =====\n")


        if len(value) < int(p["limit"]):
            break
        start += int(p["limit"])

    return {"items": items}


def _flatten_values(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten e-Stat VALUE records into a DataFrame.

    Different tables put the numeric in different places:
      - "$"
      - "#text"
      - sometimes even a plain key like "VALUE" or similar.

    This function tries, in order:
      1) "$"
      2) "#text"
      3) the first non-@ key whose value looks numeric
    """
    rows = []
    for rec in records:
        out: Dict[str, Any] = {}
        value_set = False

        for k, v in rec.items():
            # Attribute-style metadata: "@area", "@tab", etc.
            if k.startswith("@"):
                out[k[1:]] = v
                continue

            # Preferred explicit numeric keys
            if k in ("$", "#text", "value") and not value_set:
                out["value"] = v
                value_set = True

            # Keep original field as well (just in case we want to inspect later)
            out[k] = v

        # Fallback: if we still don't have "value", look for any
        # field that looks numeric (contains at least one digit).
        if not value_set:
            for k, v in rec.items():
                if k.startswith("@"):
                    continue
                if isinstance(v, (int, float)) and not value_set:
                    out["value"] = v
                    value_set = True
                    break
                if isinstance(v, str) and any(ch.isdigit() for ch in v) and not value_set:
                    out["value"] = v
                    value_set = True
                    break

        rows.append(out)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df




def _find_area_col(df: pd.DataFrame, valid_codes: set[str]) -> str:
    """
    Find the column that contains JIS area codes for the wards we requested.
    We look for a column where at least some values match valid_codes.
    """
    # Prefer common names
    for candidate in ("area", "area_code", "tab_code", "cat01"):
        if candidate in df.columns:
            col = df[candidate].astype(str)
            if col.isin(valid_codes).any():
                return candidate

    # Fallback: scan everything
    for c in df.columns:
        col = df[c].astype(str)
        if col.isin(valid_codes).any():
            return c

    raise RuntimeError("Could not determine area column from e-Stat response")


def _series_from_table(
    app_id: str,
    table_id: str,
    cd_time: Optional[str],
    area_codes: List[str],
    label: str,
    cat_kwargs: Dict[str, str],
) -> pd.DataFrame:
    print(f"📥 Requesting {label} from e-Stat table {table_id} ...")
    res = _request_estat(app_id, table_id, cd_time, area_codes, cat_kwargs)
    df = _flatten_values(res["items"])
    if df.empty:
        print(f"⚠ No rows for {label} (statsDataId={table_id})")
        return pd.DataFrame(columns=["area_code", "time", label])

    area_col = _find_area_col(df, set(area_codes))
    time_col = "time" if "time" in df.columns else (
        "time_code" if "time_code" in df.columns else None
    )
    if time_col is None:
        time_col = "time"
        df[time_col] = None

    out = df[[area_col, time_col, "value"]].rename(
        columns={area_col: "area_code", time_col: "time", "value": label}
    )
    out = out[out["area_code"].isin(area_codes)].copy()
    out = out.sort_values(["area_code", "time"]).drop_duplicates(
        ["area_code"], keep="last"
    )

    if not out.empty:
        print(
            f"   → {label}: loaded {len(out)} wards | "
            f"min={out[label].min()}, max={out[label].max()}"
        )
    return out


def _percent_from_pair(
    app_id: str,
    num_id: str,
    den_id: str,
    cd_time: Optional[str],
    area_codes: List[str],
    label: str,
    num_cats: Dict[str, str],
    den_cats: Dict[str, str],
) -> pd.DataFrame:
    print(f"🧮 Computing {label} from num={num_id}, den={den_id} ...")

    num = _series_from_table(app_id, num_id, cd_time, area_codes, "num", num_cats)
    den = _series_from_table(app_id, den_id, cd_time, area_codes, "den", den_cats)

    if num.empty or den.empty:
        print(f"⚠ Could not compute {label}: numerator or denominator empty.")
        base = pd.DataFrame({"area_code": area_codes})
        return base.assign(**{label: pd.NA})

    df = num.merge(den, on="area_code", how="outer", suffixes=("_num", "_den"))
    df[label] = (df["num"] / df["den"] * 100).round(4)

    if not df.empty:
        print(
            f"   → {label} (%): loaded {len(df)} wards | "
            f"min={df[label].min()}, max={df[label].max()}"
        )

    return df[["area_code", label]]


# -------------------------------------------------------------
# Main feature builder
# -------------------------------------------------------------
def build_city_features(
    city: str,
    wards: Dict[str, str],
    app_id: str,
    # A: direct percent tables
    age65_id: Optional[str],
    alone65_id: Optional[str],
    poverty_id: Optional[str],
    # B: numerator / denominator pairs
    age65_num_id: Optional[str],
    age65_den_id: Optional[str],
    alone65_num_id: Optional[str],
    alone65_den_id: Optional[str],
    poverty_num_id: Optional[str],
    poverty_den_id: Optional[str],
    # category filters
    cats_age65: Dict[str, str],
    cats_alone65: Dict[str, str],
    cats_poverty: Dict[str, str],
    cats_age65_num: Dict[str, str],
    cats_age65_den: Dict[str, str],
    cats_alone65_num: Dict[str, str],
    cats_alone65_den: Dict[str, str],
    cats_poverty_num: Dict[str, str],
    cats_poverty_den: Dict[str, str],
    cd_time: Optional[str],
) -> pd.DataFrame:
    area_codes = list(wards.keys())
    frames: list[pd.DataFrame] = []

    # ---------- pct_age65p (you chose: B, compute from num/den) ----------
    if age65_num_id and age65_den_id:
        frames.append(
            _percent_from_pair(
                app_id,
                age65_num_id,
                age65_den_id,
                cd_time,
                area_codes,
                "pct_age65p",
                cats_age65_num,
                cats_age65_den,
            )
        )
    elif age65_id:
        frames.append(
            _series_from_table(app_id, age65_id, cd_time, area_codes, "pct_age65p", cats_age65)
        )
    else:
        print("⚠ No source for pct_age65p provided.")

    # ---------- pct_single65p (you chose: A, direct percent) ----------
    if alone65_id:
        frames.append(
            _series_from_table(
                app_id, alone65_id, cd_time, area_codes, "pct_single65p", cats_alone65
            )
        )
    elif alone65_num_id and alone65_den_id:
        frames.append(
            _percent_from_pair(
                app_id,
                alone65_num_id,
                alone65_den_id,
                cd_time,
                area_codes,
                "pct_single65p",
                cats_alone65_num,
                cats_alone65_den,
            )
        )
    else:
        print("⚠ No source for pct_single65p provided.")

    # ---------- poverty_rate (you chose: A, direct percent) ----------
    if poverty_id:
        frames.append(
            _series_from_table(
                app_id, poverty_id, cd_time, area_codes, "poverty_rate", cats_poverty
            )
        )
    elif poverty_num_id and poverty_den_id:
        frames.append(
            _percent_from_pair(
                app_id,
                poverty_num_id,
                poverty_den_id,
                cd_time,
                area_codes,
                "poverty_rate",
                cats_poverty_num,
                cats_poverty_den,
            )
        )
    else:
        print("⚠ No source for poverty_rate provided.")

    if not frames:
        raise SystemExit("❌ Provide at least one indicator (age65 / single65 / poverty).")

    feat: Optional[pd.DataFrame] = None
    for f in frames:
        feat = f if feat is None else feat.merge(f, on="area_code", how="outer")

    feat = feat.rename(columns={"area_code": "ward_jis"})
    feat["ward_name"] = feat["ward_jis"].map(wards)

    cols = ["ward_jis", "ward_name", "pct_age65p", "pct_single65p", "poverty_rate"]
    for c in cols:
        if c not in feat.columns:
            feat[c] = pd.NA

    feat = feat[cols].sort_values("ward_jis").reset_index(drop=True)
    feat["city"] = city

    return feat


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Ingest e-Stat for Japanese wards (Tokyo 23 wards or Osaka wards), "
            "using percent tables or numerator/denominator pairs."
        )
    )

    p.add_argument(
        "--city",
        choices=["tokyo", "osaka"],
        default="tokyo",
        help="Target city to ingest (tokyo or osaka). Default: tokyo.",
    )

    # direct percent tables (pattern A)
    p.add_argument("--age65-id", default="")
    p.add_argument("--alone65-id", default="")
    p.add_argument("--poverty-id", default="")

    # numerator / denominator pairs (pattern B)
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
    p.add_argument(
        "--raw-out",
        default="",
        help="Raw CSV output. If empty, a city-specific default is used.",
    )
    p.add_argument(
        "--features-out",
        default="",
        help="Features parquet output. If empty, a city-specific default is used.",
    )

    args = p.parse_args()

    cfg = CITY_CONFIG[args.city]
    wards = cfg["wards"]

    raw_out_path = Path(args.raw_out or cfg["raw_default"])
    features_out_path = Path(args.features_out or cfg["features_default"])
    raw_out_path.parent.mkdir(parents=True, exist_ok=True)
    features_out_path.parent.mkdir(parents=True, exist_ok=True)

    app_id = _load_app_id()

    feat = build_city_features(
        city=args.city,
        wards=wards,
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

    # Save: human-readable CSV + parquet for pipeline
    feat.to_csv(raw_out_path, index=False, encoding="utf-8-sig")
    feat.to_parquet(features_out_path, index=False)

    print(f"✅ City:           {args.city}")
    print(f"✅ Saved raw:      {raw_out_path}")
    print(f"✅ Saved features: {features_out_path}")
    print(feat.head())


if __name__ == "__main__":
    main()
