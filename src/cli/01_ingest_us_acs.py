# src/cli/01_ingest_us_acs.py
from __future__ import annotations
import os
import time
from pathlib import Path
import requests
import pandas as pd

# -----------------------------
# ACS 5-year (detail tables)
# -----------------------------
BASE = "https://api.census.gov/data/2020/acs/acs5"
OUT_CSV = Path("data/raw/us_acs_ny_2020.csv")

# All variables below are from acs/acs5 (NOT the DP profile tables)
# We will compute %65+ ourselves from B01001 sex-by-age detail
VARS = {
    # totals
    "pop_total": "B01001_001E",
    "hh_total": "B11001_001E",
    # male 65+
    "m65_66": "B01001_020E",
    "m67_69": "B01001_021E",
    "m70_74": "B01001_022E",
    "m75_79": "B01001_023E",
    "m80_84": "B01001_024E",
    "m85p":   "B01001_025E",
    # female 65+
    "f65_66": "B01001_044E",
    "f67_69": "B01001_045E",
    "f70_74": "B01001_046E",
    "f75_79": "B01001_047E",
    "f80_84": "B01001_048E",
    "f85p":   "B01001_049E",
    # elderly living alone (numerator)
    "alone65_num": "B11007_007E",
    # poverty numerator/denominator
    "pov_num":   "B17001_002E",
    "pov_denom": "B17001_001E",
}

# NYC counties (FIPS) = Manhattan(061), Brooklyn(047), Queens(081), Bronx(005), Staten Island(085)
NYC_COUNTIES = {"061", "047", "081", "005", "085"}


def fetch_with_retry(params: dict, tries: int = 4, backoff: float = 0.7) -> list[list[str]]:
    """Robust GET with simple backoff for 429/503."""
    for i in range(tries):
        r = requests.get(BASE, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 503):
            time.sleep(backoff * (2 ** i))
            continue
        r.raise_for_status()
    raise RuntimeError(f"Failed after {tries} tries")


def pull_nyc_tracts(api_key: str | None = None) -> pd.DataFrame:
    # Build GET param list and include geography selectors for TRACTS
    get_fields = ",".join(list(VARS.values()) + ["NAME"])
    params = {
        "get": get_fields,
        "for": "tract:*",
        # IMPORTANT: when requesting tracts you must also include county
        "in": "state:36 county:*",
    }
    if api_key:
        params["key"] = api_key

    js = fetch_with_retry(params)
    df = pd.DataFrame(js[1:], columns=js[0])

    # Keep only NYC's 5 counties
    df = df[df["county"].isin(NYC_COUNTIES)].copy()

    # Rename ACS codes -> friendly names
    rename_map = {v: k for k, v in VARS.items()}
    df = df.rename(columns=rename_map)

    # Convert numeric columns (use the friendly names AFTER renaming)
    numeric_cols = list(rename_map.values())
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived metrics
    df["age65p_num"] = (
        df["m65_66"] + df["m67_69"] + df["m70_74"] + df["m75_79"] + df["m80_84"] + df["m85p"]
        + df["f65_66"] + df["f67_69"] + df["f70_74"] + df["f75_79"] + df["f80_84"] + df["f85p"]
    )
    df["pct_age65p"]   = (df["age65p_num"] / df["pop_total"] * 100).round(2)
    df["pct_alone65p"] = (df["alone65_num"] / df["hh_total"] * 100).round(2)
    df["poverty_rate"] = (df["pov_num"] / df["pov_denom"] * 100).round(2)

    # GEOID for joins
    df["GEOID"] = df["state"] + df["county"] + df["tract"]

    keep = [
        "GEOID", "NAME", "state", "county", "tract",
        "pct_age65p", "pct_alone65p", "poverty_rate", "pop_total"
    ]
    return df[keep]


def main(out_csv: str = str(OUT_CSV)):
    key = os.getenv("US_CENSUS_KEY")  # optional but recommended
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df = pull_nyc_tracts(api_key=key)
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved: {out_csv}  rows={len(df)}")


if __name__ == "__main__":
    main()
