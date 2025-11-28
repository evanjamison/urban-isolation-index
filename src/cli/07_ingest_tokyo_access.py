"""
07_ingest_tokyo_access.py — clean & standardize an 'access to resources' proxy
for Tokyo's 23 wards.

Expected input CSV (default: data/raw/tokyo_access_proxy.csv):

    ward_jis,ward_name,access_raw
    13101,千代田区,12
    13102,中央区,15
    ...

Output CSV (default: data/interim/jp_tokyo_access_proxy.csv):

    ward_jis,ward_name,access_raw,access_z

Where access_z is a z-score (higher = better access).
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest & standardize a Tokyo access proxy by ward."
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default="data/raw/tokyo_access_proxy.csv",
        help="Raw CSV with columns: ward_jis, ward_name, access_raw.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="data/interim/jp_tokyo_access_proxy.csv",
        help="Cleaned CSV with added access_z column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_path = Path(args.raw_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📥 Loading raw access data from {raw_path} ...")
    df = pd.read_csv(raw_path)

    # Basic cleaning / type handling
    if "ward_jis" not in df.columns or "access_raw" not in df.columns:
        raise ValueError(
            "Input must contain at least 'ward_jis' and 'access_raw' columns."
        )

    # Ensure ward_jis is a 5-digit string (like your other tables)
    df["ward_jis"] = df["ward_jis"].astype(str).str.strip()

    # Coerce access_raw to numeric
    df["access_raw"] = pd.to_numeric(df["access_raw"], errors="coerce")

    # Drop rows with missing access_raw (and warn)
    before = len(df)
    df = df.dropna(subset=["access_raw"])
    after = len(df)
    if after < before:
        print(f"⚠️ Dropped {before - after} rows with missing access_raw values.")

    # Compute z-score: (x - mean) / std
    mean = df["access_raw"].mean()
    std = df["access_raw"].std()

    if std == 0 or pd.isna(std):
        raise ValueError(
            "Standard deviation of access_raw is zero or NaN. "
            "Check that you have varying values across wards."
        )

    df["access_z"] = (df["access_raw"] - mean) / std

    # Sort by ward_jis for sanity
    df = df.sort_values("ward_jis").reset_index(drop=True)

    print("\nSummary of access_z:")
    print(df["access_z"].describe())

    print(f"\n✅ Saving cleaned access data to {out_path}")
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
