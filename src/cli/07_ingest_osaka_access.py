from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# ---- Project paths ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Osaka ward-level social access and compute access_z."
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "osaka_access_proxy.csv",
        help="Raw Osaka access file shaped like Tokyo: ward_jis, ward_name, access_raw",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "jp_osaka_access_proxy.csv",
        help="Output CSV with ward_jis, ward_name, access_raw, access_z.",
    )
    args = parser.parse_args()

    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading raw Osaka access from {args.raw_path} ...")
    df = pd.read_csv(args.raw_path)

    # ---- Standardise columns -----------------------------------------------
    # Expect something like Tokyo: ward_jis, ward_name, access_raw
    if "ward_jis" not in df.columns:
        raise ValueError("Expected 'ward_jis' column in Osaka access file.")

    if "ward_name_ja" not in df.columns:
        if "ward_name" in df.columns:
            df = df.rename(columns={"ward_name": "ward_name_ja"})
        else:
            raise ValueError(
                "Expected either 'ward_name_ja' or 'ward_name' column "
                f"in {args.raw_path}"
            )

    if "access_raw" not in df.columns:
        raise ValueError("Expected 'access_raw' column in Osaka access file.")

    # Make sure ward_jis is numeric / consistent with other files
    df["ward_jis"] = df["ward_jis"].astype("int64")

    # ---- Compute z-score (access_z) ----------------------------------------
    x = df["access_raw"].astype(float)
    mean = x.mean()
    std = x.std(ddof=0)

    if std == 0 or pd.isna(std):
        print("⚠️ access_raw has zero (or NaN) variance; setting access_z = 0.")
        df["access_z"] = 0.0
    else:
        df["access_z"] = (x - mean) / std

    print("\n🧮 Summary of access_raw:")
    print(x.describe())
    print("\n🧮 Summary of access_z:")
    print(df["access_z"].describe())

    # Keep a clean, predictable column order
    out_cols = ["ward_jis", "ward_name_ja", "access_raw", "access_z"]
    df[out_cols].to_csv(args.out_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ Wrote Osaka access proxy with z-scores → {args.out_path}")


if __name__ == "__main__":
    main()
