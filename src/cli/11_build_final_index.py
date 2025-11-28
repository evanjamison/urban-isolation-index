"""
11_build_final_index.py

Build FINAL Isolation Index (iso_final) for a city (Tokyo or Osaka),
combining:

  - Demographic isolation (pct_age65p_z, pct_single65p_z)
  - Socioeconomic risk (poverty_rate_z)
  - Transit access (transit_z, with higher transit => lower isolation)

This version does **NOT** require or use any access_* columns.

Usage examples:

  # Tokyo
  .\\.venv\\Scripts\\python.exe -m src.cli.11_build_final_index ^
      --input data/processed/jp_tokyo_index_full.csv ^
      --out   data/processed/jp_tokyo_iso_final.csv

  # Osaka
  .\\.venv\\Scripts\\python.exe -m src.cli.11_build_final_index ^
      --input data/processed/jp_osaka_index_full.csv ^
      --out   data/processed/jp_osaka_iso_final.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build final Isolation Index (iso_final) from merged index+transit."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Merged dataset with demographics + poverty + transit_z.",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output CSV path for final index.",
    )
    return p.parse_args(argv)


def safe_z(x: pd.Series) -> pd.Series:
    """Compute z-score safely (handles std = 0)."""
    std = x.std(ddof=0)
    if std == 0 or np.isnan(std):
        return x * 0
    return (x - x.mean()) / std


def main(argv=None):
    args = parse_args(argv)
    input_path = Path(args.input)
    out_path = Path(args.out)

    print(f"📂 Loading merged dataset from {input_path} ...")
    df = pd.read_csv(input_path)

    required = [
        "pct_age65p_z",
        "pct_single65p_z",
        "poverty_rate_z",
        "transit_z",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for iso_final: {missing}")

    # ----------------------------------------------------------------------
    # 1) Define components (NO access components)
    # ----------------------------------------------------------------------
    print("🧮 Computing components (demo / socio / transit) ...")

    # Demographic isolation (equal weight between age65 and single65)
    df["demo_component"] = (
        0.5 * df["pct_age65p_z"] +
        0.5 * df["pct_single65p_z"]
    )

    # Socioeconomic risk
    df["socio_component"] = df["poverty_rate_z"]

    # Transit access (more transit = lower isolation → negative sign)
    df["transit_component"] = -df["transit_z"]

    # ----------------------------------------------------------------------
    # 2) Normalize each component to mean=0, std=1
    # ----------------------------------------------------------------------
    for col in ["demo_component", "socio_component", "transit_component"]:
        zcol = col + "_z"
        df[zcol] = safe_z(df[col])

    # ----------------------------------------------------------------------
    # 3) Weighted final index (no access term)
    #    Original weights were: demo 0.40, socio 0.30, transit 0.10 (access 0.20)
    #    After dropping access, we renormalize remaining 0.40:0.30:0.10
    #    → demo 0.50, socio 0.375, transit 0.125 (sum = 1.0)
    # ----------------------------------------------------------------------
    w_demo = 0.50
    w_socio = 0.375
    w_transit = 0.125

    print("📊 Building final weighted Isolation Index (iso_final) ...")
    df["iso_final"] = (
        w_demo * df["demo_component_z"] +
        w_socio * df["socio_component_z"] +
        w_transit * df["transit_component_z"]
    )

    print("\nSummary of iso_final:")
    print(df["iso_final"].describe())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n💾 Saved FINAL Isolation Index to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
