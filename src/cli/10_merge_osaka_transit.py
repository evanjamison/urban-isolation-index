from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Osaka base, social access, and transit into jp_osaka_features.parquet."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "jp_osaka_base_clean.csv",
        help="Clean Osaka base demographics.",
    )
    parser.add_argument(
        "--access",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "jp_osaka_access_proxy.csv",
        help="Osaka social access proxy.",
    )
    parser.add_argument(
        "--transit",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "jp_osaka_transit.csv",
        help="Osaka transit access metrics.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "jp_osaka_features.parquet",
        help="Output parquet with Osaka features.",
    )

    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.base)
    access = pd.read_csv(args.access)
    transit = pd.read_csv(args.transit)

    key = "ward_name_ja"
    for df_name, df_obj in [("base", base), ("access", access), ("transit", transit)]:
        if key not in df_obj.columns:
            raise ValueError(f"{df_name} data missing '{key}' column.")

    df = (
        base.merge(access, on=key, how="left")
            .merge(transit, on=key, how="left")
    )

    # Make sure core D-IRI features exist for 03_build_index
    required_features = ["pct_age65p", "pct_single65p", "poverty_rate", "access_social", "access_transit"]
    missing = [c for c in required_features if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required features for Osaka index: {missing}\n"
            "Check 01/07/09 Osaka ingest scripts."
        )

    df.to_parquet(args.out, index=False)
    print(f"[ok] Wrote Osaka features parquet → {args.out}")


if __name__ == "__main__":
    main()
