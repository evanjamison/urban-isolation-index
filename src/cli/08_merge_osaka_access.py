"""
08_merge_osaka_access.py

Merge Osaka isolation index with Osaka social access proxy,
and create iso_index_with_access (same logic as Tokyo).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def merge_access(index_path: str, access_path: str, out_path: str) -> None:
    index_path = Path(index_path)
    access_path = Path(access_path)
    out_path = Path(out_path)

    print(f"📂 Loading Osaka index from {index_path} ...")
    idx = pd.read_csv(index_path, dtype={"ward_jis": "int64"})

    print(f"📂 Loading Osaka access proxy from {access_path} ...")
    acc = pd.read_csv(access_path, dtype={"ward_jis": "int64"})

    # Required access columns
    required = ["ward_jis", "access_raw", "access_z"]
    missing = [c for c in required if c not in acc.columns]
    if missing:
        raise ValueError(f"Missing columns in access file: {missing}")

    # Keep only needed columns
    acc_small = acc[["ward_jis", "access_raw", "access_z"]].copy()

    print("🔗 Merging Osaka index + access on ward_jis ...")
    merged = idx.merge(acc_small, on="ward_jis", how="left", validate="one_to_one")

    # Exactly the same logic as Tokyo:
    # Higher access_z = better access → reduces isolation.
    merged["iso_index_with_access"] = merged["iso_index"] - 0.3 * merged["access_z"]

    print("\n🧮 Summary of iso_index_with_access:")
    print(merged["iso_index_with_access"].describe())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ Osaka index with access saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Osaka access proxy into Osaka index")
    parser.add_argument(
        "--index-path",
        default="data/processed/jp_osaka_index.csv",
        help="Path to existing Osaka index CSV",
    )
    parser.add_argument(
        "--access-path",
        default="data/interim/jp_osaka_access_proxy.csv",
        help="Path to cleaned Osaka access proxy CSV",
    )
    parser.add_argument(
        "--out-path",
        default="data/processed/jp_osaka_index_with_access.csv",
        help="Output CSV with merged data and new index",
    )

    args = parser.parse_args()
    merge_access(args.index_path, args.access_path, args.out_path)


if __name__ == "__main__":
    main()
