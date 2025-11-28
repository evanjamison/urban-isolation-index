"""
08_merge_access.py

Merge the Tokyo access proxy data into the existing Tokyo isolation index
and create a new version of the index that explicitly includes access_z.

Usage (from project root):

  .\.venv\Scripts\python.exe -m src.cli.08_merge_access
"""

import argparse
from pathlib import Path

import pandas as pd


def merge_access(index_path: str, access_path: str, out_path: str) -> None:
    index_path = Path(index_path)
    access_path = Path(access_path)
    out_path = Path(out_path)

    print(f"📂 Loading index from {index_path} ...")
    idx = pd.read_csv(index_path, dtype={"ward_jis": "int64"})

    print(f"📂 Loading access data from {access_path} ...")
    acc = pd.read_csv(access_path, dtype={"ward_jis": "int64"})

    # keep only what we need from access file
    acc_small = acc[["ward_jis", "access_raw", "access_z"]].copy()

    print("🔗 Merging on ward_jis ...")
    merged = idx.merge(acc_small, on="ward_jis", how="left", validate="one_to_one")

    # --- New index version that includes access_z explicitly ---
    # Old iso_index (z-score) already combines age / alone / poverty.
    # Here we *adjust* it so that better access (higher access_z)
    # reduces isolation risk.
    #
    # You can tune this weight later; starting with 0.3 is reasonable.
    merged["iso_index_with_access"] = merged["iso_index"] - 0.3 * merged["access_z"]

    print("\n🧮 Summary of iso_index_with_access:")
    print(merged["iso_index_with_access"].describe())

    # Save result
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Saved merged index with access to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Tokyo access proxy into index")
    parser.add_argument(
        "--index-path",
        default="data/processed/jp_tokyo_index.csv",
        help="Path to existing Tokyo index CSV",
    )
    parser.add_argument(
        "--access-path",
        default="data/interim/jp_tokyo_access_proxy.csv",
        help="Path to cleaned Tokyo access proxy CSV",
    )
    parser.add_argument(
        "--out-path",
        default="data/processed/jp_tokyo_index_with_access.csv",
        help="Output CSV with merged data and new index",
    )

    args = parser.parse_args()
    merge_access(args.index_path, args.access_path, args.out_path)


if __name__ == "__main__":
    main()
