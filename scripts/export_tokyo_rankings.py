# scripts/export_tokyo_rankings.py
"""
Export CSV tables of Tokyo ward rankings for D-IRI and PCA-IRI.
"""

import argparse
import os
import pandas as pd


def ensure_parent(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index-path", default="data/processed/jp_tokyo_with_designed_pca.csv")
    p.add_argument("--out-dir", default="out/tables_tokyo")
    args = p.parse_args()

    ensure_parent(args.out_dir)

    print("📥 Loading dataset…")
    df = pd.read_csv(args.index_path)

    # Rankings
    diri_rank = df.sort_values("iri_designed", ascending=False)
    pca_rank = df.sort_values("iri_pca", ascending=False)

    # Save full tables
    diri_rank.to_csv(os.path.join(args.out_dir, "tokyo_diri_full_ranking.csv"), index=False)
    pca_rank.to_csv(os.path.join(args.out_dir, "tokyo_pca_full_ranking.csv"), index=False)

    # Save top & bottom 10
    diri_rank.head(10).to_csv(os.path.join(args.out_dir, "tokyo_diri_top10.csv"), index=False)
    diri_rank.tail(10).to_csv(os.path.join(args.out_dir, "tokyo_diri_bottom10.csv"), index=False)

    pca_rank.head(10).to_csv(os.path.join(args.out_dir, "tokyo_pca_top10.csv"), index=False)
    pca_rank.tail(10).to_csv(os.path.join(args.out_dir, "tokyo_pca_bottom10.csv"), index=False)

    print("✅ Ranking tables saved.")


if __name__ == "__main__":
    main()
