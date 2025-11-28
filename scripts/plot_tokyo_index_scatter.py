# scripts/plot_tokyo_index_scatter.py
"""
Scatterplot comparing Designed IRI vs PCA IRI for Tokyo wards.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np


def ensure_parent(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index-path", default="data/processed/jp_tokyo_with_designed_pca.csv")
    p.add_argument("--out-path", default="out/spatial_tokyo/tokyo_diri_vs_pca_scatter.png")
    args = p.parse_args()

    print("📥 Loading data…")
    df = pd.read_csv(args.index_path)

    x = df["iri_designed"]
    y = df["iri_pca"]

    r, pval = pearsonr(x, y)
    print(f"Correlation r = {r:.3f}, p = {pval:.4g}")

    # Fit regression line
    m, b = np.polyfit(x, y, 1)
    line = m * x + b

    print("🎨 Plotting…")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, color="steelblue")
    ax.plot(x, line, color="darkred", lw=2)

    ax.set_xlabel("Designed Isolation Index (D-IRI)")
    ax.set_ylabel("PCA-based Isolation Index")
    ax.set_title(f"D-IRI vs PCA Isolation Index\nr = {r:.3f}")

    ensure_parent(args.out_path)
    plt.savefig(args.out_path, dpi=300)
    plt.close()
    print(f"💾 Scatterplot saved to {args.out_path}")


if __name__ == "__main__":
    main()
