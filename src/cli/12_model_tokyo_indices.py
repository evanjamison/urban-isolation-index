# src/cli/12_model_tokyo_indices.py
"""
12_model_tokyo_indices.py

Run regression models and build a PCA-based isolation index for Tokyo wards.

Usage (from project root, inside venv):

  .\.venv\Scripts\python.exe -m src.cli.12_model_tokyo_indices ^
      --in-path data/processed/jp_tokyo_with_designed.csv ^
      --out-path data/processed/jp_tokyo_with_designed_pca.csv ^
      --summary-out out/modeling_tokyo/modeling_summary.txt
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import statsmodels.api as sm


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def run_ols(y, X, add_const: bool = True):
    if add_const:
        X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-path", default="data/processed/jp_tokyo_with_designed.csv")
    p.add_argument("--out-path", default="data/processed/jp_tokyo_with_designed_pca.csv")
    p.add_argument(
        "--summary-out",
        default="out/modeling_tokyo/modeling_summary.txt",
    )
    args = p.parse_args()

    print(f"📥 Loading dataset from {args.in_path} ...")
    df = pd.read_csv(args.in_path)

    required_cols = [
        "ward_jis",
        "ward_name",
        "iri_designed",
        "pct_age65p_z",
        "pct_single65p_z",
        "poverty_rate_z",
        "access_z",
        "transit_z",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("\n✅ Dataset columns available:")
    print(df.columns.tolist())

    # ------------------------------------------------------------------
    # 1) OLS regression on designed index
    # ------------------------------------------------------------------
    predictors = [
        "pct_age65p_z",
        "pct_single65p_z",
        "poverty_rate_z",
        "access_z",
        "transit_z",
    ]
    X = df[predictors]
    y = df["iri_designed"]

    print("\n🧠 Fitting OLS: iri_designed ~ demographics + access + transit ...")
    ols_model = run_ols(y, X)
    print(ols_model.summary())

    # Optional: OLS for iso_index if present
    ols_iso_model = None
    if "iso_index" in df.columns:
        print("\n🧠 Fitting OLS: iso_index ~ demographics + access + transit ...")
        y_iso = df["iso_index"]
        ols_iso_model = run_ols(y_iso, X)
        print(ols_iso_model.summary())

    # ------------------------------------------------------------------
    # 2) PCA-based isolation index
    # ------------------------------------------------------------------
    print("\n🧮 Running PCA to build data-driven isolation index ...")

    # Copy and flip access / transit so higher => worse (more isolated)
    X_pca = df[predictors].copy()
    X_pca["access_z"] = -X_pca["access_z"]
    X_pca["transit_z"] = -X_pca["transit_z"]

    pca = PCA(n_components=1)
    pc1_scores = pca.fit_transform(X_pca.values).flatten()

    # Rescale PC1 to mean 0, std 1 for interpretability
    pc1_z = (pc1_scores - pc1_scores.mean()) / pc1_scores.std(ddof=0)
    df["iri_pca"] = pc1_z

    print("Explained variance ratio (PC1):", float(pca.explained_variance_ratio_[0]))
    print("\n🔗 Correlation between iri_designed and iri_pca:")
    corr = np.corrcoef(df["iri_designed"], df["iri_pca"])[0, 1]
    print(f"  corr(iri_designed, iri_pca) = {corr:.3f}")

    # ------------------------------------------------------------------
    # 3) Save outputs
    # ------------------------------------------------------------------
    ensure_parent_dir(args.out_path)
    df.to_csv(args.out_path, index=False)
    print(f"\n💾 Saved dataset with PCA index to {args.out_path}")

    ensure_parent_dir(args.summary_out)
    with open(args.summary_out, "w", encoding="utf-8") as f:
        f.write("Tokyo Isolation Index — Modeling Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write("Input file: " + args.in_path + "\n")
        f.write("Output file (with iri_pca): " + args.out_path + "\n\n")

        f.write("OLS: iri_designed ~ pct_age65p_z + pct_single65p_z + "
                "poverty_rate_z + access_z + transit_z\n")
        f.write("-" * 70 + "\n")
        f.write(str(ols_model.summary()))
        f.write("\n\n")

        if ols_iso_model is not None:
            f.write("OLS: iso_index ~ same predictors\n")
            f.write("-" * 70 + "\n")
            f.write(str(ols_iso_model.summary()))
            f.write("\n\n")

        f.write("PCA-based index:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Explained variance ratio (PC1): "
                f"{float(pca.explained_variance_ratio_[0]):.4f}\n")
        f.write(f"corr(iri_designed, iri_pca): {corr:.4f}\n")

    print(f"📝 Modeling summary written to {args.summary_out}")
    print("\n✅ 12_model_tokyo_indices completed.")


if __name__ == "__main__":
    main()
