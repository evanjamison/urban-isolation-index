# -*- coding: utf-8 -*-
"""
06_modeling_suite.py — OLS, Ridge regression, and PCA for Tokyo isolation index.

Usage (from project root):

  .\.venv\Scripts\python.exe -m src.cli.06_modeling_suite

What this does:
- Loads ward-level Tokyo index (jp_tokyo_index.csv)
- Runs:
    1) Fixed OLS using raw % predictors (reduced collinearity)
    2) Ridge regression using standardized predictors
    3) PCA on standardized predictors
- Saves:
    out/modeling/ols_summary.txt
    out/modeling/ols_coefficients.csv
    out/modeling/ridge_coefficients.csv
    out/modeling/pca_loadings.csv
    out/modeling/pca_explained_variance.csv
    out/modeling/modeling_report.txt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd

# statsmodels for OLS
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ImportError as exc:
    raise SystemExit(
        "statsmodels is required for 06_modeling_suite.py.\n"
        "Install it with:\n\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install statsmodels\n"
    ) from exc

# scikit-learn for Ridge + PCA
try:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required for 06_modeling_suite.py.\n"
        "Install it with:\n\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install scikit-learn\n"
    ) from exc


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_index(index_path: Path) -> pd.DataFrame:
    if not index_path.is_file():
        raise FileNotFoundError(f"Index CSV not found: {index_path}")
    df = pd.read_csv(index_path)
    if "iso_index" not in df.columns:
        raise ValueError(
            f"'iso_index' column not found in {index_path}. "
            "Make sure 03_build_index.py has been run."
        )
    return df


# ---------------------------------------------------------------------
# 1) FIXED OLS WITH REDUCED COLLINEARITY
# ---------------------------------------------------------------------

def run_fixed_ols(df: pd.DataFrame, outdir: Path) -> dict:
    """
    Run OLS with reduced collinearity using raw % predictors.

    We use:
        iso_index ~ pct_age65p + poverty_rate

    (We drop pct_single65p because it is nearly collinear with pct_age65p.)
    """
    # Check available predictors
    base_predictors = ["pct_age65p", "poverty_rate"]
    predictors = [c for c in base_predictors if c in df.columns]
    if len(predictors) == 0:
        raise ValueError(
            f"No raw % predictors found among {base_predictors}. "
            "Cannot run fixed OLS."
        )

    formula = "iso_index ~ " + " + ".join(predictors)
    model = smf.ols(formula=formula, data=df).fit()

    # Save summary
    ols_summary_path = outdir / "ols_summary.txt"
    ols_summary_path.write_text(model.summary().as_text(), encoding="utf-8")

    # Save coefficients table
    params = model.params
    bse = model.bse
    tvalues = model.tvalues
    pvalues = model.pvalues
    conf = model.conf_int()
    conf.columns = ["ci_lower", "ci_upper"]

    coef_df = pd.concat([params, bse, tvalues, pvalues, conf], axis=1)
    coef_df.columns = ["coef", "std_err", "t", "p_value", "ci_lower", "ci_upper"]
    coef_df.to_csv(outdir / "ols_coefficients.csv")

    print("✅ Fixed OLS completed.")
    print(f"   Formula: {formula}")
    print(f"   Summary: {ols_summary_path}")

    return {
        "model": model,
        "formula": formula,
        "predictors": predictors,
        "coefficients": coef_df,
    }


# ---------------------------------------------------------------------
# 2) RIDGE REGRESSION (STANDARDIZED PREDICTORS)
# ---------------------------------------------------------------------

def run_ridge(df: pd.DataFrame, outdir: Path) -> dict:
    """
    Run Ridge regression using standardized predictors.

    We use z-scored variables if available, otherwise raw %:
        X = [pct_age65p_z, pct_single65p_z, poverty_rate_z]
    or fallback to their raw counterparts.

    y = iso_index
    """
    # Prefer z-score predictors if present
    z_cols = ["pct_age65p_z", "pct_single65p_z", "poverty_rate_z"]
    available_z = [c for c in z_cols if c in df.columns]

    if len(available_z) >= 2:
        X_cols = available_z
    else:
        # fallback to raw % columns
        raw_cols = ["pct_age65p", "pct_single65p", "poverty_rate"]
        X_cols = [c for c in raw_cols if c in df.columns]

    if len(X_cols) == 0:
        raise ValueError(
            "No suitable predictors found for Ridge regression "
            "(neither z-scores nor raw % columns)."
        )

    X = df[X_cols].values
    y = df["iso_index"].values

    # Standardize predictors
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Cross-validated ridge (no store_cv_values — not supported in your sklearn)
    alphas = np.logspace(-3, 3, 20)
    ridge = RidgeCV(alphas=alphas)   # <--- FIXED LINE
    ridge.fit(X_std, y)

    best_alpha = ridge.alpha_
    coefs = ridge.coef_

    ridge_df = pd.DataFrame(
        {"predictor": X_cols, "coef": coefs},
    )
    ridge_df.to_csv(outdir / "ridge_coefficients.csv", index=False)

    print("✅ Ridge regression completed.")
    print(f"   Predictors: {X_cols}")
    print(f"   Best alpha: {best_alpha:.4f}")

    return {
        "model": ridge,
        "scaler": scaler,
        "X_cols": X_cols,
        "alpha": best_alpha,
        "coeff_df": ridge_df,
    }



# ---------------------------------------------------------------------
# 3) PCA ON PREDICTORS
# ---------------------------------------------------------------------

def run_pca(df: pd.DataFrame, outdir: Path) -> dict:
    """
    Run PCA on the standardized predictors.

    Uses z-scores if present, else raw % (with standardization).
    """
    z_cols = ["pct_age65p_z", "pct_single65p_z", "poverty_rate_z"]
    available_z = [c for c in z_cols if c in df.columns]

    if len(available_z) >= 2:
        X_cols = available_z
    else:
        raw_cols = ["pct_age65p", "pct_single65p", "poverty_rate"]
        X_cols = [c for c in raw_cols if c in df.columns]

    if len(X_cols) < 2:
        raise ValueError(
            "Need at least 2 predictors for PCA; "
            f"found only {X_cols}"
        )

    X = df[X_cols].values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # PCA with up to len(X_cols) components
    n_components = len(X_cols)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_std)

    # Loadings table (components x variables)
    loadings = pd.DataFrame(
        pca.components_,
        columns=X_cols,
        index=[f"PC{i+1}" for i in range(n_components)],
    )
    loadings.to_csv(outdir / "pca_loadings.csv")

    # Explained variance
    evr = pd.DataFrame(
        {
            "PC": [f"PC{i+1}" for i in range(n_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    evr.to_csv(outdir / "pca_explained_variance.csv", index=False)

    # Correlation between iso_index and PC1 scores
    iso = df["iso_index"].values
    pc1_scores = scores[:, 0]
    corr_pc1 = np.corrcoef(iso, pc1_scores)[0, 1]

    print("✅ PCA completed.")
    print(f"   Predictors: {X_cols}")
    print(f"   Explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"   Corr(iso_index, PC1) = {corr_pc1:.3f}")

    return {
        "pca": pca,
        "scaler": scaler,
        "X_cols": X_cols,
        "scores": scores,
        "loadings": loadings,
        "explained_variance": evr,
        "corr_iso_pc1": corr_pc1,
    }


# ---------------------------------------------------------------------
# Combined report writer
# ---------------------------------------------------------------------

def write_combined_report(
    outdir: Path,
    df: pd.DataFrame,
    ols_result: dict,
    ridge_result: dict,
    pca_result: dict,
) -> None:
    report_path = outdir / "modeling_report.txt"

    iso_desc = df["iso_index"].describe()

    text = []

    text.append("=" * 72)
    text.append("Tokyo Isolation Index — Modeling Suite Summary")
    text.append("=" * 72)
    text.append("")

    # Dataset info
    text.append("Dataset info")
    text.append("-" * 72)
    text.append(f"Number of wards: {len(df)}")
    text.append(f"Columns: {', '.join(df.columns)}")
    text.append("")
    text.append("iso_index (z-score) summary:")
    for stat in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
        text.append(f"  {stat:>5}: {iso_desc[stat]:8.3f}")
    text.append("")

    # OLS
    text.append("1) Fixed OLS (reduced collinearity)")
    text.append("-" * 72)
    text.append(f"Formula: {ols_result['formula']}")
    text.append("")
    text.append("Coefficients:")
    text.append(ols_result["coefficients"].to_string(float_format=lambda x: f"{x: .4f}"))
    text.append("")

    # Ridge
    text.append("2) Ridge regression (standardized predictors)")
    text.append("-" * 72)
    text.append(f"Predictors used: {', '.join(ridge_result['X_cols'])}")
    text.append(f"Best alpha (CV): {ridge_result['alpha']:.4f}")
    text.append("")
    text.append("Coefficients:")
    text.append(ridge_result["coeff_df"].to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    text.append("")

    # PCA
    text.append("3) PCA on predictors")
    text.append("-" * 72)
    text.append(f"Predictors used: {', '.join(pca_result['X_cols'])}")
    text.append("")
    text.append("Explained variance ratio:")
    text.append(
        pca_result["explained_variance"].to_string(index=False, float_format=lambda x: f"{x: .4f}")
    )
    text.append("")
    text.append("Loadings (components x predictors):")
    text.append(pca_result["loadings"].to_string(float_format=lambda x: f"{x: .4f}"))
    text.append("")
    text.append(f"Correlation between iso_index and PC1 scores: {pca_result['corr_iso_pc1']:.3f}")
    text.append("")

    text.append("End of modeling suite report.")
    text.append("")

    report_path.write_text("\n".join(text), encoding="utf-8")
    print(f"📝 Wrote combined modeling report: {report_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run OLS, Ridge regression, and PCA for Tokyo isolation index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--index-path",
        default="data/processed/jp_tokyo_index.csv",
        help="Path to Tokyo ward-level isolation index CSV.",
    )
    ap.add_argument(
        "--outdir",
        default="out/modeling",
        help="Directory to write modeling outputs.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    index_path = Path(args.index_path)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    print(f"📥 Loading index from {index_path} ...")
    df = load_index(index_path)

    # 1) OLS
    print("▶ Running fixed OLS ...")
    ols_result = run_fixed_ols(df, outdir)

    # 2) Ridge
    print("▶ Running Ridge regression ...")
    ridge_result = run_ridge(df, outdir)

    # 3) PCA
    print("▶ Running PCA ...")
    pca_result = run_pca(df, outdir)

    # Combined report
    write_combined_report(outdir, df, ols_result, ridge_result, pca_result)

    print("✅ Modeling suite complete.")


if __name__ == "__main__":
    main()
