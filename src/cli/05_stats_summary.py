"""
05_stats_summary.py — statistical analysis for Tokyo isolation index

Usage (from project root):

  .\.venv\Scripts\python.exe -m src.cli.05_stats_summary

Optional args:

  --index-path  path to index CSV (default: data/processed/jp_tokyo_index.csv)
  --outdir      output directory for stats (default: out/stats)

This script will:

  * Load the ward-level isolation index dataset
  * Compute Pearson & Spearman correlations
  * Fit an OLS regression for iso_index
  * Run a one-way ANOVA by poverty terciles
  * Save tables + a plain-text report into out/stats/
"""

import argparse
import os
from textwrap import dedent

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
except ImportError as exc:  # helpful error if statsmodels isn't installed
    raise SystemExit(
        "statsmodels is required for 05_stats_summary.py.\n"
        "Install it with:\n\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install statsmodels\n"
    ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_index(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index file not found: {path}")
    df = pd.read_csv(path)
    # Basic sanity check
    if "iso_index" not in df.columns:
        raise ValueError(
            f"'iso_index' column not found in {path}. "
            "Make sure 03_build_index has been run."
        )
    return df


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_correlations(df: pd.DataFrame, metrics: list[str]) -> dict[str, pd.DataFrame]:
    """Return Pearson & Spearman correlation matrices for the given columns."""
    cols = [c for c in metrics if c in df.columns]
    if len(cols) < 2:
        raise ValueError(
            f"Need at least 2 of {metrics} to compute correlations; found only {cols}"
        )
    pearson = df[cols].corr(method="pearson")
    spearman = df[cols].corr(method="spearman")
    return {"pearson": pearson, "spearman": spearman, "used_cols": cols}


def fit_regression(df: pd.DataFrame, predictors: list[str]):
    """Fit OLS iso_index ~ predictors (only using columns that exist)."""
    used = [c for c in predictors if c in df.columns]
    if not used:
        raise ValueError(
            f"No predictor columns available from {predictors}. "
            "Cannot fit regression."
        )

    formula = "iso_index ~ " + " + ".join(used)
    model = smf.ols(formula=formula, data=df).fit()
    return model, formula, used


def run_anova_poverty_terciles(df: pd.DataFrame):
    """
    One-way ANOVA: iso_index by poverty terciles.

    If poverty_rate_z is missing, returns (None, None).
    """
    if "poverty_rate_z" not in df.columns:
        return None, None

    # Create terciles: low, mid, high poverty
    df = df.copy()
    df["poverty_group"] = pd.qcut(
        df["poverty_rate_z"],
        q=3,
        labels=["low", "mid", "high"],
        duplicates="drop",
    )

    model = smf.ols("iso_index ~ C(poverty_group)", data=df).fit()
    table = anova_lm(model, typ=2)
    return table, df


def write_report(
    outdir: str,
    df: pd.DataFrame,
    corr_result: dict,
    reg_model,
    reg_formula: str,
    reg_predictors: list[str],
    anova_table: pd.DataFrame | None,
) -> None:
    """Write a text report summarising all results."""
    report_path = os.path.join(outdir, "tokyo_stats_report.txt")

    pearson = corr_result["pearson"]
    spearman = corr_result["spearman"]

    # Basic summary stats for iso_index
    iso = df["iso_index"].describe()

    # Top/bottom 5 for context
    top5 = df.sort_values("iso_index", ascending=False).head(5)
    bottom5 = df.sort_values("iso_index", ascending=True).head(5)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 68 + "\n")
        f.write("Tokyo Isolation Index — Statistical Summary\n")
        f.write("=" * 68 + "\n\n")

        # ------------------------------------------------------------------
        f.write("Dataset info\n")
        f.write("-" * 68 + "\n")
        f.write(f"Number of wards: {len(df)}\n")
        f.write(f"Columns available: {', '.join(df.columns)}\n\n")

        # ------------------------------------------------------------------
        f.write("iso_index (z-score) summary\n")
        f.write("-" * 68 + "\n")
        for stat in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
            f.write(f"{stat:>5}: {iso[stat]:8.3f}\n")
        f.write("\n")

        # ------------------------------------------------------------------
        f.write("Top 5 wards by iso_index (highest isolation)\n")
        f.write("-" * 68 + "\n")
        for _, row in top5.iterrows():
            f.write(
                f"- {int(row['ward_jis'])} {row['ward_name']}: "
                f"iso_index = {row['iso_index']:.3f}\n"
            )
        f.write("\n")

        f.write("Bottom 5 wards by iso_index (lowest isolation)\n")
        f.write("-" * 68 + "\n")
        for _, row in bottom5.iterrows():
            f.write(
                f"- {int(row['ward_jis'])} {row['ward_name']}: "
                f"iso_index = {row['iso_index']:.3f}\n"
            )
        f.write("\n")

        # ------------------------------------------------------------------
        f.write("Correlation analysis (iso_index vs other metrics)\n")
        f.write("-" * 68 + "\n")
        f.write("Pearson correlations:\n")
        f.write(pearson.to_string(float_format=lambda x: f"{x: .3f}") + "\n\n")
        f.write("Spearman correlations:\n")
        f.write(spearman.to_string(float_format=lambda x: f"{x: .3f}") + "\n\n")

        # ------------------------------------------------------------------
        f.write("OLS regression\n")
        f.write("-" * 68 + "\n")
        f.write(f"Formula: {reg_formula}\n\n")
        f.write(reg_model.summary().as_text() + "\n\n")

        # ------------------------------------------------------------------
        if anova_table is not None:
            f.write("One-way ANOVA: iso_index by poverty terciles\n")
            f.write("-" * 68 + "\n")
            f.write(anova_table.to_string(float_format=lambda x: f"{x: .4f}") + "\n\n")
        else:
            f.write("ANOVA: poverty_rate_z not available — skipped.\n\n")

        f.write("End of report.\n")

    print(f"✓ Wrote stats report: {report_path}")


def save_tables(
    outdir: str,
    corr_result: dict,
    reg_model,
    reg_predictors: list[str],
    anova_table: pd.DataFrame | None,
) -> None:
    """Save CSV tables for correlations, regression coeffs, ANOVA."""
    pearson = corr_result["pearson"]
    spearman = corr_result["spearman"]

    pearson.to_csv(os.path.join(outdir, "correlations_pearson.csv"))
    spearman.to_csv(os.path.join(outdir, "correlations_spearman.csv"))

    # Regression coefficients table
    params = reg_model.params
    bse = reg_model.bse
    tvalues = reg_model.tvalues
    pvalues = reg_model.pvalues
    conf = reg_model.conf_int()
    conf.columns = ["ci_lower", "ci_upper"]

    reg_df = pd.concat(
        [params, bse, tvalues, pvalues, conf],
        axis=1,
    )
    reg_df.columns = ["coef", "std_err", "t", "p_value", "ci_lower", "ci_upper"]
    reg_df.to_csv(os.path.join(outdir, "regression_coefficients.csv"))

    if anova_table is not None:
        anova_table.to_csv(os.path.join(outdir, "anova_poverty_terciles.csv"))

    print(f"✓ Saved correlation / regression / ANOVA tables to: {outdir}")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statistical analysis for Tokyo isolation index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index-path",
        default="data/processed/jp_tokyo_index.csv",
        help="Path to ward-level index CSV",
    )
    parser.add_argument(
        "--outdir",
        default="out/stats",
        help="Directory for statistics outputs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_outdir(args.outdir)

    print(f"🔎 Loading index from {args.index_path} ...")
    df = load_index(args.index_path)

    # Metrics we care about (use only those that exist)
    metric_cols = [
        "iso_index",
        "pct_age65p_z",
        "pct_single65p_z",
        "poverty_rate_z",
    ]

    # Correlations
    print("📊 Computing correlations ...")
    corr_result = compute_correlations(df, metric_cols)

    # Regression
    predictors = ["pct_age65p_z", "pct_single65p_z", "poverty_rate_z"]
    print("📈 Fitting OLS regression ...")
    reg_model, reg_formula, used_predictors = fit_regression(df, predictors)

    # ANOVA
    print("🧪 Running one-way ANOVA by poverty terciles ...")
    anova_table, _ = run_anova_poverty_terciles(df)

    # Save tables + report
    save_tables(args.outdir, corr_result, reg_model, used_predictors, anova_table)
    write_report(
        args.outdir,
        df,
        corr_result,
        reg_model,
        reg_formula,
        used_predictors,
        anova_table,
    )

    print("✅ Statistical analysis complete.")


if __name__ == "__main__":
    main()
