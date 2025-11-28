"""
09_model_with_access.py

Run a simple OLS model for the *new* isolation index that includes access:

   iso_index_with_access ~ pct_age65p + pct_single65p + poverty_rate + access_z

Usage (from project root):

  .\.venv\Scripts\python.exe -m src.cli.09_model_with_access
"""

from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def run_model(index_path: str, outdir: str) -> None:
    index_path = Path(index_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading data from {index_path} ...")
    df = pd.read_csv(index_path, dtype={"ward_jis": "int64"})

    # Quick sanity check
    print("\n🔎 Columns available:")
    print(df.columns.tolist())

    print("\n📊 iso_index_with_access summary:")
    print(df["iso_index_with_access"].describe())

    print("\n📊 access_z summary:")
    print(df["access_z"].describe())

    # ---- OLS model -------------------------------------------------------
    formula = "iso_index_with_access ~ pct_age65p + pct_single65p + poverty_rate + access_z"
    print(f"\n🧠 Fitting OLS model:\n  {formula}\n")

    model = smf.ols(formula=formula, data=df).fit()

    # Print to console
    print(model.summary())

    # Save text summary
    summary_path = outdir / "ols_with_access_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())
    print(f"\n✅ Saved OLS summary to {summary_path}")

    # Also save a tidy coefficient table
    coefs = (
        model.summary2()
        .tables[1]
        .reset_index()
        .rename(columns={"index": "term"})
    )
    coefs_path = outdir / "ols_with_access_coefs.csv"
    coefs.to_csv(coefs_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved coefficient table to {coefs_path}")


def main() -> None:
    index_path = "data/processed/jp_tokyo_index_with_access.csv"
    outdir = "out/modeling_with_access"
    run_model(index_path, outdir)


if __name__ == "__main__":
    main()
