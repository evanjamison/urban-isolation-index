# -*- coding: utf-8 -*-
"""
05_summary_report.py — summary + diagnostics for Tokyo isolation index

What this does
--------------
- Loads the Tokyo isolation index table produced by 03_build_index.py
  (default: data/processed/jp_tokyo_index.csv).
- Computes summary statistics for iso_index.
- Identifies top/bottom wards and outliers.
- Saves:
    out/reports/tokyo_iso_summary.txt
    out/plots/tokyo_iso_hist.png
    out/plots/tokyo_iso_box.png

Usage
-----
# basic (uses default paths)
python -m src.cli.05_summary_report

# with explicit paths
python -m src.cli.05_summary_report ^
  --index-csv data/processed/jp_tokyo_index.csv ^
  --report-out out/reports/tokyo_iso_summary.txt ^
  --plots-dir out/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def summarize_iso(df: pd.DataFrame) -> str:
    """Build a human-readable summary string for iso_index."""
    if "iso_index" not in df.columns:
        raise KeyError("Expected 'iso_index' column in index table.")

    iso = df["iso_index"].astype(float)

    n = iso.shape[0]
    desc = iso.describe()

    # Top / bottom 5
    top5 = df.sort_values("iso_index", ascending=False).head(5)
    bottom5 = df.sort_values("iso_index", ascending=True).head(5)

    # Outliers using |z| >= 2 threshold
    high_out = df[df["iso_index"] >= 2.0]
    low_out = df[df["iso_index"] <= -2.0]

    lines: List[str] = []

    lines.append("Tokyo Isolation Index — Summary Report")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Number of wards: {n}")
    lines.append("")
    lines.append("iso_index (z-score) summary:")
    lines.append(f"  Mean   : {desc['mean']:.3f}")
    lines.append(f"  Std    : {desc['std']:.3f}")
    lines.append(f"  Min    : {desc['min']:.3f}")
    lines.append(f"  25%    : {desc['25%']:.3f}")
    lines.append(f"  Median : {desc['50%']:.3f}")
    lines.append(f"  75%    : {desc['75%']:.3f}")
    lines.append(f"  Max    : {desc['max']:.3f}")
    lines.append("")

    def _fmt_block(title: str, frame: pd.DataFrame) -> List[str]:
        if frame.empty:
            return [f"{title}: (none)"]
        out = [title + ":"]
        for _, row in frame.itertuples(name=None):
            # row layout from df: (index, ward_jis, ward_name, ..., iso_index)
            # safer to use .loc
            pass
        return out

    # Rebuild with explicit columns for clarity
    def _block_from_df(title: str, frame: pd.DataFrame) -> List[str]:
        if frame.empty:
            return [f"{title}: (none)"]
        out = [title + ":"]
        for _, r in frame.iterrows():
            ward_code = r.get("ward_jis", "")
            ward_name = r.get("ward_name", "")
            val = float(r["iso_index"])
            out.append(f"  - {ward_code} {ward_name}: iso_index = {val:.3f}")
        return out

    lines.extend(_block_from_df("Top 5 wards by iso_index (highest isolation)", top5))
    lines.append("")
    lines.extend(_block_from_df("Bottom 5 wards by iso_index (lowest isolation)", bottom5))
    lines.append("")

    lines.extend(_block_from_df("High outliers (iso_index ≥ 2.0)", high_out))
    lines.append("")
    lines.extend(_block_from_df("Low outliers (iso_index ≤ -2.0)", low_out))
    lines.append("")

    # If there are other interesting columns, list them
    metric_cols = [
        c for c in df.columns
        if c not in ("ward_jis", "ward_name", "iso_index")
        and df[c].dtype != "object"
    ]
    if metric_cols:
        lines.append("Other numeric metrics present:")
        for c in metric_cols:
            lines.append(f"  - {c}")
        lines.append("")

    lines.append("End of report.")
    return "\n".join(lines)


def make_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    """Create histogram and boxplot for iso_index."""

    plots_dir.mkdir(parents=True, exist_ok=True)

    iso = df["iso_index"].astype(float)

    # Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(iso, bins=10)
    plt.axvline(0, linestyle="--")
    plt.axvline(1, linestyle=":",)
    plt.axvline(-1, linestyle=":",)
    plt.axvline(2, linestyle="--")
    plt.axvline(-2, linestyle="--")
    plt.title("Tokyo isolation index — histogram")
    plt.xlabel("iso_index (z-score)")
    plt.ylabel("Count")
    hist_path = plots_dir / "tokyo_iso_hist.png"
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()

    # Boxplot
    plt.figure(figsize=(3, 4))
    plt.boxplot(iso, vert=True)
    plt.ylabel("iso_index (z-score)")
    plt.title("Tokyo isolation index — boxplot")
    box_path = plots_dir / "tokyo_iso_box.png"
    plt.tight_layout()
    plt.savefig(box_path, dpi=150)
    plt.close()

    print(f"✅ Saved histogram: {hist_path}")
    print(f"✅ Saved boxplot : {box_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize Tokyo isolation index and produce diagnostics."
    )
    ap.add_argument(
        "--index-csv",
        default="data/processed/jp_tokyo_index.csv",
        help="CSV file with ward_jis, ward_name, iso_index, etc.",
    )
    ap.add_argument(
        "--report-out",
        default="out/reports/tokyo_iso_summary.txt",
        help="Path to write the text summary report.",
    )
    ap.add_argument(
        "--plots-dir",
        default="out/plots",
        help="Directory to save diagnostic plots.",
    )

    args = ap.parse_args()

    index_path = Path(args.index_csv)
    report_path = Path(args.report_out)
    plots_dir = Path(args.plots_dir)

    if not index_path.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_path}")

    df = pd.read_csv(index_path)
    if "iso_index" not in df.columns:
        raise KeyError(
            f"'iso_index' column not found in {index_path}. "
            "Make sure 03_build_index.py has been run."
        )

    _ensure_dir(report_path)

    report_text = summarize_iso(df)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"✅ Wrote summary report to: {report_path}")
    print("")
    print(report_text)
    print("")

    make_plots(df, plots_dir)


if __name__ == "__main__":
    main()
