#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Analyze speedup_detail.csv to produce per-op tables and charts.

Usage:
  python tools/analyze_speedup.py                      # default 0.8 threshold
  python tools/analyze_speedup.py --threshold 0.95     # 95% threshold
  python tools/analyze_speedup.py --threshold 0.95 --data-dir speedup_analysis

Generates per threshold in speedup_analysis_{threshold_pct}/:
- per_op_tables.md    — per-op shape×dtype tables
- charts/              — per-op speedup bar charts
- below_threshold.md   — operators needing work
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parent
# ── Category definitions ───────────────────────────────────────────────────
CATEGORY_MAP = {
    "abs": "unary", "acos": "unary", "acosh": "unary", "asin": "unary",
    "asinh": "unary", "atan": "unary", "atanh": "unary", "ceil": "unary",
    "conj": "unary", "cos": "unary", "cosh": "unary", "exp": "unary",
    "floor": "unary", "identity": "unary", "log": "unary", "mish": "unary",
    "neg": "unary", "rcp": "unary", "relu": "unary", "sigmoid": "unary",
    "sin": "unary", "sinh": "unary", "soft_plus": "unary", "soft_sign": "unary",
    "sqrt": "unary", "swish": "unary", "tan": "unary", "tanh": "unary",
    "add": "binary", "mul": "binary", "max": "binary", "min": "binary",
    "contraction": "contraction", "contraction_trinary": "contraction",
    "elementwise_trinary": "contraction",
    "block_sparse_contraction": "sparse",
}

# Set via command-line; module-level default for backward compat
THRESHOLD = 0.8
THRESHOLD_PCT = 80  # used for directory naming / display


def _threshold_label():
    """Human-readable threshold label, e.g. '0.95x' or '0.8x'."""
    return f"{THRESHOLD:.2f}x"


def _threshold_pct_label():
    """Percentage label, e.g. '95%' or '80%'."""
    return f"{int(THRESHOLD * 100)}%"


def load_data(csv_path):
    """Load speedup_detail.csv, return dict: op_name -> [(shape, dtype, speedup)]"""
    data = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            op = row["op"]
            speedup = float(row["speedup"])
            data[op].append((row["shape"], row["dtype"], speedup))
    return data


def op_stats(op_data):
    """Compute statistics for an operator."""
    speeds = [s for _, _, s in op_data]
    n = len(speeds)
    avg = sum(speeds) / n
    geo = np.prod(speeds) ** (1 / n) if n > 0 else 0
    below = sum(1 for s in speeds if s < THRESHOLD)
    below_rate = below / n * 100 if n > 0 else 0
    return {
        "n": n, "avg": avg, "geo": geo,
        "min": min(speeds), "max": max(speeds),
        "below": below, "below_rate": below_rate,
        "passes": below == 0,
    }


# ── Chart styling ──────────────────────────────────────────────────────────
COLORS = {
    "unary": "#4ECDC4", "binary": "#45B7D1",
    "contraction": "#96CEB4", "sparse": "#FFEAA7",
}
THRESHOLD_COLOR = "#FF6B6B"
PASS_COLOR = "#51CF66"
GLOW_COLOR = "#FFA94D"


def chart_op(op_name, op_data, cat, charts_dir):
    """Generate a grouped bar chart: one group per (shape, dtype), one bar = speedup."""
    # Sort by speedup (ascending) so worst cases are on the left
    rows = sorted(op_data, key=lambda x: x[2])
    labels = [f"{dtype}\n{shape}" for shape, dtype, _ in rows]
    speeds = [s for _, _, s in rows]

    n = len(rows)
    fig_w = max(10, n * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    x = np.arange(n)
    bar_colors = [THRESHOLD_COLOR if s < THRESHOLD else COLORS.get(cat, "#999") for s in speeds]
    bars = ax.bar(x, speeds, color=bar_colors, edgecolor="white", linewidth=0.5)

    # Highlight bars below threshold
    for i, (bar, s) in enumerate(zip(bars, speeds)):
        if s < THRESHOLD:
            bar.set_edgecolor("#FF0000")
            bar.set_linewidth(1.5)

    # Threshold line
    ax.axhline(y=THRESHOLD, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(n - 0.5, THRESHOLD + 0.02, f"Threshold ({THRESHOLD}x)", color=THRESHOLD_COLOR,
            fontsize=8, ha="right", va="bottom", fontstyle="italic")

    # 1.0x baseline
    ax.axhline(y=1.0, color="#666", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Speedup (Triton / cuTensor)")
    ax.set_title(f"Kernel Performance: {op_name} ({cat})", fontsize=12, fontweight="bold")

    stats = op_stats(op_data)
    info = (f"N={stats['n']}  avg={stats['avg']:.3f}x  geo={stats['geo']:.3f}x  "
            f"below_{_threshold_label()}={stats['below']}/{stats['n']}")
    ax.text(0.5, 1.02, info, transform=ax.transAxes, fontsize=8, ha="center", color="#555")

    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = charts_dir / f"{op_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def chart_category_summary(cat, op_summaries, charts_dir):
    """Generate a summary chart for a category showing avg speedup per op."""
    ops = sorted(op_summaries.keys(), key=lambda o: op_summaries[o]["avg"])
    avgs = [op_summaries[o]["avg"] for o in ops]
    geos = [op_summaries[o]["geo"] for o in ops]
    belows = [op_summaries[o]["below"] for o in ops]

    n = len(ops)
    fig_w = max(10, n * 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    x = np.arange(n)
    width = 0.35

    bar_colors = [THRESHOLD_COLOR if avgs[i] < 1.0 else COLORS.get(cat, "#999") for i in range(n)]
    bars_avg = ax.bar(x - width / 2, avgs, width, label="Arithmetic Mean", color=bar_colors, edgecolor="white")
    bars_geo = ax.bar(x + width / 2, geos, width, label="Geometric Mean", color="#BEBEBE", edgecolor="white")

    # Annotate below-threshold counts
    for i, (b, cnt) in enumerate(zip(bars_avg, belows)):
        if cnt > 0:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                    f"{cnt}<{_threshold_label()}", ha="center", fontsize=6,
                    color=THRESHOLD_COLOR, fontweight="bold")

    ax.axhline(y=THRESHOLD, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axhline(y=1.0, color="#666", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Speedup")
    ax.set_title(f"Kernel Performance Summary: {cat.upper()} Operators", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = charts_dir / f"_summary_{cat}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def generate_markdown_tables(all_data, op_summaries):
    """Generate per-op markdown tables."""
    sections = []

    for cat in ["unary", "binary", "contraction", "sparse"]:
        cat_ops = sorted([op for op, cat2 in CATEGORY_MAP.items() if cat2 == cat and op in all_data])

        sections.append(f"\n## {cat.upper()} ({len(cat_ops)} operators)\n")

        for op in cat_ops:
            op_data = all_data[op]
            stats = op_summaries[op]

            status = "✅" if stats["passes"] else f"⚠️ ({stats['below']}/{stats['n']} below {_threshold_label()})"
            sections.append(f"### {op} — avg={stats['avg']:.3f}x geo={stats['geo']:.3f}x {status}\n")

            # Table
            sections.append(f"| Shape | Dtype | Speedup |")
            sections.append(f"|-------|-------|---------|")

            # Sort by speedup ascending (worst first)
            for shape, dtype, speedup in sorted(op_data, key=lambda x: x[2]):
                flag = " ⚠️" if speedup < THRESHOLD else ""
                sections.append(f"| {shape} | {dtype} | {speedup:.4f}x{flag} |")

            sections.append("")

    return "\n".join(sections)


def generate_below_threshold_report(all_data, op_summaries):
    """Generate a report of operators that need attention."""
    lines = []
    lines.append(f"# Operators Below {_threshold_pct_label()} Performance Threshold\n")
    lines.append(f"Threshold: speedup < {_threshold_label()} (Triton vs cuTensor)\n")

    # By category
    for cat in ["unary", "binary", "contraction", "sparse"]:
        cat_ops = {}
        for op, stats in op_summaries.items():
            if CATEGORY_MAP.get(op) == cat and stats["below"] > 0:
                cat_ops[op] = stats

        if not cat_ops:
            continue

        lines.append(f"## {cat.upper()} — {len(cat_ops)} ops need attention\n")

        for op, stats in sorted(cat_ops.items(), key=lambda x: x[1]["below_rate"], reverse=True):
            lines.append(f"### {op}")
            lines.append(f"- **Below-threshold cases:** {stats['below']}/{stats['n']} ({stats['below_rate']:.1f}%)")
            lines.append(f"- **Avg speedup:** {stats['avg']:.4f}x")
            lines.append(f"- **Geo mean:** {stats['geo']:.4f}x")
            lines.append(f"- **Min speedup:** {stats['min']:.4f}x")

            # List specific failing cases
            op_data = all_data[op]
            fails = [(shape, dtype, s) for shape, dtype, s in op_data if s < THRESHOLD]
            if fails:
                lines.append(f"- **Failing cases:**")
                for shape, dtype, s in sorted(fails, key=lambda x: x[2]):
                    lines.append(f"  - `{shape}` {dtype}: {s:.4f}x")

            lines.append("")

    # Summary table
    lines.append("## Summary\n")
    lines.append(f"| Operator | Category | Cases | Below {_threshold_label()} | Min | Avg | Geo Mean |")
    lines.append("|----------|----------|-------|------------|-----|-----|----------|")

    for op, stats in sorted(op_summaries.items(), key=lambda x: x[1]["below_rate"], reverse=True):
        if stats["below"] > 0:
            cat = CATEGORY_MAP.get(op, "unknown")
            lines.append(
                f"| {op} | {cat} | {stats['n']} | {stats['below']} ({stats['below_rate']:.1f}%) "
                f"| {stats['min']:.4f}x | {stats['avg']:.4f}x | {stats['geo']:.4f}x |"
            )

    return "\n".join(lines)


def main():
    global THRESHOLD, THRESHOLD_PCT

    parser = argparse.ArgumentParser(description="Analyze speedup benchmark results")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Speedup threshold (default: 0.8)")
    parser.add_argument("--data-dir", type=str, default="speedup_analysis",
                        help="Directory containing speedup_detail.csv (default: speedup_analysis)")
    args = parser.parse_args()

    THRESHOLD = args.threshold
    THRESHOLD_PCT = int(THRESHOLD * 100)

    DATA_DIR = ROOT / args.data_dir
    CSV_PATH = DATA_DIR / "speedup_detail.csv"

    SPEEDUP_DIR = ROOT / f"speedup_analysis_{THRESHOLD_PCT}"
    CHARTS_DIR = SPEEDUP_DIR / "charts"
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run bench_speedup_all.py first.")
        return

    print(f"Loading data from {CSV_PATH}...")
    all_data = load_data(CSV_PATH)
    print(f"Loaded {len(all_data)} operators, {sum(len(v) for v in all_data.values())} total cases")

    # Compute stats
    op_summaries = {op: op_stats(data) for op, data in all_data.items()}

    # Overall stats
    all_speeds = [s for data in all_data.values() for _, _, s in data]
    total = len(all_speeds)
    below_total = sum(1 for s in all_speeds if s < THRESHOLD)
    print(f"\nOverall: {total} cases, {below_total} below {_threshold_label()} ({below_total/total*100:.1f}%)")

    # Generate per-op charts
    print("\nGenerating per-op charts...")
    for op, op_data in all_data.items():
        cat = CATEGORY_MAP.get(op, "unknown")
        chart_op(op, op_data, cat, CHARTS_DIR)

    # Generate category summary charts
    print("Generating category summary charts...")
    for cat in ["unary", "binary", "contraction", "sparse"]:
        cat_ops = {op: op_summaries[op] for op, c in CATEGORY_MAP.items()
                   if c == cat and op in op_summaries}
        if cat_ops:
            chart_category_summary(cat, cat_ops, CHARTS_DIR)

    # Generate markdown tables
    print("Generating markdown tables...")
    tables_md = generate_markdown_tables(all_data, op_summaries)
    tables_path = SPEEDUP_DIR / "per_op_tables.md"
    tables_path.write_text(tables_md, encoding="utf-8")
    print(f"  -> {tables_path}")

    # Generate below-threshold report
    print("Generating below-threshold report...")
    report_md = generate_below_threshold_report(all_data, op_summaries)
    report_path = SPEEDUP_DIR / "below_threshold.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"  -> {report_path}")

    # Print summary to terminal
    print(f"\n{'='*60}")
    print(f"OPERATORS BELOW {_threshold_pct_label()} THRESHOLD (need work):")
    print(f"{'='*60}")
    below_ops = [(op, s) for op, s in op_summaries.items() if s["below"] > 0]
    if below_ops:
        for op, stats in sorted(below_ops, key=lambda x: x[1]["below_rate"], reverse=True):
            print(f"  {op:35s} {stats['below']}/{stats['n']} cases below {_threshold_label()}  "
                  f"min={stats['min']:.4f}x  avg={stats['avg']:.4f}x")
    else:
        print(f"  None! All operators pass the {_threshold_pct_label()} threshold.")
    print(f"\nCharts: {CHARTS_DIR}")
    print(f"Tables: {tables_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
