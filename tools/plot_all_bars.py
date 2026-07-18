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

"""Generate bar charts comparing Triton vs cuTensor latency.

Reads benchmark/results/*/benchmark_kernel_full.csv and produces
one chart per (operator x dtype) for kernel-mode data.
"""
import ast, csv, os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmark" / "results"
OUTPUT_DIR = ROOT / "benchmark" / "chart"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OPS = [
    "CUTENSOR_OP_GETT",
    "CUTENSOR_OP_TGETT",
    "CUTENSOR_OP_TTGT",
    "CUTENSOR_OP_TENSOR_CONTRACTION_TRINARY",
    "CUTENSOR_OP_BLOCK_SPARSE_TENSOR_CONTRACTION",
]

DISPLAY_NAMES = {
    "CUTENSOR_OP_GETT": "GETT",
    "CUTENSOR_OP_TGETT": "TGETT",
    "CUTENSOR_OP_TTGT": "TTGT",
    "CUTENSOR_OP_TENSOR_CONTRACTION_TRINARY": "TRINARY",
    "CUTENSOR_OP_BLOCK_SPARSE_TENSOR_CONTRACTION": "BLOCK_SPARSE",
}

DTYPE_LABELS = {
    "torch.float16": "float16",
    "torch.float32": "float32",
    "torch.bfloat16": "bfloat16",
}

TRITON_COLOR = "#2196F3"
CUTENSOR_COLOR = "#FF9800"


def parse_shape(raw):
    if isinstance(raw, str):
        return ast.literal_eval(raw)
    return raw


def shape_label(shape):
    s = parse_shape(shape)
    if isinstance(s, (list, tuple)):
        return "x".join(str(d) for d in s[:2])
    return str(s)


def load_all():
    """Load all rows, keyed by (op, dtype, mode)."""
    data = defaultdict(list)
    for op in OPS:
        csv_path = RESULTS_DIR / op / "benchmark_kernel_full.csv"
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                key = (op, row["dtype"], row["mode"])
                data[key].append(row)
    return data


def sort_rows(rows):
    """Sort by total elements (area heuristic)."""
    def key(r):
        s = parse_shape(r["shape"])
        if isinstance(s, (list, tuple)) and len(s) >= 2:
            return s[0] * s[1]
        return 0
    return sorted(rows, key=key)


def make_chart(rows, op, dtype, mode, output_dir):
    """Generate a single bar chart."""
    rows = sort_rows(rows)
    shapes = [shape_label(r["shape"]) for r in rows]
    triton_ms = [float(r["latency"]) for r in rows]
    cutensor_ms = [float(r["latency_base"]) for r in rows]

    n = len(shapes)
    if n == 0:
        return

    dlabel = DTYPE_LABELS.get(dtype, dtype)
    op_name = DISPLAY_NAMES.get(op, op)

    x = range(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, n * 1.6 + 2), 5.5))

    bars_t = ax.bar([i - width / 2 for i in x], triton_ms, width,
                    label="Triton", color=TRITON_COLOR, edgecolor="white", linewidth=0.5, zorder=2)
    bars_c = ax.bar([i + width / 2 for i in x], cutensor_ms, width,
                    label="cuTensor", color=CUTENSOR_COLOR, edgecolor="white", linewidth=0.5, zorder=2)

    ax.set_xlabel("Shape")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"{op_name} — Triton vs cuTensor ({dlabel}, {mode})", fontsize=13, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(shapes, rotation=45 if n > 6 else 0, ha="right" if n > 6 else "center", fontsize=8.5)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    # Log scale if dynamic range > 50x
    all_vals = triton_ms + cutensor_ms
    if max(all_vals) / (min(all_vals) + 1e-9) > 50:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    # Speedup annotations
    for i, (t, c) in enumerate(zip(triton_ms, cutensor_ms)):
        speedup = c / t if t > 0 else 0
        if speedup > 1.01:
            color, tag = "#4CAF50", "Triton wins"
        elif speedup < 0.99:
            color, tag = "#F44336", "cuTensor wins"
        else:
            color, tag = "#9E9E9E", "tie"
        ax.annotate(f"{speedup:.2f}x",
                    xy=(i, max(t, c)), xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5, color=color, fontweight="bold")

    fig.tight_layout()
    fname = f"{op_name}_{dlabel}_{mode}.png"
    out_path = output_dir / fname
    fig.savefig(out_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return fname


def main():
    data = load_all()
    if not data:
        print("No benchmark data found.")
        return

    for (op, dtype, mode), rows in sorted(data.items()):
        # Skip modes we don't need
        if mode not in ("kernel", "operator"):
            continue
        fname = make_chart(rows, op, dtype, mode, OUTPUT_DIR)
        if fname:
            print(f"  {fname}  ({len(rows)} shapes)")

    print(f"\nCharts saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
