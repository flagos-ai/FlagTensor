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

"""Run all operator speedup benchmarks vs cuTensor and produce summary.

Usage: FLAGTENSOR_BENCHMARK_MAX_SHAPES=5 python tools/bench_speedup_all.py
Set FLAGTENSOR_BENCHMARK_MAX_SHAPES to limit shape count for faster runs.
Output: benchmark/results/SPEEDUP/
"""
import csv, json, os, re, subprocess, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPEEDUP_RE = re.compile(r'speedup=(\d+\.?\d*)x')
SHAPE_RE = re.compile(r'shape=\(([^)]+)\)')
DTYPE_RE = re.compile(r'dtype=(torch\.\S+)')

CATEGORY_ORDER = ["unary", "binary", "contraction", "sparse"]
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


def run_pytest_for_op(op_name):
    """Run a single per-op benchmark test via pytest and parse speedup output."""
    test_file = f"benchmark/test_CUTENSOR_OP_{op_name.upper()}_perf.py"
    test_path = os.path.join(ROOT, test_file)
    if not os.path.exists(test_path):
        return []

    cmd = (
        f"cd {ROOT} && python -m pytest {test_file} -v -s --no-header -p no:warnings "
        f"--tb=short 2>&1"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr

    results = []
    for line in output.splitlines():
        m = SPEEDUP_RE.search(line)
        if m:
            shape_m = SHAPE_RE.search(line)
            dtype_m = DTYPE_RE.search(line)
            results.append({
                "shape": shape_m.group(1) if shape_m else "?",
                "dtype": dtype_m.group(1) if dtype_m else "?",
                "speedup": float(m.group(1)),
            })
    return results


def main():
    os.makedirs(os.path.join(ROOT, "benchmark/results/SPEEDUP"), exist_ok=True)

    # Determine all ops to run
    all_ops = sorted(CATEGORY_MAP.keys())
    print(f"Running speedup benchmarks for {len(all_ops)} operators...")
    print(f"Max shapes: {os.getenv('FLAGTENSOR_BENCHMARK_MAX_SHAPES', 'all')}")
    print(f"Dtypes: {os.getenv('FLAGTENSOR_BENCHMARK_DTYPES', 'default')}")
    print()

    all_results = {}    # op_name -> [detail dicts]
    errors = []

    for idx, op_name in enumerate(all_ops, 1):
        print(f"[{idx:2d}/{len(all_ops)}] {op_name:35s}...", end=" ", flush=True)
        try:
            results = run_pytest_for_op(op_name)
            if results:
                all_results[op_name] = results
                speeds = [r["speedup"] for r in results]
                avg = sum(speeds) / len(speeds)
                geo = 1.0
                for s in speeds:
                    geo *= s
                geo = geo ** (1.0 / len(speeds))
                n_cases = len(speeds)
                lt_08 = sum(1 for s in speeds if s < 0.8)
                flag = f"  *** {lt_08} cases <0.8x" if lt_08 > 0 else ""
                print(f"{n_cases:3d} cases  avg={avg:.3f}x  geo={geo:.3f}x{flag}")
            else:
                errors.append(op_name)
                print("NO SPEEDUP DATA")
        except Exception as e:
            errors.append(op_name)
            print(f"ERROR: {e}")

    # ── Build summary ──
    print("\n" + "=" * 80)
    print("SPEEDUP SUMMARY: Triton vs cuTensor")
    print("=" * 80)

    category_data = defaultdict(list)
    for op_name, results in sorted(all_results.items()):
        speeds = [r["speedup"] for r in results]
        cat = CATEGORY_MAP.get(op_name, "unknown")
        for r in results:
            category_data[cat].append(r["speedup"])
        category_data[f"{cat}_ops"].append({
            "op": op_name,
            "avg": sum(speeds) / len(speeds),
            "min": min(speeds),
            "max": max(speeds),
            "cases": len(speeds),
            "below_08x": sum(1 for s in speeds if s < 0.8),
        })

    overall_speeds = []
    for cat in CATEGORY_ORDER:
        cat_speeds = category_data.get(cat, [])
        overall_speeds.extend(cat_speeds)
        op_list = category_data.get(f"{cat}_ops", [])

        if not cat_speeds:
            continue

        avg = sum(cat_speeds) / len(cat_speeds)
        geo = 1.0
        for s in cat_speeds:
            geo *= s
        geo = geo ** (1.0 / len(cat_speeds))
        lt_08 = sum(1 for s in cat_speeds if s < 0.8)
        gt_1 = sum(1 for s in cat_speeds if s >= 1.0)

        print(f"\n--- {cat.upper()} ({len(op_list)} ops, {len(cat_speeds)} cases) ---")
        for op in sorted(op_list, key=lambda x: x["avg"]):
            marker = " *** BELOW 0.8x" if op["avg"] < 0.8 else ""
            print(f"  {op['op']:32s} avg={op['avg']:.3f}x  min={op['min']:.3f}x  max={op['max']:.3f}x  n={op['cases']}{marker}")
        print(f"  Category: avg={avg:.3f}x  geo_mean={geo:.3f}x  <0.8x={lt_08}/{len(cat_speeds)}  >=1.0x={gt_1}/{len(cat_speeds)}")

    # Overall
    total = len(overall_speeds)
    overall_avg = sum(overall_speeds) / total if total else 0
    overall_geo = 1.0
    for s in overall_speeds:
        overall_geo *= s
    overall_geo = overall_geo ** (1.0 / total) if total else 0
    lt_08 = sum(1 for s in overall_speeds if s < 0.8)
    gt_1 = sum(1 for s in overall_speeds if s >= 1.0)
    pct_08 = 100 * lt_08 / total if total else 0

    print(f"\n{'=' * 80}")
    print(f"OVERALL: {len(all_results)} ops, {total} cases")
    print(f"  avg speedup = {overall_avg:.3f}x")
    print(f"  geo mean    = {overall_geo:.3f}x")
    print(f"  < 0.8x      = {lt_08}/{total} ({pct_08:.1f}%)")
    print(f"  >= 1.0x     = {gt_1}/{total} ({100*gt_1/total:.1f}%)" if total else "")
    print(f"  errors       = {len(errors)}" + (f": {', '.join(errors)}" if errors else ""))
    pct_80 = 100 * (total - lt_08) / total if total else 0
    print(f"\n  达到 80% 性能的 case 占比: {pct_80:.1f}%")
    print(f"  达标 (≥80%): {'YES' if pct_80 >= 80 else 'NO'} (规范要求 ≥80% 算子不低于 80%)")

    # ── Save detail CSV ──
    csv_path = os.path.join(ROOT, "benchmark/results/SPEEDUP/speedup_detail.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["op", "category", "shape", "dtype", "speedup"])
        w.writeheader()
        for op_name, results in sorted(all_results.items()):
            cat = CATEGORY_MAP.get(op_name, "unknown")
            for r in results:
                w.writerow({"op": op_name, "category": cat, "shape": r["shape"], "dtype": r["dtype"], "speedup": r["speedup"]})
    print(f"\nDetail CSV: {csv_path}")

    # ── Save summary JSON ──
    json_path = os.path.join(ROOT, "benchmark/results/SPEEDUP/speedup_summary.json")
    summary = {
        "overall": {
            "ops_count": len(all_results),
            "total_cases": total,
            "avg_speedup": round(overall_avg, 4),
            "geo_mean_speedup": round(overall_geo, 4),
            "below_08x": lt_08,
            "above_1x": gt_1,
            "pct_above_80pct": round(pct_80, 1),
            "pass_80pct_threshold": pct_80 >= 80,
            "errors": errors,
        },
        "by_category": {},
        "by_op": {},
    }
    for cat in CATEGORY_ORDER:
        op_list = category_data.get(f"{cat}_ops", [])
        cat_speeds = category_data.get(cat, [])
        if cat_speeds:
            cat_avg = sum(cat_speeds) / len(cat_speeds)
            summary["by_category"][cat] = {
                "ops_count": len(op_list),
                "cases": len(cat_speeds),
                "avg_speedup": round(cat_avg, 4),
                "below_08x": sum(1 for s in cat_speeds if s < 0.8),
            }
    for op_name, results in sorted(all_results.items()):
        speeds = [r["speedup"] for r in results]
        summary["by_op"][op_name] = {
            "category": CATEGORY_MAP.get(op_name, "unknown"),
            "avg": round(sum(speeds) / len(speeds), 4),
            "min": round(min(speeds), 4),
            "max": round(max(speeds), 4),
            "cases": len(speeds),
            "below_08x": sum(1 for s in speeds if s < 0.8),
        }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON: {json_path}")


if __name__ == "__main__":
    main()
