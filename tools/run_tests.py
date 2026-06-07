#!/usr/bin/env python3
"""Run accuracy and performance tests for FlagTensor operators.

Usage:
    python tools/run_tests.py                              # run all accuracy tests
    python tools/run_tests.py --op abs,sinh,add             # specific ops
    python tools/run_tests.py --category unary              # category filter
    python tools/run_tests.py --run-perf                    # run performance tests
    python tools/run_tests.py --gpus 0                      # specify GPU
"""
import argparse
import json
import os
import re
import subprocess
import sys
import yaml
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent
CONF = ROOT / "conf" / "operators.yaml"

ANSI_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')


def load_operators():
    with open(CONF) as f:
        data = yaml.safe_load(f)
    return data.get("operators", [])


def parse_pytest_summary(text):
    clean = ANSI_RE.sub("", text)
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for m in re.finditer(r"(\d+)\s+([A-Za-z_]+)", clean):
        key = m.group(2).lower()
        if key in counts:
            counts[key] = int(m.group(1))
    counts["total"] = counts["passed"] + counts["failed"] + counts["skipped"]
    return counts


CATEGORY_TEST_FILE = {
    "unary": "tests/unary/test_unary_correctness.py",
    "binary": "tests/binary/test_binary_correctness.py",
    "contraction": "tests/contraction/test_contraction_correctness.py",
    "sparse": "tests/sparse/test_sparse_correctness.py",
}
CATEGORY_BENCH_FILE = {
    "unary": "benchmark/test_unary_perf.py",
    "binary": "benchmark/test_binary_perf.py",
    "contraction": "benchmark/test_contraction_perf.py",
    "sparse": "benchmark/test_sparse_perf.py",
}


def run_pytest(marker, test_file, gpu_id, extra_args=""):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = f"{sys.executable} -m pytest -m '{marker}' -v --tb=short {extra_args} {test_file}"
    p = subprocess.run(cmd, shell=True, cwd=ROOT, env=env, capture_output=True, text=True)
    output = p.stdout + "\n" + p.stderr
    return parse_pytest_summary(output), p.returncode, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", help="Comma-separated operator names")
    parser.add_argument("--category", help="Operator category (unary/binary/contraction/sparse)")
    parser.add_argument(
        "--stages", default=None,
        help="Comma-separated operator stages (alpha/beta/stable/experimental/active). 'all' selects all stages.",
    )
    parser.add_argument("--run-perf", action="store_true", help="Run performance tests instead of accuracy")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--results-dir", default=None, help="Output directory")
    args = parser.parse_args()

    ops = load_operators()
    if args.op:
        selected = set(args.op.split(","))
        ops = [o for o in ops if o["name"] in selected]
    if args.category:
        ops = [o for o in ops if o["category"] == args.category]

    # --stages filtering
    if args.stages:
        if args.stages == "all":
            pass  # no filtering
        else:
            selected_stages = set(args.stages.split(","))
            filtered = []
            for op_item in ops:
                stage_dicts = op_item.get("stages", [])
                op_stages = set()
                for s in stage_dicts:
                    if isinstance(s, dict):
                        op_stages.update(s.keys())
                if not op_stages:
                    # Fallback: use 'status' field
                    status = op_item.get("status", "stable")
                    op_stages = {status}
                if op_stages & selected_stages:
                    filtered.append(op_item)
            ops = filtered

    if not ops:
        print("[ERROR] No operators matched")
        sys.exit(1)

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    results_dir = Path(args.results_dir) if args.results_dir else ROOT / "results" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    file_map = CATEGORY_BENCH_FILE if args.run_perf else CATEGORY_TEST_FILE

    for i, op in enumerate(ops):
        marker = op.get("benchmark_mark" if args.run_perf else "correctness_mark", op["name"])
        gpu = gpus[i % len(gpus)]
        test_file = file_map.get(op["category"], "tests")
        print(f"[{i+1}/{len(ops)}] {op['name']} (GPU {gpu}, {test_file})...", end=" ", flush=True)

        summary, code, output = run_pytest(marker, test_file, gpu)
        status = "PASS" if (code == 0 and summary["failed"] == 0 and summary["errors"] == 0) else "FAIL"
        results[op["name"]] = {**summary, "status": status, "gpu": gpu}
        print(f"{status} ({summary['passed']}/{summary['total']})")

        log_dir = results_dir / op["name"]
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / ("perf.log" if args.run_perf else "accuracy.log")).write_text(output)

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {summary_path}")
    print(f"Passed: {sum(1 for r in results.values() if r['status']=='PASS')}/{len(results)}")


if __name__ == "__main__":
    main()
