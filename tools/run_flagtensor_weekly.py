#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger("flagtensor.weekly")
logging.basicConfig(
    level=logging.INFO,
    format="[flagtensor-weekly] %(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_utils import build_env_payload
from flagtensor_registry import filter_operator_specs
from flagtensor_registry import load_operator_registry

SUMMARY_LOCK = threading.Lock()
GLOBAL_RESULTS = {}
SPEED_RE = re.compile(
    r"shape=(?P<shape>.+?) dtype=(?P<dtype>\S+) triton_ms=(?P<triton>[0-9.eE+-]+) cutensor_ms=(?P<cutensor>[0-9.eE+-]+) speedup=(?P<speedup>[0-9.eE+-]+)x"
)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_ops(op_list=None, category=None, registry=None, include_blocked=False, mode=None):
    specs = load_operator_registry(registry_path=registry)
    selected = []
    if op_list:
        with open(op_list, encoding="utf-8") as f:
            selected = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    filtered = filter_operator_specs(
        specs,
        names=selected or None,
        categories=category,
        include_blocked=include_blocked,
        mode=mode,
    )
    return [spec.name for spec in filtered]


def now_ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_cmd_capture(cmd, cwd=None, env=None):
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = process.communicate()
    return out or "", err or "", process.returncode


def parse_pytest_summary_from_text(text):
    counters = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for name in counters:
        match = re.search(rf"(\d+)\s+{name}", text)
        if match:
            counters[name] = int(match.group(1))
    total = counters["passed"] + counters["failed"] + counters["skipped"]
    return counters, total


def parse_perf_rows(text):
    rows = []
    for line in text.splitlines():
        match = SPEED_RE.search(line.strip())
        if not match:
            continue
        rows.append(
            {
                "shape": match.group("shape"),
                "dtype": match.group("dtype"),
                "triton_ms": float(match.group("triton")),
                "cutensor_ms": float(match.group("cutensor")),
                "speedup": float(match.group("speedup")),
            }
        )
    return rows


def run_accuracy(op, gpu_id, project_root, op_dir):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = f'{sys.executable} -m pytest tests -m "{op}" -vs'
    out, err, code = run_cmd_capture(cmd, cwd=project_root, env=env)
    combined = out + "\n" + err
    log_path = os.path.join(op_dir, "accuracy.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(combined)
    counters, total = parse_pytest_summary_from_text(combined)
    if counters["failed"] > 0 or (counters["errors"] > 0 and total == 0):
        status = "FAIL"
    else:
        status = "PASS"
    return {
        "status": status,
        "passed": counters["passed"],
        "failed": counters["failed"],
        "skipped": counters["skipped"],
        "errors": counters["errors"],
        "total": total,
        "exit_code": code,
        "log_path": log_path,
    }


def run_benchmark(op, gpu_id, project_root, op_dir):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("FLAGTENSOR_BENCHMARK_MODE", "kernel")
    cmd = f'{sys.executable} -m pytest benchmark -m "{op}" -vs'
    out, err, code = run_cmd_capture(cmd, cwd=project_root, env=env)
    combined = out + "\n" + err
    log_path = os.path.join(op_dir, "perf.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(combined)
    rows = parse_perf_rows(combined)
    avg_speedup = sum(row["speedup"] for row in rows) / len(rows) if rows else 0.0
    max_speedup = max((row["speedup"] for row in rows), default=0.0)
    status = "PASS" if code == 0 else "FAIL"
    return {
        "status": status,
        "exit_code": code,
        "log_path": log_path,
        "performance_rows": rows,
        "avg_speedup": avg_speedup,
        "max_speedup": max_speedup,
    }


def write_summary(summary_map, results_dir):
    from openpyxl import Workbook

    summary_path = os.path.join(results_dir, "summary.json")
    payload = []
    for op, info in sorted(summary_map.items()):
        payload.append(
            {
                "operator": op,
                "gpu": info["gpu"],
                "accuracy": info["accuracy"],
                "performance": info["performance"],
            }
        )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    markdown_path = os.path.join(results_dir, "summary.md")
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("# FlagTensor Weekly Summary\n\n")
        f.write("| operator | gpu | acc_status | passed | failed | skipped | perf_status | avg_speedup | max_speedup |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for op, info in sorted(summary_map.items()):
            accuracy = info["accuracy"]
            performance = info["performance"]
            f.write(
                f"| {op} | {info['gpu']} | {accuracy['status']} | {accuracy['passed']} | "
                f"{accuracy['failed']} | {accuracy['skipped']} | {performance['status']} | "
                f"{performance['avg_speedup']:.6f} | {performance['max_speedup']:.6f} |\n"
            )

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "WeeklySummary"
    sheet.append(
        [
            "operator",
            "gpu",
            "acc_status",
            "passed",
            "failed",
            "skipped",
            "errors",
            "total",
            "perf_status",
            "avg_speedup",
            "max_speedup",
            "accuracy_log",
            "perf_log",
        ]
    )
    for op, info in sorted(summary_map.items()):
        accuracy = info["accuracy"]
        performance = info["performance"]
        sheet.append(
            [
                op,
                info["gpu"],
                accuracy["status"],
                accuracy["passed"],
                accuracy["failed"],
                accuracy["skipped"],
                accuracy["errors"],
                accuracy["total"],
                performance["status"],
                performance["avg_speedup"],
                performance["max_speedup"],
                accuracy["log_path"],
                performance["log_path"],
            ]
        )
    workbook.save(os.path.join(results_dir, "summary.xlsx"))


def write_env(project_root, results_dir):
    payload = build_env_payload(project_root=project_root)
    env_path = os.path.join(results_dir, "env.json")
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def worker_process_ops(gpu_id, ops_list, project_root, results_dir):
    for op in ops_list:
        if not op:
            continue
        op_dir = os.path.join(results_dir, op)
        ensure_dir(op_dir)
        accuracy = run_accuracy(op, gpu_id, project_root, op_dir)
        performance = run_benchmark(op, gpu_id, project_root, op_dir)
        with SUMMARY_LOCK:
            GLOBAL_RESULTS[op] = {
                "gpu": gpu_id,
                "accuracy": accuracy,
                "performance": performance,
            }
            write_summary(GLOBAL_RESULTS, results_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--op-list", required=False)
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--include-blocked", action="store_true")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--mode", choices=["kernel", "operator", "wrapper"], default="kernel")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    ops = load_ops(
        op_list=args.op_list,
        category=args.category,
        registry=args.registry,
        include_blocked=args.include_blocked,
        mode=args.mode,
    )
    if not ops:
        raise SystemExit("No operators selected from registry/op-list")
    os.environ["FLAGTENSOR_BENCHMARK_MODE"] = args.mode
    gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    results_dir = args.results_dir or os.path.join(args.project_root, f"weekly_results_{now_ts()}")
    ensure_dir(results_dir)
    write_env(args.project_root, results_dir)

    tasks = {gpu: [] for gpu in gpus}
    for idx, op in enumerate(ops):
        tasks[gpus[idx % len(gpus)]].append(op)

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(worker_process_ops, gpu, tasks[gpu], args.project_root, results_dir)
            for gpu in gpus
            if tasks[gpu]
        ]
        for future in as_completed(futures):
            future.result()

    logger.info("weekly run complete results_dir=%s", results_dir)
    sys.stdout.write(json.dumps({"results_dir": results_dir}, indent=2) + "\n")


if __name__ == "__main__":
    main()
