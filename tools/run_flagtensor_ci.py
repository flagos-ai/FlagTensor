#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_utils import build_env_payload


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd, cwd=None, env=None):
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out, _ = process.communicate()
    return process.returncode, out or ""


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_ops(op=None, op_list=None):
    if op_list:
        lines = Path(op_list).read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if op:
        return [op]
    raise ValueError("either --op or --op-list is required")


def uppercase_op(op: str) -> str:
    return op.upper()


def correctness_test_path(op: str) -> Path:
    return ROOT / "ctests" / f"test_CUTENSOR_OP_{uppercase_op(op)}.py"


def benchmark_test_path(op: str) -> Path:
    return ROOT / "benchmark" / f"test_CUTENSOR_OP_{uppercase_op(op)}_perf.py"


def export_environment(results_dir: Path):
    payload = build_env_payload(project_root=ROOT)
    output = results_dir / "env.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def write_markdown_summary(summary, results_dir: Path):
    lines = [
        "# FlagTensor CI Summary",
        "",
        "| operator | correctness | perf | libtuner_cold | libtuner_warm |",
        "| --- | --- | --- | --- | --- |",
    ]
    for op in summary["ops"]:
        correctness = summary["correctness"].get(op, {}).get("status", "N/A")
        performance = summary["performance"].get(op, {}).get("status", "N/A")
        libtuner = summary["libtuner_compare"].get(op, {})
        cold = libtuner.get("cold", {}).get("status", "N/A")
        warm = libtuner.get("warm", {}).get("status", "N/A")
        lines.append(f"| {op} | {correctness} | {performance} | {cold} | {warm} |")
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_correctness(op: str, results_dir: Path, env):
    test_path = correctness_test_path(op)
    if not test_path.exists():
        return {
            "status": "MISSING",
            "exit_code": 1,
            "log_path": None,
            "test_path": str(test_path),
        }
    log_path = results_dir / op / "correctness.log"
    cmd = f"pytest -vs {test_path.name}"
    code, output = run_cmd(cmd, cwd=test_path.parent, env=env)
    write_text(log_path, output)
    return {
        "status": "PASS" if code == 0 else "FAIL",
        "exit_code": code,
        "log_path": str(log_path),
        "test_path": str(test_path),
    }


def smoke_env(base_env, smoke: bool, dtypes=None, max_shapes=None, warmup=None, repetitions=None):
    env = dict(base_env)
    if smoke:
        env.setdefault("FLAGTENSOR_BENCHMARK_WARMUP", "2")
        env.setdefault("FLAGTENSOR_BENCHMARK_REPETITIONS", "5")
        env.setdefault("FLAGTENSOR_BENCHMARK_MAX_SHAPES", "2")
        env.setdefault("FLAGTENSOR_BENCHMARK_DTYPES", "float16")
    if dtypes:
        env["FLAGTENSOR_BENCHMARK_DTYPES"] = dtypes
    if max_shapes is not None:
        env["FLAGTENSOR_BENCHMARK_MAX_SHAPES"] = str(max_shapes)
    if warmup is not None:
        env["FLAGTENSOR_BENCHMARK_WARMUP"] = str(warmup)
    if repetitions is not None:
        env["FLAGTENSOR_BENCHMARK_REPETITIONS"] = str(repetitions)
    return env


def run_perf(op: str, results_dir: Path, env, suffix: str = "perf"):
    test_path = benchmark_test_path(op)
    if not test_path.exists():
        return {
            "status": "MISSING",
            "exit_code": 1,
            "log_path": None,
            "test_path": str(test_path),
            "benchmark_csv": None,
        }
    log_path = results_dir / op / f"{suffix}.log"
    cmd = f"pytest -vs {test_path.name}"
    code, output = run_cmd(cmd, cwd=test_path.parent, env=env)
    write_text(log_path, output)
    benchmark_csv = ROOT / "benchmark" / "results" / f"CUTENSOR_OP_{uppercase_op(op)}" / "benchmark.csv"
    copied_csv = None
    if benchmark_csv.exists():
        copied_csv = results_dir / op / f"{suffix}_benchmark.csv"
        ensure_dir(copied_csv.parent)
        shutil.copy2(benchmark_csv, copied_csv)
    return {
        "status": "PASS" if code == 0 else "FAIL",
        "exit_code": code,
        "log_path": str(log_path),
        "test_path": str(test_path),
        "benchmark_csv": str(copied_csv) if copied_csv else None,
    }


def clear_libtuner_cache():
    from flagtensor.utils import libcache

    db_path = Path(libcache.store.db_path)
    if db_path.exists():
        db_path.unlink()
    return str(db_path)


def run_libtuner_compare(op: str, results_dir: Path, base_env, smoke: bool, dtypes=None, max_shapes=None, warmup=None, repetitions=None):
    cache_path = clear_libtuner_cache()
    cold_env = smoke_env(base_env, smoke, dtypes=dtypes, max_shapes=max_shapes, warmup=warmup, repetitions=repetitions)
    warm_env = smoke_env(base_env, smoke, dtypes=dtypes, max_shapes=max_shapes, warmup=warmup, repetitions=repetitions)
    cold = run_perf(op, results_dir, cold_env, suffix="perf_cold")
    warm = run_perf(op, results_dir, warm_env, suffix="perf_warm")
    return {
        "cache_db": cache_path,
        "cold": cold,
        "warm": warm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", default=None)
    parser.add_argument("--op-list", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--run-correctness", action="store_true")
    parser.add_argument("--run-perf", action="store_true")
    parser.add_argument("--run-libtuner-compare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dtypes", default=None)
    parser.add_argument("--max-shapes", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    args = parser.parse_args()

    ops = load_ops(op=args.op, op_list=args.op_list)
    results_dir = Path(args.results_dir).resolve() if args.results_dir else ROOT / "ci_results"
    ensure_dir(results_dir)

    base_env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        base_env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    export_environment(results_dir)

    if not any([args.run_correctness, args.run_perf, args.run_libtuner_compare]):
        args.run_correctness = True
        args.run_perf = True

    summary = {
        "ops": ops,
        "results_dir": str(results_dir),
        "correctness": {},
        "performance": {},
        "libtuner_compare": {},
    }

    perf_env = smoke_env(
        base_env,
        args.smoke,
        dtypes=args.dtypes,
        max_shapes=args.max_shapes,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )

    for op in ops:
        ensure_dir(results_dir / op)
        if args.run_correctness:
            summary["correctness"][op] = run_correctness(op, results_dir, base_env)
        if args.run_perf:
            summary["performance"][op] = run_perf(op, results_dir, perf_env)
        if args.run_libtuner_compare:
            summary["libtuner_compare"][op] = run_libtuner_compare(
                op,
                results_dir,
                base_env,
                smoke=args.smoke,
                dtypes=args.dtypes,
                max_shapes=args.max_shapes,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_summary(summary, results_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
