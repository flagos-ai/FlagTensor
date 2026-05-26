#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


logger = logging.getLogger("flagtensor.ci")
logging.basicConfig(
    level=logging.INFO,
    format="[flagtensor-ci] %(asctime)s %(levelname)s %(message)s",
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
from flagtensor_registry import get_operator_map
from flagtensor_registry import load_operator_registry


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd, cwd=None, env=None, stream_to_terminal=False, label=""):
    """
    Run a shell command. By default stdout/stderr are captured only (no terminal output until done).
    With stream_to_terminal=True, each line is printed immediately while still collecting full output.
    """
    prefix = f"[{label}] " if label else ""
    logger.info(f"starting subprocess cwd={cwd!s} cmd={cmd!r}")
    t0 = time.monotonic()
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    chunks = []
    if process.stdout is not None:
        if stream_to_terminal:
            for line in process.stdout:
                chunks.append(line)
                sys.stdout.write(f"{prefix}{line}")
        else:
            chunks.append(process.stdout.read() or "")
    process.wait()
    elapsed = time.monotonic() - t0
    out = "".join(chunks)
    logger.info(f"subprocess finished exit={process.returncode} elapsed_s={elapsed:.2f} bytes={len(out)}")
    return process.returncode, out or ""


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_speedup_stats(csv_path: Path):
    if not csv_path.exists():
        return {"avg_speedup": None, "max_speedup": None, "row_count": 0}
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    speedups = []
    for row in rows:
        value = row.get("speedup")
        if value in (None, ""):
            continue
        speedups.append(float(value))
    return {
        "avg_speedup": statistics.mean(speedups) if speedups else None,
        "max_speedup": max(speedups) if speedups else None,
        "row_count": len(speedups),
    }


def load_ops(op=None, op_list=None, category=None, mode=None, exclude_ops=None, include_blocked=False, registry_path=None):
    specs = load_operator_registry(registry_path=registry_path)
    requested_ops = []
    if op_list:
        lines = Path(op_list).read_text(encoding="utf-8").splitlines()
        requested_ops = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if op:
        requested_ops.append(op)
    filtered = filter_operator_specs(
        specs,
        names=requested_ops or None,
        exclude_names=exclude_ops,
        categories=category,
        mode=mode,
        include_blocked=include_blocked,
    )
    return [spec.name for spec in filtered]


def filter_ops(ops, exclude_ops=None):
    excluded = {item.strip().lower() for item in (exclude_ops or []) if item and item.strip()}
    return [op for op in ops if op.lower() not in excluded]


def discover_ops():
    return [spec.name for spec in load_operator_registry()]


def uppercase_op(op: str) -> str:
    return op.upper()


def correctness_suite_path() -> Path:
    return ROOT / "tests"


def correctness_test_path(op: str) -> Path:
    return ROOT / "ctests" / f"test_CUTENSOR_OP_{uppercase_op(op)}.py"


def benchmark_test_path(op: str) -> Path:
    return ROOT / "benchmark" / f"test_CUTENSOR_OP_{uppercase_op(op)}_perf.py"


def benchmark_suite_path() -> Path:
    return ROOT / "benchmark"


def benchmark_suite_file_for_op(op: str, registry_map: dict) -> Path:
    """Map operator to its category-level benchmark file.

    Returns None if the operator or its category is not recognized.
    """
    _CATEGORY_FILE_MAP = {
        "unary": "test_unary_perf.py",
        "binary": "test_binary_perf.py",
        "contraction": "test_contraction_perf.py",
        "sparse": "test_sparse_perf.py",
    }
    spec = registry_map.get(op)
    if spec is None:
        return None
    filename = _CATEGORY_FILE_MAP.get(spec.category)
    if filename is None:
        return None
    return benchmark_suite_path() / filename


def benchmark_csv_path(op: str, mode: str) -> Path:
    benchmark_dir = ROOT / "benchmark" / "results" / f"CUTENSOR_OP_{uppercase_op(op)}"
    mode_csv = benchmark_dir / f"benchmark_{mode}.csv"
    if mode_csv.exists():
        return mode_csv
    fallback_csv = benchmark_dir / "benchmark.csv"
    if fallback_csv.exists():
        return fallback_csv
    return mode_csv


def export_environment(results_dir: Path):
    payload = build_env_payload(project_root=ROOT)
    output = results_dir / "env.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def write_markdown_summary(summary, results_dir: Path):
    lines = [
        "# FlagTensor CI Summary",
        "",
        "| operator | correctness | perf | avg_speedup | max_speedup | libtuner_cold | libtuner_warm |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for op in summary["ops"]:
        correctness = summary["correctness"].get(op, {}).get("status", "N/A")
        performance_info = summary["performance"].get(op, {})
        performance = performance_info.get("status", "N/A")
        avg_speedup = performance_info.get("avg_speedup")
        max_speedup = performance_info.get("max_speedup")
        libtuner = summary["libtuner_compare"].get(op, {})
        cold = libtuner.get("cold", {}).get("status", "N/A")
        warm = libtuner.get("warm", {}).get("status", "N/A")
        avg_text = f"{avg_speedup:.6f}" if avg_speedup is not None else "N/A"
        max_text = f"{max_speedup:.6f}" if max_speedup is not None else "N/A"
        lines.append(f"| {op} | {correctness} | {performance} | {avg_text} | {max_text} | {cold} | {warm} |")
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _status_histogram(block: dict) -> str:
    counts = {}
    for info in block.values():
        st = info.get("status", "?")
        counts[st] = counts.get(st, 0) + 1
    return " ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def _ops_not_pass(block: dict) -> list:
    return [op for op, info in block.items() if info.get("status") != "PASS"]


def log_run_summary(summary: dict) -> None:
    rd = summary["results_dir"]
    logger.info(f"done. artifacts: {rd}/summary.json, {rd}/summary.md")
    if summary.get("correctness"):
        logger.info(f"correctness: {_status_histogram(summary['correctness'])}")
        bad = _ops_not_pass(summary["correctness"])
        if bad:
            logger.info(f"correctness not PASS ({len(bad)}): {', '.join(bad)}")
    if summary.get("performance"):
        logger.info(f"perf: {_status_histogram(summary['performance'])}")
        bad = _ops_not_pass(summary["performance"])
        if bad:
            logger.info(f"perf not PASS ({len(bad)}): {', '.join(bad)}")
    if summary.get("libtuner_compare"):
        cold = {op: v["cold"] for op, v in summary["libtuner_compare"].items()}
        warm = {op: v["warm"] for op, v in summary["libtuner_compare"].items()}
        logger.info(f"libtuner cold: {_status_histogram(cold)} | warm: {_status_histogram(warm)}")
        bad_c = _ops_not_pass(cold)
        bad_w = _ops_not_pass(warm)
        if bad_c or bad_w:
            logger.info(f"libtuner not PASS — cold ({len(bad_c)}): {', '.join(bad_c)} | warm ({len(bad_w)}): {', '.join(bad_w)}")


def run_correctness(op: str, results_dir: Path, env, stream_subprocess: bool = False):
    suite_path = correctness_suite_path()
    test_path = correctness_test_path(op)
    if suite_path.exists():
        target_path = suite_path
        cmd = f'{sys.executable} -m pytest -vs {suite_path.name} -m "{op}"'
    elif test_path.exists():
        target_path = test_path
        cmd = f"{sys.executable} -m pytest -vs {test_path.name}"
    else:
        logger.info(f"correctness skip {op}: missing {suite_path} and {test_path}")
        return {
            "status": "MISSING",
            "exit_code": 1,
            "log_path": None,
            "test_path": str(test_path),
        }
    log_path = results_dir / op / "correctness.log"
    code, output = run_cmd(
        cmd,
        cwd=target_path.parent,
        env=env,
        stream_to_terminal=stream_subprocess,
        label=f"correctness:{op}",
    )
    write_text(log_path, output)
    return {
        "status": "PASS" if code == 0 else "FAIL",
        "exit_code": code,
        "log_path": str(log_path),
        "test_path": str(target_path),
    }


def run_perf(op: str, results_dir: Path, env, suffix: str = "perf", stream_subprocess: bool = False, registry_map: dict = None):
    # Prefer category-level benchmark suite with -m <op> selection.
    # Fall back to legacy per-op benchmark file for debugging.
    suite_path = benchmark_suite_path()
    suite_file = benchmark_suite_file_for_op(op, registry_map or {})
    if suite_file is not None and suite_file.exists():
        target_path = suite_path
        test_path = suite_file
        test_filename = suite_file.name
        cmd = f'{sys.executable} -m pytest -vs {test_filename} -m "{op}"'
    else:
        test_path = benchmark_test_path(op)
        if not test_path.exists():
            logger.info(f"perf skip {op} ({suffix}): missing {suite_file or test_path}")
            return {
                "status": "MISSING",
                "exit_code": 1,
                "log_path": None,
                "test_path": str(suite_file) if suite_file else str(test_path),
                "benchmark_csv": None,
            }
        target_path = test_path.parent
        test_filename = test_path.name
        cmd = f"{sys.executable} -m pytest -vs {test_filename}"
    log_path = results_dir / op / f"{suffix}.log"
    code, output = run_cmd(
        cmd,
        cwd=target_path,
        env=env,
        stream_to_terminal=stream_subprocess,
        label=f"perf:{op}:{suffix}",
    )
    write_text(log_path, output)
    benchmark_mode = env.get("FLAGTENSOR_BENCHMARK_MODE", "kernel")
    benchmark_csv = benchmark_csv_path(op, benchmark_mode)
    copied_csv = None
    speedup_stats = {"avg_speedup": None, "max_speedup": None, "row_count": 0}
    if benchmark_csv.exists():
        copied_csv = results_dir / op / f"{suffix}_benchmark.csv"
        ensure_dir(copied_csv.parent)
        shutil.copy2(benchmark_csv, copied_csv)
        speedup_stats = load_speedup_stats(benchmark_csv)
    return {
        "status": "PASS" if code == 0 else "FAIL",
        "exit_code": code,
        "log_path": str(log_path),
        "test_path": str(test_path),
        "benchmark_csv": str(copied_csv) if copied_csv else None,
        "avg_speedup": speedup_stats["avg_speedup"],
        "max_speedup": speedup_stats["max_speedup"],
        "row_count": speedup_stats["row_count"],
    }


def clear_libtuner_cache():
    logger.info("clear_libtuner_cache: importing flagtensor.utils.libcache")
    from flagtensor.utils import libcache

    db_path = Path(libcache.store.db_path)
    if db_path.exists():
        db_path.unlink()
    return str(db_path)


def run_libtuner_compare(
    op: str,
    results_dir: Path,
    base_env,
    smoke: bool,
    mode: str = "kernel",
    dtypes=None,
    max_shapes=None,
    warmup=None,
    repetitions=None,
    stream_subprocess: bool = False,
    registry_map: dict = None,
):
    logger.info(f"libtuner_compare {op}: clearing cache")
    cache_path = clear_libtuner_cache()
    cold_env = smoke_env(base_env, smoke, mode=mode, dtypes=dtypes, max_shapes=max_shapes, warmup=warmup, repetitions=repetitions)
    warm_env = smoke_env(base_env, smoke, mode=mode, dtypes=dtypes, max_shapes=max_shapes, warmup=warmup, repetitions=repetitions)
    logger.info(f"libtuner_compare {op}: cold run")
    cold = run_perf(op, results_dir, cold_env, suffix="perf_cold", stream_subprocess=stream_subprocess, registry_map=registry_map)
    logger.info(f"libtuner_compare {op}: warm run")
    warm = run_perf(op, results_dir, warm_env, suffix="perf_warm", stream_subprocess=stream_subprocess, registry_map=registry_map)
    return {
        "cache_db": cache_path,
        "cold": cold,
        "warm": warm,
    }


def smoke_env(base_env, smoke: bool, mode: str = "kernel", dtypes=None, max_shapes=None, warmup=None, repetitions=None):
    env = dict(base_env)
    if mode is not None:
        env["FLAGTENSOR_BENCHMARK_MODE"] = mode
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", default=None)
    parser.add_argument("--op-list", default=None)
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument("--exclude-op", action="append", default=None)
    parser.add_argument("--include-blocked", action="store_true")
    parser.add_argument("--registry", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--run-correctness", action="store_true")
    parser.add_argument("--run-perf", action="store_true")
    parser.add_argument("--run-libtuner-compare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--mode", choices=["kernel", "operator", "wrapper"], default=None)
    parser.add_argument("--dtypes", default=None)
    parser.add_argument("--max-shapes", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="stream pytest stdout/stderr to this terminal in real time (still written to per-op log files)",
    )
    parser.add_argument(
        "--dump-json-summary",
        action="store_true",
        help="after the run, print the full summary object as JSON to stdout (default is a short text summary only)",
    )
    args = parser.parse_args()

    registry_map = get_operator_map(registry_path=args.registry)
    ops = load_ops(
        op=args.op,
        op_list=args.op_list,
        category=args.category,
        mode=args.mode,
        exclude_ops=args.exclude_op,
        include_blocked=args.include_blocked,
        registry_path=args.registry,
    )
    results_dir = Path(args.results_dir).resolve() if args.results_dir else ROOT / "ci_results"
    ensure_dir(results_dir)

    logger.info(f"root={ROOT}")
    logger.info(f"results_dir={results_dir}")
    logger.info(f"ops count={len(ops)}: {ops[:5]}{' ...' if len(ops) > 5 else ''}")
    logger.info(f"flags: run_correctness={args.run_correctness} run_perf={args.run_perf} run_libtuner_compare={args.run_libtuner_compare} smoke={args.smoke} verbose={args.verbose}")

    base_env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        base_env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        logger.info(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices!r}")

    logger.info("writing env.json")
    export_environment(results_dir)

    if not any([args.run_correctness, args.run_perf, args.run_libtuner_compare]):
        args.run_correctness = True
        args.run_perf = True
        logger.info("no run_* flags set; defaulting to correctness + perf")

    stream = args.verbose
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
        mode=args.mode or "kernel",
        dtypes=args.dtypes,
        max_shapes=args.max_shapes,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )

    for i, op in enumerate(ops, start=1):
        logger.info(f"--- operator {i}/{len(ops)}: {op!r} ---")
        ensure_dir(results_dir / op)
        spec = registry_map.get(op)
        if spec and spec.is_blocked:
            logger.info(f"{op}: blocked in registry ({spec.skip_reason or 'no reason provided'})")
        if args.run_correctness:
            logger.info(f"{op}: correctness starting")
            summary["correctness"][op] = run_correctness(op, results_dir, base_env, stream_subprocess=stream)
            st = summary["correctness"][op].get("status")
            logger.info(f"{op}: correctness done status={st}")
        if args.run_perf:
            logger.info(f"{op}: perf starting")
            summary["performance"][op] = run_perf(op, results_dir, perf_env, stream_subprocess=stream, registry_map=registry_map)
            st = summary["performance"][op].get("status")
            logger.info(f"{op}: perf done status={st}")
        if args.run_libtuner_compare:
            summary["libtuner_compare"][op] = run_libtuner_compare(
                op,
                results_dir,
                base_env,
                smoke=args.smoke,
                mode=args.mode or "kernel",
                dtypes=args.dtypes,
                max_shapes=args.max_shapes,
                warmup=args.warmup,
                repetitions=args.repetitions,
                stream_subprocess=stream,
                registry_map=registry_map,
            )
            c = summary["libtuner_compare"][op]["cold"].get("status")
            w = summary["libtuner_compare"][op]["warm"].get("status")
            logger.info(f"{op}: libtuner_compare done cold={c} warm={w}")

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_summary(summary, results_dir)
    log_run_summary(summary)
    if args.dump_json_summary:
        sys.stdout.write(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
