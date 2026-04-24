#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    """Line-buffered progress line for CI debugging (child output may still be buffered)."""
    print(f"[flagtensor-ci] {msg}", flush=True)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_utils import build_env_payload


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd, cwd=None, env=None, stream_to_terminal=False, label=""):
    """
    Run a shell command. By default stdout/stderr are captured only (no terminal output until done).
    With stream_to_terminal=True, each line is printed immediately while still collecting full output.
    """
    prefix = f"[{label}] " if label else ""
    log(f"starting subprocess cwd={cwd!s} cmd={cmd!r}")
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
                print(f"{prefix}{line}", end="", flush=True)
        else:
            chunks.append(process.stdout.read() or "")
    process.wait()
    elapsed = time.monotonic() - t0
    out = "".join(chunks)
    log(f"subprocess finished exit={process.returncode} elapsed_s={elapsed:.2f} bytes={len(out)}")
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


def load_ops(op=None, op_list=None):
    if op_list:
        lines = Path(op_list).read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if op:
        return [op]
    return discover_ops()


def filter_ops(ops, exclude_ops=None):
    excluded = {item.strip().lower() for item in (exclude_ops or []) if item and item.strip()}
    return [op for op in ops if op.lower() not in excluded]


def discover_ops():
    pattern = re.compile(r"test_CUTENSOR_OP_(?P<name>[A-Z0-9_]+)(?:_perf)?\.py$")
    ops = set()
    for directory in (ROOT / "ctests", ROOT / "benchmark"):
        if not directory.exists():
            continue
        for path in directory.glob("test_CUTENSOR_OP_*.py"):
            match = pattern.match(path.name)
            if not match:
                continue
            ops.add(match.group("name").lower())
    return sorted(ops)


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
    log(f"done. artifacts: {rd}/summary.json, {rd}/summary.md")
    if summary.get("correctness"):
        log(f"correctness: {_status_histogram(summary['correctness'])}")
        bad = _ops_not_pass(summary["correctness"])
        if bad:
            log(f"correctness not PASS ({len(bad)}): {', '.join(bad)}")
    if summary.get("performance"):
        log(f"perf: {_status_histogram(summary['performance'])}")
        bad = _ops_not_pass(summary["performance"])
        if bad:
            log(f"perf not PASS ({len(bad)}): {', '.join(bad)}")
    if summary.get("libtuner_compare"):
        cold = {op: v["cold"] for op, v in summary["libtuner_compare"].items()}
        warm = {op: v["warm"] for op, v in summary["libtuner_compare"].items()}
        log(f"libtuner cold: {_status_histogram(cold)} | warm: {_status_histogram(warm)}")
        bad_c = _ops_not_pass(cold)
        bad_w = _ops_not_pass(warm)
        if bad_c or bad_w:
            log(f"libtuner not PASS — cold ({len(bad_c)}): {', '.join(bad_c)} | warm ({len(bad_w)}): {', '.join(bad_w)}")


def run_correctness(op: str, results_dir: Path, env, stream_subprocess: bool = False):
    test_path = correctness_test_path(op)
    if not test_path.exists():
        log(f"correctness skip {op}: missing {test_path}")
        return {
            "status": "MISSING",
            "exit_code": 1,
            "log_path": None,
            "test_path": str(test_path),
        }
    log_path = results_dir / op / "correctness.log"
    cmd = f"{sys.executable} -m pytest -vs {test_path.name}"
    code, output = run_cmd(
        cmd,
        cwd=test_path.parent,
        env=env,
        stream_to_terminal=stream_subprocess,
        label=f"correctness:{op}",
    )
    write_text(log_path, output)
    return {
        "status": "PASS" if code == 0 else "FAIL",
        "exit_code": code,
        "log_path": str(log_path),
        "test_path": str(test_path),
    }


def smoke_env(base_env, smoke: bool, mode: str = "kernel", dtypes=None, max_shapes=None, warmup=None, repetitions=None):
    env = dict(base_env)
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


def run_perf(op: str, results_dir: Path, env, suffix: str = "perf", stream_subprocess: bool = False):
    test_path = benchmark_test_path(op)
    if not test_path.exists():
        log(f"perf skip {op} ({suffix}): missing {test_path}")
        return {
            "status": "MISSING",
            "exit_code": 1,
            "log_path": None,
            "test_path": str(test_path),
            "benchmark_csv": None,
        }
    log_path = results_dir / op / f"{suffix}.log"
    cmd = f"{sys.executable} -m pytest -vs {test_path.name}"
    code, output = run_cmd(
        cmd,
        cwd=test_path.parent,
        env=env,
        stream_to_terminal=stream_subprocess,
        label=f"perf:{op}:{suffix}",
    )
    write_text(log_path, output)
    benchmark_dir = ROOT / "benchmark" / "results" / f"CUTENSOR_OP_{uppercase_op(op)}"
    benchmark_csv = benchmark_dir / "benchmark_kernel.csv"
    if not benchmark_csv.exists():
        benchmark_csv = benchmark_dir / "benchmark.csv"
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
    log("clear_libtuner_cache: importing flagtensor.utils.libcache")
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
):
    log(f"libtuner_compare {op}: clearing cache")
    cache_path = clear_libtuner_cache()
    cold_env = smoke_env(base_env, smoke, mode=mode, dtypes=dtypes, max_shapes=max_shapes, warmup=warmup, repetitions=repetitions)
    warm_env = smoke_env(base_env, smoke, mode=mode, dtypes=dtypes, max_shapes=max_shapes, warmup=warmup, repetitions=repetitions)
    log(f"libtuner_compare {op}: cold run")
    cold = run_perf(op, results_dir, cold_env, suffix="perf_cold", stream_subprocess=stream_subprocess)
    log(f"libtuner_compare {op}: warm run")
    warm = run_perf(op, results_dir, warm_env, suffix="perf_warm", stream_subprocess=stream_subprocess)
    return {
        "cache_db": cache_path,
        "cold": cold,
        "warm": warm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", default=None)
    parser.add_argument("--op-list", default=None)
    parser.add_argument("--exclude-op", action="append", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--run-correctness", action="store_true")
    parser.add_argument("--run-perf", action="store_true")
    parser.add_argument("--run-libtuner-compare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--mode", choices=["kernel", "operator", "wrapper"], default="kernel")
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

    ops = filter_ops(load_ops(op=args.op, op_list=args.op_list), exclude_ops=args.exclude_op)
    results_dir = Path(args.results_dir).resolve() if args.results_dir else ROOT / "ci_results"
    ensure_dir(results_dir)

    log(f"root={ROOT}")
    log(f"results_dir={results_dir}")
    log(f"ops count={len(ops)}: {ops[:5]}{' ...' if len(ops) > 5 else ''}")
    log(f"flags: run_correctness={args.run_correctness} run_perf={args.run_perf} run_libtuner_compare={args.run_libtuner_compare} smoke={args.smoke} verbose={args.verbose}")

    base_env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        base_env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        log(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices!r}")

    log("writing env.json")
    export_environment(results_dir)

    if not any([args.run_correctness, args.run_perf, args.run_libtuner_compare]):
        args.run_correctness = True
        args.run_perf = True
        log("no run_* flags set; defaulting to correctness + perf")

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
        mode=args.mode,
        dtypes=args.dtypes,
        max_shapes=args.max_shapes,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )

    for i, op in enumerate(ops, start=1):
        log(f"--- operator {i}/{len(ops)}: {op!r} ---")
        ensure_dir(results_dir / op)
        if args.run_correctness:
            log(f"{op}: correctness starting")
            summary["correctness"][op] = run_correctness(op, results_dir, base_env, stream_subprocess=stream)
            st = summary["correctness"][op].get("status")
            log(f"{op}: correctness done status={st}")
        if args.run_perf:
            log(f"{op}: perf starting")
            summary["performance"][op] = run_perf(op, results_dir, perf_env, stream_subprocess=stream)
            st = summary["performance"][op].get("status")
            log(f"{op}: perf done status={st}")
        if args.run_libtuner_compare:
            summary["libtuner_compare"][op] = run_libtuner_compare(
                op,
                results_dir,
                base_env,
                smoke=args.smoke,
                mode=args.mode,
                dtypes=args.dtypes,
                max_shapes=args.max_shapes,
                warmup=args.warmup,
                repetitions=args.repetitions,
                stream_subprocess=stream,
            )
            c = summary["libtuner_compare"][op]["cold"].get("status")
            w = summary["libtuner_compare"][op]["warm"].get("status")
            log(f"{op}: libtuner_compare done cold={c} warm={w}")

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_summary(summary, results_dir)
    log_run_summary(summary)
    if args.dump_json_summary:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
