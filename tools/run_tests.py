#!/usr/bin/env python3
"""Run accuracy and performance tests for FlagTensor operators.

Reads operator inventory from conf/operators.yaml, dispatches accuracy +
performance tests across multiple GPUs in parallel, and aggregates results.

When stdout is a TTY, a live display shows per-GPU progress with a progress bar.
When piped, falls back to plain line-by-line output.

Usage:
    python tools/run_tests.py                                    # all ops, GPU 0
    python tools/run_tests.py --stages stable --gpus 0,1         # stable ops, 2 GPUs
    python tools/run_tests.py --ops abs,exp,add --gpus 0         # specific ops
    python tools/run_tests.py --stages stable --gpus 0 --dump-output
"""
import argparse
import datetime
import json
import os
import queue as queue_module
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONF = ROOT / "conf" / "operators.yaml"

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

TIMEOUT_SIGNAL = -100

# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------
IS_TTY = sys.stdout.isatty()
RED = "\033[31m" if IS_TTY else ""
GREEN = "\033[32m" if IS_TTY else ""
YELLOW = "\033[93m" if IS_TTY else ""
CYAN = "\033[36m" if IS_TTY else ""
DIM = "\033[2m" if IS_TTY else ""
NC = "\033[0m" if IS_TTY else ""


def log(msg, **kwargs):
    print(f"{GREEN}[INFO]{NC} {msg}", flush=True, **kwargs)


def warn(msg, **kwargs):
    print(f"{YELLOW}[WARN]{NC} {msg}", flush=True, **kwargs)


def err(msg, **kwargs):
    print(f"{RED}[ERROR]{NC} {msg}", flush=True, **kwargs)


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Progress bar (TTY only)
# ---------------------------------------------------------------------------
def _progress_bar(done, total, width=40, color=""):
    if not total:
        return " " * width
    frac = done * width / total
    full = int(frac)
    bar = "█" * full + " " * (width - full)
    return f"{color}{bar}{NC}"


def _format_status(status, dur):
    mapping = {"Passed": (GREEN, "OK"), "Failed": (RED, "FAILED"),
               "Timeout": (RED, "TIMEOUT"), "Error": (RED, "ERROR"),
               "NotFound": (YELLOW, "NOTFOUND"), "Skipped": (YELLOW, "SKIPPED")}
    col, label = mapping.get(status, (YELLOW, status.upper()))
    return f"{col}[{label:<8} {dur:>6.1f}s]{NC}"


# ---------------------------------------------------------------------------
# LiveDisplay — pinned footer with GPU status + progress bar
# ---------------------------------------------------------------------------
class LiveDisplay:
    def __init__(self, gpu_ids, op_count):
        self.gpu_ids = gpu_ids
        self.op_count = op_count
        self.gpu_index = {gid: i + 1 for i, gid in enumerate(gpu_ids)}
        self.footer = [
            self._fmt_progress(0),
            *[f"{DIM}[GPU {gid:2d}] idle{NC}" for gid in gpu_ids],
        ]
        self.n = len(self.footer)
        self.footer_drawn = False
        self.per_gpu_done = {gid: 0 for gid in gpu_ids}

    def _fmt_progress(self, tests_done):
        total = self.op_count * 2  # accuracy + benchmark per op
        color = GREEN if tests_done >= total else CYAN
        bar_width = 40
        bar = _progress_bar(tests_done, total, bar_width, color=color)
        ops_done = tests_done // 2
        return f"[Progress] [{bar}]  {ops_done}/{self.op_count} ops"

    def _draw(self):
        if not IS_TTY:
            return
        for line in self.footer:
            sys.stdout.write(line + "\n")
        sys.stdout.flush()
        self.footer_drawn = True

    def _erase(self):
        if not IS_TTY or not self.footer_drawn:
            return
        for _ in range(self.n):
            sys.stdout.write("\033[A\033[2K")

    def init(self):
        if IS_TTY:
            self._draw()

    def log_line(self, msg):
        if IS_TTY:
            self._erase()
            sys.stdout.write(msg + "\n")
            self._draw()
        else:
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()

    def update_gpu(self, gpu_id, status_line):
        idx = self.gpu_index.get(gpu_id)
        if idx is None:
            return
        self.footer[idx] = status_line
        if IS_TTY:
            self._erase()
            self._draw()

    def update_progress(self, tests_done):
        self.footer[0] = self._fmt_progress(tests_done)
        if IS_TTY:
            self._erase()
            self._draw()
        else:
            sys.stdout.write(self.footer[0] + "\n")
            sys.stdout.flush()

    def finish(self):
        if IS_TTY:
            self._erase()
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# Operator discovery
# ---------------------------------------------------------------------------
def load_operators():
    with open(CONF) as f:
        return yaml.safe_load(f).get("operators", [])


def filter_operators(ops, selected_names=None, category=None, stages=None):
    """Filter operator list. Returns filtered list."""
    if selected_names:
        wanted = set(selected_names.split(","))
        ops = [o for o in ops if o["name"] in wanted]
    if category:
        ops = [o for o in ops if o["category"] == category]
    if stages and stages != "all":
        wanted_stages = set(stages.split(","))
        filtered = []
        for op_item in ops:
            stage_dicts = op_item.get("stages", [])
            op_stages = set()
            for s in stage_dicts:
                if isinstance(s, dict):
                    op_stages.update(s.keys())
            if not op_stages:
                op_stages = {op_item.get("status", "stable")}
            if op_stages & wanted_stages:
                filtered.append(op_item)
        ops = filtered
    return ops


# ---------------------------------------------------------------------------
# Test path discovery
# ---------------------------------------------------------------------------
def get_test_file(op):
    """Find the best accuracy test file for an operator.

    Returns: Path to test file, or category-level fallback.
    """
    cat = op.get("category", "unary")
    op_name = op.get("name", "")

    # New per-operator test file
    per_op = ROOT / "tests" / cat / f"test_{op_name}.py"
    if per_op.exists():
        return str(per_op)

    # Legacy category-level test
    legacy = ROOT / "tests" / cat / f"test_{cat}_correctness.py"
    if legacy.exists():
        return str(legacy)

    return str(ROOT / "tests")


def get_bench_file(op):
    """Find the benchmark test file for an operator."""
    cat = op.get("category", "unary")
    bf = ROOT / "benchmark" / f"test_{cat}_perf.py"
    if bf.exists():
        return str(bf)
    return str(ROOT / "benchmark")


# ---------------------------------------------------------------------------
# pytest invocation
# ---------------------------------------------------------------------------
def run_cmd(cmd, cwd=None, env=None, timeout=600, dump_output=False, out_dir=None, flavor=""):
    stdout = subprocess.DEVNULL
    stderr = subprocess.DEVNULL
    if dump_output and out_dir:
        ensure_dir(Path(out_dir))
        try:
            stdout = open(Path(out_dir) / f"{flavor}_stdout.log", "w")
            stderr = open(Path(out_dir) / f"{flavor}_stderr.log", "w")
        except Exception:
            pass

    p = subprocess.Popen(
        shlex.split(cmd), cwd=cwd or ROOT, env=env,
        stdout=stdout, stderr=stderr, start_new_session=True,
    )
    try:
        p.wait(timeout=timeout)
        return p.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            p.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        return TIMEOUT_SIGNAL
    finally:
        if dump_output and isinstance(stdout, type(open(os.devnull))):
            stdout.close()
        if dump_output and isinstance(stderr, type(open(os.devnull))):
            stderr.close()


def parse_pytest_summary(text):
    clean = ANSI_RE.sub("", text)
    counts = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0, "errors": 0}
    for m in re.finditer(r"(\d+)\s+([A-Za-z_]+)", clean):
        key = m.group(2).lower()
        if key in counts:
            counts[key] = int(m.group(1))
    counts["total"] = counts["passed"] + counts["failed"] + counts["skipped"]
    return counts


def _get_env(gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    return env


# ---------------------------------------------------------------------------
# Accuracy test (single op, single GPU)
# ---------------------------------------------------------------------------
def run_accuracy(op, gpu_id, dump_output=False):
    marker = op.get("correctness_mark", op["name"])
    test_file = get_test_file(op)
    cmd = f"{sys.executable} -m pytest -m '{marker}' -v --tb=short {test_file}"

    out_dir = None
    if dump_output:
        out_dir = ROOT / "results" / op["name"]

    dur = time.time()
    # Capture stderr/stdout for parsing (even if dump_output also writes to files)
    p = subprocess.Popen(
        shlex.split(cmd), cwd=ROOT, env=_get_env(gpu_id),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        stdout, stderr = p.communicate(timeout=600)
        code = p.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            stdout, stderr = p.communicate(timeout=10)
            code = TIMEOUT_SIGNAL
        except Exception:
            code = TIMEOUT_SIGNAL
            stdout, stderr = "", ""
    dur = time.time() - dur

    output = stdout + "\n" + stderr
    summary = parse_pytest_summary(output)

    if code == TIMEOUT_SIGNAL:
        status = "Timeout"
    elif code == 0 and summary["failed"] == 0 and summary["errors"] == 0:
        status = "Passed"
    else:
        status = "Failed"

    result = {**summary, "status": status, "duration": dur, "exit_code": code}

    if dump_output and out_dir:
        ensure_dir(Path(out_dir))
        (Path(out_dir) / "accuracy.log").write_text(output)

    return result


# ---------------------------------------------------------------------------
# Benchmark test (single op, single GPU)
# ---------------------------------------------------------------------------
def run_benchmark(op, gpu_id, dump_output=False):
    marker = op.get("benchmark_mark", op["name"])
    bench_file = get_bench_file(op)

    # Use new per-op test if available
    cat = op.get("category", "unary")
    per_op_bench = ROOT / "benchmark" / f"test_{op['name']}_perf.py"
    if not per_op_bench.exists() and cat in ("unary", "binary"):
        # Legacy benchmark files are loaded via benchmark/test_{cat}_perf.py
        cmd = (
            f"{sys.executable} -m pytest -m '{marker}' "
            f"--record json --output benchmark_result.json "
            f"-v --tb=short {bench_file}"
        )
    else:
        cmd = (
            f"{sys.executable} -m pytest -m '{marker}' -v --tb=short {bench_file}"
        )

    env = _get_env(gpu_id)
    out_dir = None
    if dump_output:
        out_dir = ROOT / "results" / op["name"]

    dur = time.time()
    p = subprocess.Popen(
        shlex.split(cmd), cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        stdout, stderr = p.communicate(timeout=1200)
        code = p.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            stdout, stderr = p.communicate(timeout=10)
            code = TIMEOUT_SIGNAL
        except Exception:
            code = TIMEOUT_SIGNAL
            stdout, stderr = "", ""
    dur = time.time() - dur

    output = stdout + "\n" + stderr

    # Try to parse benchmark_result.json for speedup data
    speedup = 0.0
    result_json_path = ROOT / "benchmark" / "benchmark_result.json"
    bench_data = {}
    if result_json_path.exists():
        try:
            bench_data = json.loads(result_json_path.read_text())
            speeds = _extract_speedups(bench_data, op["name"])
            speedup = sum(speeds) / len(speeds) if speeds else 0.0
            result_json_path.unlink()  # clean up
        except (json.JSONDecodeError, ValueError):
            pass

    if code == TIMEOUT_SIGNAL:
        status = "Timeout"
    elif code == 0:
        status = "Passed"
    else:
        status = "Failed"

    result = {"status": status, "duration": dur, "exit_code": code, "speedup": speedup, "data": bench_data}

    if dump_output and out_dir:
        ensure_dir(Path(out_dir))
        (Path(out_dir) / "perf.log").write_text(output)

    return result


def _extract_speedups(data, op_name):
    """Recursively extract speedup values from benchmark JSON."""
    speeds = []
    if isinstance(data, dict):
        for k, v in data.items():
            if k == "speedup" and isinstance(v, (int, float)):
                speeds.append(float(v))
            elif isinstance(v, (dict, list)):
                speeds.extend(_extract_speedups(v, op_name))
    elif isinstance(data, list):
        for item in data:
            speeds.extend(_extract_speedups(item, op_name))
    return speeds


# ---------------------------------------------------------------------------
# Worker process — one per GPU
# ---------------------------------------------------------------------------
def worker_proc(gpu_id, work_queue, display_queue, dump_output, output_dir):
    """Worker process — one per GPU. Pulls ops from queue, runs accuracy + benchmark."""
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    worker_result = {}
    while True:
        try:
            op = work_queue.get_nowait()
        except queue_module.Empty:
            break
        op = op.strip()
        if not op:
            continue

        op_data = {"name": op}
        for o in load_operators():
            if o["name"] == op:
                op_data = o
                break

        # --- Accuracy ---
        display_queue.put(("start", gpu_id, "accuracy", op))
        acc = run_accuracy(op_data, gpu_id, dump_output=dump_output)
        display_queue.put(("done", gpu_id, "accuracy", op, acc.get("status", "Error"), acc.get("duration", 0)))

        worker_result[op] = {"accuracy": acc, "performance": None}
        json_path = output_dir / f"summary{gpu_id}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(worker_result, indent=2))

        # --- Benchmark ---
        display_queue.put(("start", gpu_id, "benchmark", op))
        perf = run_benchmark(op_data, gpu_id, dump_output=dump_output)
        display_queue.put(("done", gpu_id, "benchmark", op, perf.get("status", "Error"), perf.get("duration", 0)))

        worker_result[op] = {"accuracy": acc, "performance": perf}
        json_path.write_text(json.dumps(worker_result, indent=2))

    display_queue.put(("exit", gpu_id))


# ---------------------------------------------------------------------------
# Display loop — reads from display_queue, updates LiveDisplay
# ---------------------------------------------------------------------------
def display_loop(display_queue, display, n_workers):
    exited = 0
    tests_done = 0
    per_gpu_done = {gid: 0 for gid in display.gpu_ids}

    while exited < n_workers:
        try:
            msg = display_queue.get(timeout=1)
        except Exception:
            continue

        kind = msg[0]
        if kind == "exit":
            gpu_id = msg[1]
            n = per_gpu_done.get(gpu_id, 0)
            display.update_gpu(gpu_id, f"{DIM}[GPU {gpu_id:2d}] done ({n} ops){NC}")
            exited += 1
        elif kind == "start":
            _, gpu_id, phase, op = msg
            label = "accuracy" if phase == "accuracy" else "benchmark"
            op_col = op.ljust(24)
            n = per_gpu_done.get(gpu_id, 0)
            if IS_TTY:
                display.update_gpu(gpu_id, f"[GPU {gpu_id:2d}] ({n:>3} done)  {label}  {op_col}")
            else:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                display.log_line(f"{GREEN}[INFO]{NC} [{ts}][GPU {gpu_id:2d}] {label}  {op_col} ...")
        elif kind == "done":
            _, gpu_id, phase, op, status, dur = msg
            tests_done += 1
            if phase == "benchmark":
                per_gpu_done[gpu_id] = per_gpu_done.get(gpu_id, 0) + 1
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            label = "acc" if phase == "accuracy" else "perf"
            op_col = op.ljust(24)
            status_str = _format_status(status, dur)
            log_line = f"{GREEN}[INFO]{NC} [{ts}][GPU {gpu_id:2d}] {label}  {op_col} {status_str}"
            if not IS_TTY:
                ops_done = tests_done // 2
                total = display.op_count
                log_line += f"  ({ops_done}/{total} ops)"
            # Update progress before logging so footer shows latest state
            display.footer[0] = display._fmt_progress(tests_done)
            display.log_line(log_line)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
def cleanup():
    # Remove leftover benchmark JSONs
    for f in ROOT.glob("benchmark/benchmark_result.json"):
        try:
            f.unlink()
        except OSError:
            pass
    for f in ROOT.glob("benchmark/result-*.log"):
        try:
            f.unlink()
        except OSError:
            pass


def terminate_workers(workers):
    for p in workers:
        if p.is_alive():
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except OSError:
                pass
    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.kill()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FlagTensor multi-GPU test runner")
    parser.add_argument("--ops", help="Comma-separated operator names")
    parser.add_argument("--category", help="Operator category (unary/binary/contraction/sparse)")
    parser.add_argument("--stages", default=None,
                        help="Operator stages (alpha/beta/stable/experimental/active). Use 'all' for all stages.")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--output-dir", default="results", help="Output directory for test results")
    parser.add_argument("--dump-output", action="store_true", help="Save per-op stdout/stderr to disk")
    parser.add_argument("--color", choices=["auto", "always", "never"], default="auto",
                        help="ANSI color output mode")
    args = parser.parse_args()

    # Apply color mode
    global IS_TTY, RED, GREEN, YELLOW, CYAN, DIM, NC
    if args.color == "always":
        IS_TTY = True
        RED, GREEN, YELLOW, CYAN, DIM, NC = "\033[31m", "\033[32m", "\033[93m", "\033[36m", "\033[2m", "\033[0m"
    elif args.color == "never":
        IS_TTY = False
        RED = GREEN = YELLOW = CYAN = DIM = NC = ""

    # Load and filter operators
    ops = load_operators()
    ops = filter_operators(ops, selected_names=args.ops, category=args.category, stages=args.stages)

    if not ops:
        err("No operators matched. Check your --ops, --category, or --stages filters.")
        sys.exit(1)

    log(f"Testing {len(ops)} operators ...")

    # Parse GPUs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not gpu_ids:
        err("No GPUs specified.")
        sys.exit(1)

    # Setup output
    output_dir = ROOT / args.output_dir
    ensure_dir(output_dir)

    # Build work queue
    work_queue = Queue()
    for op_item in ops:
        work_queue.put(op_item["name"])

    display_queue = Queue()
    display = LiveDisplay(gpu_ids, len(ops))

    # Handle Ctrl+C
    workers = []

    def on_interrupt(signum, frame):
        warn("Interrupted. Cleaning up ...")
        terminate_workers(workers)
        cleanup()
        display.finish()
        sys.exit(1)

    signal.signal(signal.SIGINT, on_interrupt)
    signal.signal(signal.SIGTERM, on_interrupt)

    # Launch workers
    for gpu in gpu_ids:
        p = Process(target=worker_proc, args=(gpu, work_queue, display_queue, args.dump_output, output_dir))
        p.start()
        workers.append(p)

    display.init()
    display_loop(display_queue, display, len(gpu_ids))

    for p in workers:
        p.join()

    display.finish()

    # --- Merge per-GPU summaries ---
    all_results = {}
    env_info = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gpus": gpu_ids,
        "operators_tested": len(ops),
    }

    for gpu_id in gpu_ids:
        gpu_file = output_dir / f"summary{gpu_id}.json"
        if gpu_file.exists():
            try:
                gpu_data = json.loads(gpu_file.read_text())
                all_results.update(gpu_data)
            except (json.JSONDecodeError, ValueError):
                err(f"GPU {gpu_id} summary is invalid JSON, skipping.")

    final = {"env": env_info, "result": all_results}
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(final, indent=2))

    # Print final tally
    acc_pass = sum(1 for v in all_results.values() if v.get("accuracy", {}).get("status") == "Passed")
    perf_pass = sum(1 for v in all_results.values() if v.get("performance", {}).get("status") == "Passed")
    log(f"Accuracy:  {acc_pass}/{len(ops)} passed")
    log(f"Performance: {perf_pass}/{len(ops)} passed")
    log(f"Results: {summary_path}")

    cleanup()


if __name__ == "__main__":
    main()
