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

# -*- coding: utf-8 -*-
"""
run_tests.py — Run accuracy and performance tests for FlagTensor operators.

When stdout is a TTY, shows a live display: completed results scroll upward
while a pinned footer shows one status line per GPU. When output is piped or
redirected, falls back to plain line-by-line output with no ANSI codes.
"""
import argparse
import datetime
import json
import os
import platform
import queue as queue_module
import shlex
import shutil
import signal
import subprocess
import sys
import time
import types
from importlib import metadata
from multiprocessing import Process, Queue
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# Make ``src/`` importable so ``import flagtensor`` / ``import flagtensor_registry``
# work when running this script from any cwd (the project is a src-layout
# package). This mirrors what benchmark/conftest.py and tests/conftest.py do.
_SRC_PATH = str(ROOT / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OPTS = argparse.Namespace()
CFG = types.SimpleNamespace()

ENV_INFO = {}

TIMEOUT = -100
WORKER_PROCESSES = []
INTERRUPTED = False

IS_TTY = sys.stdout.isatty()
USE_COLORS = IS_TTY

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[93m"
CYAN = "\033[36m"
DIM = "\033[2m"
NC = "\033[0m"

if not USE_COLORS:
    RED = GREEN = YELLOW = CYAN = DIM = NC = ""

# ---------------------------------------------------------------------------
# DType map for benchmark results (matching FlagGems tools/consts.py)
# ---------------------------------------------------------------------------
DTYPE_MAP = {
    "torch.float16": "fp16",
    "torch.float32": "fp32",
    "torch.bfloat16": "bf16",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int8": "int8",
    "torch.uint8": "uint8",
    "torch.int64": "int64",
    "torch.bool": "bool",
    "torch.complex64": "cf64",
    "torch.float8_e4m3fn": "float8_e4m3fn",
    "torch.float8_e5m2": "float8_e5m2",
}


# ===================================================================
# Logging helpers
# ===================================================================
def pinfo(msg, **kwargs):
    print(f"{GREEN}[INFO]{NC} {msg}", flush=True, **kwargs)


def perror(msg, **kwargs):
    print(f"{RED}[ERROR]{NC} {msg}", flush=True, **kwargs)


def pwarn(msg, **kwargs):
    print(f"{YELLOW}[WARN]{NC} {msg}", flush=True, **kwargs)


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)


# ===================================================================
# LiveDisplay — TTY progress bar + per-GPU status
# ===================================================================
class LiveDisplay:
    def __init__(self, gpu_ids, op_count, op_width=20):
        self.gpu_ids = gpu_ids
        self.op_count = op_count
        self.op_width = op_width
        self.gpu_index = {gid: i + 1 for i, gid in enumerate(gpu_ids)}
        nums_width = len(f"{op_count}/{op_count} ops")
        self.bar_width = max(20, 55 + op_width - 12 - 3 - nums_width)
        self.nums_width = nums_width
        progress_line = self._fmt_progress(0)
        gpu_lines = [f"{DIM}[GPU {gid:2d}] idle{NC}" for gid in gpu_ids]
        self.footer = [progress_line] + gpu_lines
        self.n = len(self.footer)
        self.footer_drawn = False

    def _fmt_progress(self, tests_done):
        total_tests = self.op_count * 2
        color = GREEN if tests_done >= total_tests else CYAN
        bar = (
            _progress_bar(tests_done, total_tests, self.bar_width, color=color)
            if self.op_count
            else " " * self.bar_width
        )
        ops_done = tests_done // 2
        nums = f"{ops_done}/{self.op_count} ops"
        return f"[Progress] [{color}{bar}{NC}]  {nums:>{self.nums_width}}"

    def _draw_footer(self):
        if not IS_TTY:
            return
        for line in self.footer:
            sys.stdout.write(line + "\n")
        sys.stdout.flush()
        self.footer_drawn = True

    def _erase_footer(self):
        if not IS_TTY or not self.footer_drawn:
            return
        for _ in range(self.n):
            sys.stdout.write("\033[A\033[2K")

    def init(self):
        if IS_TTY:
            self._draw_footer()

    def log(self, msg):
        if IS_TTY:
            self._erase_footer()
            sys.stdout.write(msg + "\n")
            self._draw_footer()
        else:
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()

    def update_gpu(self, gpu_id, status_line):
        idx = self.gpu_index.get(gpu_id)
        if idx is None:
            return
        self.footer[idx] = status_line
        if IS_TTY:
            self._erase_footer()
            self._draw_footer()

    def update_progress(self, tests_done):
        self.footer[0] = self._fmt_progress(tests_done)
        if IS_TTY:
            self._erase_footer()
            self._draw_footer()
        else:
            sys.stdout.write(self.footer[0] + "\n")
            sys.stdout.flush()

    def finish(self):
        if IS_TTY:
            self._erase_footer()
            sys.stdout.flush()


def _progress_bar(done, total, width=40, color=""):
    if not total:
        return " " * width
    frac = done * width / total
    full = int(frac)
    has_half = (frac - full) >= 0.5 and full < width
    empty = width - full - (1 if has_half else 0)
    bar = "█" * full
    if has_half:
        bar += f"{DIM}█{NC}{color}"
    bar += " " * empty
    return bar


def _format_status(status, dur):
    STATUS_MAP = {
        "Passed": (GREEN, "OK"),
        "Failed": (RED, "FAILED"),
        "Timeout": (RED, "TIMEOUT"),
        "Error": (RED, "ERROR"),
        "NotFound": (YELLOW, "NOTFOUND"),
        "Skipped": (YELLOW, "SKIPPED"),
    }
    color, label = STATUS_MAP.get(status, (YELLOW, status.upper()))
    return f"{color}[{label:<8} {dur:>6.1f}s]{NC}"


# ===================================================================
# Operator discovery (from YAML)
# ===================================================================
def get_ops_from_inventory():
    catalog = []
    try:
        op_inventory = ROOT / "conf" / "operators.yaml"
        with open(str(op_inventory), "r") as f:
            data = yaml.safe_load(f)
            catalog = data.get("ops", [])
    except Exception as e:
        perror(f"Failed to load operator inventory: {e}")
    return catalog


# ===================================================================
# Environment probing
# ===================================================================
def _probe_torch():
    ENV_INFO.setdefault("torch", {})
    try:
        import torch
        ENV_INFO["torch"]["version"] = torch.__version__
        pinfo(f"PyTorch detected ... {torch.__version__}")
    except Exception as e:
        perror(f"pytorch not installed, please fix it - {e}")
        sys.exit(-1)

    # Detect the active accelerator backend. FlagTensor runtime auto-discovers
    # the vendor (NVIDIA CUDA / Huawei Ascend NPU / ...). We probe both the
    # generic runtime flag and the legacy torch.cuda path so the env record
    # stays compatible with older single-vendor runs.
    try:
        from flagtensor.runtime import (
            device as _ft_device,
            device_str as _ft_device_str,
            is_accelerator_available as _ft_is_accelerator_available,
        )
        ENV_INFO["torch"]["device_name"] = _ft_device_str
        ENV_INFO["torch"]["vendor"] = _ft_device.vendor_name
        ENV_INFO["torch"]["accelerator_available"] = _ft_is_accelerator_available()
        pinfo(f"PyTorch device ... {_ft_device_str} (vendor={_ft_device.vendor_name})")
        try:
            dev_count = _ft_device.device_count
            ENV_INFO["torch"]["device_count"] = dev_count
            pinfo(f"PyTorch device count ... {dev_count}")
        except Exception:
            ENV_INFO["torch"]["device_count"] = 0
    except Exception:
        # Fallback: legacy NVIDIA-only probing.
        try:
            cuda_available = torch.cuda.is_available()
            ENV_INFO["torch"]["cuda_available"] = cuda_available
            ENV_INFO["torch"]["device_name"] = "cuda"
            ENV_INFO["torch"]["vendor"] = "nvidia"
            ENV_INFO["torch"]["accelerator_available"] = cuda_available
            pinfo(f"PyTorch CUDA support ... {cuda_available}")
        except Exception:
            ENV_INFO["torch"]["cuda_available"] = False
            ENV_INFO["torch"]["accelerator_available"] = False
            ENV_INFO["torch"]["device_name"] = "N/A"
            ENV_INFO["torch"]["vendor"] = "unknown"

        try:
            dev_name = torch.cuda.get_device_name()
            ENV_INFO["torch"]["device_name_hardware"] = dev_name
            pinfo(f"PyTorch device name ... {dev_name}")
        except Exception:
            ENV_INFO["torch"]["device_name_hardware"] = "N/A"

        try:
            dev_count = torch.cuda.device_count()
            ENV_INFO["torch"]["device_count"] = dev_count
            pinfo(f"PyTorch device count ... {dev_count}")
        except Exception:
            ENV_INFO["torch"]["device_count"] = 0


def _probe_flagtree():
    try:
        version = metadata.version("flagtree")
        ENV_INFO["flagtree"] = version
        pinfo(f"FlagTree detected ... {version}")
        has_flagtree = True
    except Exception:
        has_flagtree = False
        ENV_INFO["flagtree"] = None
        pwarn("FlagTree not installed, testing Triton ...")

    try:
        import triton
        version = triton.__version__
        ENV_INFO["triton"] = {"version": version}
        pinfo(f"Triton detected ... {version}")
        if version:
            has_config = hasattr(triton, "Config")
            ENV_INFO["triton"]["has_config"] = has_config
            pinfo(f"Triton has Config ... [{has_config}]")
    except Exception:
        ENV_INFO["triton"] = None
        if not has_flagtree:
            perror("Neither FlagTree nor Triton is installed, please fix it.")
            sys.exit(-1)


def _probe_flagtensor():
    try:
        import flagtensor
        ENV_INFO["flagtensor"] = {"version": getattr(flagtensor, "__version__", "0.1.0")}
        pinfo(f"flagtensor loaded OK")
    except Exception as e:
        perror(f"flagtensor failed to load: {e}")
        sys.exit(-1)


def probe_env():
    ENV_INFO["architecture"] = platform.machine()
    ENV_INFO["os_name"] = platform.system()
    ENV_INFO["os_release"] = platform.release()
    ENV_INFO["python"] = platform.python_version()

    _probe_torch()
    _probe_flagtree()
    _probe_flagtensor()


# ===================================================================
# GPU env
# ===================================================================
def get_env(gpu_id):
    """Build the per-worker environment, isolating one accelerator device.

    On NVIDIA we set ``CUDA_VISIBLE_DEVICES``; on Ascend we set
    ``ASCEND_RT_VISIBLE_DEVICES`` (the CANN equivalent understood by
    torch_npu). Both are exported so the same worker code runs on either
    vendor without per-test branching.
    """
    env = os.environ.copy()
    # NVIDIA device isolation
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Ascend device isolation (CANN / torch_npu)
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    return env


# ===================================================================
# Command execution with timeout
# ===================================================================
def run_cmd(op, cmd, cwd=None, env=None, timeout=600, flavor=None):
    stdout = subprocess.DEVNULL
    stderr = subprocess.DEVNULL
    if CFG.dump_output:
        op_dir = CFG.output_dir / op
        stdout_log = str(op_dir / f"{flavor}_stdout.log")
        stderr_log = str(op_dir / f"{flavor}_stderr.log")
        try:
            stdout = open(stdout_log, "w")
            stderr = open(stderr_log, "w")
        except Exception:
            pass

    if cwd is None:
        cwd = ROOT

    p = subprocess.Popen(
        shlex.split(cmd),
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )

    try:
        p.wait(timeout=timeout)
        return p.returncode
    except subprocess.TimeoutExpired:
        pgid = os.getpgid(p.pid)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return TIMEOUT
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return TIMEOUT
    except Exception as e:
        perror(f"run_cmd failed: {e}")
        return -1
    finally:
        if stdout != subprocess.DEVNULL:
            stdout.close()
        if stderr != subprocess.DEVNULL:
            stderr.close()


# ===================================================================
# Accuracy result parsing (from conftest --record json output)
# ===================================================================
def parse_accuracy_data(result_file):
    raw_data = {}
    try:
        with result_file.open("r") as f:
            raw_data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return {
            "total": 0, "skipped": 0, "failed": 0, "passed": 0,
            "status": "Error",
            "details": {"error": f"Invalid JSON in {result_file}"},
        }

    passed = []
    skipped = {}
    failed = {}
    num_skipped = 0
    num_failed = 0
    num_passed = 0
    skipped_with_issue = False

    for test_case, item in raw_data.items():
        case_str = test_case[:test_case.find("[")]
        result = item.get("result", "")
        params = [case_str]
        for k, v in item.get("params", {}).items():
            params.append(str(v).replace(" ", ""))
        param_str = ":".join(params)

        if result == "passed":
            passed.append(param_str)
            num_passed += 1
        elif result == "skipped":
            reason = item.get("reason", "Unknown")
            if "Issue" in reason:
                skipped_with_issue = True
            skipped.setdefault(reason, set())
            skipped[reason].add(param_str)
            num_skipped += 1
        else:
            reason = item.get("reason", "Unknown")
            failed.setdefault(reason, set())
            failed[reason].add(param_str)
            num_failed += 1

    num_total = num_passed + num_skipped + num_failed
    result = {
        "total": num_total, "skipped": num_skipped,
        "failed": num_failed, "passed": num_passed, "details": {},
    }
    if len(skipped) == 0 and len(failed) == 0:
        result["status"] = "NotFound" if len(passed) == 0 else "Passed"
        return result

    if num_failed > 0:
        result["status"] = "Failed"
        for k, v in failed.items():
            failed[k] = list(v)
        result["details"]["failed"] = failed
        return result

    result["status"] = "Failed" if skipped_with_issue else "Skipped"
    for k, v in skipped.items():
        skipped[k] = list(v)
    result["details"]["skipped"] = skipped
    return result


# ===================================================================
# Benchmark result parsing (from conftest --record json output)
# ===================================================================
def parse_perf_data(op, result_file):
    raw_data = {}
    try:
        with result_file.open("r") as f:
            raw_data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return {"status": "Error", "reason": f"Invalid JSON in {result_file}"}

    # The benchmark output may contain multiple top-level keys (e.g. a new op
    # name AND a legacy identifier for the same operator).  We first match by
    # the exact op name; if that record lacks performance details we scan all
    # remaining keys for the first one that *does* carry details, so that the
    # parsed data block is always populated regardless of which key pytest's
    # JSON plugin attributed the details to.
    data = raw_data.get(op, {})
    if not data:
        return {"status": "NotFound"}

    # If the primary key has no details, try to find them on a sibling key.
    # This also covers cases where the primary key has result="failed" but a
    # legacy key still recorded partial benchmark data.
    if not data.get("details"):
        for alt_key, alt_value in raw_data.items():
            if alt_key == op:
                continue
            alt_details = alt_value.get("details")
            if alt_details:
                data = dict(data)  # shallow copy to leave original intact
                data["details"] = alt_details
                break

    result = data.get("result", "NotFound")
    if result in ("failed", "skipped"):
        # Still try to populate bench data so summary.json is not empty.
        bench_res = {}
        records = data.get("details", [])
        for item in records:
            dtype = DTYPE_MAP.get(item.get("dtype", ""), item.get("dtype", ""))
            details = {}
            total = 0.0
            count = 0
            for res in item.get("result", []):
                shape = str(res.get("shape_detail", "Unknown")).replace(" ", "")
                details.setdefault(shape, {})
                details[shape]["base"] = res.get("latency_base", 0.0)
                details[shape]["gems"] = res.get("latency", 0.0)
                speedup = res.get("speedup", 0.0)
                details[shape]["speedup"] = speedup
                count += 1
                total += speedup
            if details:
                bench_res[dtype] = {
                    "result": "OK",
                    "details": details,
                    "speedup": total / count,
                }
        return {
            "status": result.title(),
            "reason": data.get("reason", "Unknown"),
            "test_case": data.get("test_case", "Unknown"),
            "data": bench_res,
        }

    bench_res = {}
    records = data.get("details", [])

    for item in records:
        dtype = DTYPE_MAP.get(item.get("dtype", ""), item.get("dtype", ""))
        details = {}
        total = 0.0
        count = 0
        for res in item.get("result", []):
            shape = str(res.get("shape_detail", "Unknown")).replace(" ", "")
            details.setdefault(shape, {})
            details[shape]["base"] = res.get("latency_base", 0.0)
            details[shape]["gems"] = res.get("latency", 0.0)
            speedup = res.get("speedup", 0.0)
            details[shape]["speedup"] = speedup
            count += 1
            total += speedup

        if details:
            bench_res[dtype] = {
                "result": "OK",
                "details": details,
                "speedup": total / count,
            }
        else:
            bench_res[dtype] = {
                "result": "Unknown", "details": {}, "speedup": 0,
            }

    # Use the primary key's result status ("passed"/"failed"), but fall back
    # to "Passed" when bench_res has valid data even if the primary key
    # doesn't carry an explicit result field.
    status = (
        result.title() if result not in ("NotFound",)
        else "Passed" if bench_res else "NotFound"
    )
    return {
        "status": status,
        "data": bench_res,
        "test_case": data.get("test_case", "Unknown"),
    }


# ===================================================================
# Accuracy test (single op, single GPU)
# ===================================================================
def _find_accuracy_test(op: str) -> Path:
    """Find the accuracy test file for an operator."""
    # Check per-operator test files first
    for cat in ("unary", "binary", "contraction", "sparse"):
        candidate = ROOT / "tests" / cat / f"test_{op}.py"
        if candidate.exists():
            return candidate
    # Fallback to category-level test
    for op_meta in get_ops_from_inventory():
        if op_meta.get("name") == op:
            cat = op_meta.get("category", "unary")
            legacy = ROOT / "tests" / cat / f"test_{cat}_correctness.py"
            if legacy.exists():
                return legacy
    return ROOT / "tests"


def _find_benchmark_test(op: str) -> Path:
    """Find the benchmark test file for an operator."""
    for op_meta in get_ops_from_inventory():
        if op_meta.get("name") == op:
            cat = op_meta.get("category", "unary")
            bf = ROOT / "benchmark" / f"test_{cat}_perf.py"
            if bf.exists():
                return bf
    return ROOT / "benchmark"


# ===================================================================
# Accuracy test (single op, single GPU)
# ===================================================================
def run_accuracy_q(gpu_id, op):
    """Run accuracy test for one op, using --record json --ref cpu. Returns result dict."""
    env = get_env(str(gpu_id))

    # Find the test file for this operator
    test_file = _find_accuracy_test(op)
    accuracy_dir = test_file.parent
    result_file = accuracy_dir / f"accuracy_{op}.json"
    if result_file.exists():
        result_file.unlink()

    op_dir = CFG.output_dir / op
    ensure_dir(op_dir)

    cmd = f'pytest {test_file} -m "{op}" --record json --output {result_file} --ref cpu -vs'

    dur = time.time()
    code = run_cmd(op, cmd, cwd=accuracy_dir, env=env, flavor="accuracy")
    dur = time.time() - dur

    if code == TIMEOUT:
        return {
            "status": "Timeout", "exit_code": TIMEOUT,
            "total": 0, "passed": 0, "failed": 0,
            "skipped": 0, "errors": 0, "duration": dur,
        }

    if not result_file.exists():
        return {
            "status": "Error", "exit_code": code,
            "total": 0, "passed": 0, "failed": 0,
            "skipped": 0, "errors": 1, "duration": dur,
            "data_file": None,
        }

    dest = op_dir / "accuracy_result.json"
    shutil.move(str(result_file), str(dest))
    result_file = dest

    result = parse_accuracy_data(result_file)
    result["exit_code"] = code
    result["duration"] = dur
    result["data_file"] = str(result_file.relative_to(CFG.output_dir))
    return result


# ===================================================================
# Benchmark test (single op, single GPU)
# ===================================================================
def run_benchmark_q(gpu_id, op):
    """Run benchmark for one op, using --level core --record json. Returns result dict."""
    env = get_env(str(gpu_id))

    bench_file = _find_benchmark_test(op)
    benchmark_dir = bench_file.parent
    result_file = benchmark_dir / f"benchmark_{op}.json"
    if result_file.exists():
        result_file.unlink()

    op_dir = CFG.output_dir / op
    ensure_dir(op_dir)

    dur = time.time()
    cmd = f'pytest {bench_file} -m "{op}" --level core --record json --output {result_file}'
    code = run_cmd(op, cmd, cwd=benchmark_dir, env=env, flavor="performance")
    dur = time.time() - dur

    if code == TIMEOUT:
        return {
            "status": "Timeout", "exit_code": TIMEOUT,
            "duration": dur, "data": {},
        }

    if not result_file.exists():
        return {
            "status": "NotFound", "duration": dur, "exit_code": code, "data": {},
        }

    dest = op_dir / "performance_result.json"
    shutil.move(str(result_file), str(dest))
    result_file = dest

    record = {
        "duration": dur, "exit_code": code,
        "data_file": str(result_file.relative_to(CFG.output_dir)), "data": {},
    }
    record.update(parse_perf_data(op, result_file))
    return record


# ===================================================================
# Worker process — one per GPU
# ===================================================================
def worker_proc(gpu_id, work_queue, display_queue):
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

        op_dir = CFG.output_dir / op
        ensure_dir(op_dir)

        # Load operator metadata from YAML
        op_meta = {}
        for o in get_ops_from_inventory():
            if o.get("name") == op or o.get("id") == op:
                op_meta = o
                break

        display_queue.put(("start", gpu_id, "accuracy", op))
        acc = run_accuracy_q(gpu_id, op)
        display_queue.put((
            "done", gpu_id, "accuracy", op,
            acc.get("status", "Error"), acc.get("duration", 0),
        ))

        display_queue.put(("start", gpu_id, "benchmark", op))
        perf = run_benchmark_q(gpu_id, op)
        display_queue.put((
            "done", gpu_id, "benchmark", op,
            perf.get("status", "Error"), perf.get("duration", 0),
        ))

        result = {
            "customized": False,
            "accuracy": acc,
            "performance": perf,
            "labels": op_meta.get("labels", []),
        }
        worker_result[op] = result

        json_path = CFG.output_dir / f"summary{gpu_id}.json"
        tmp_path = json_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(worker_result, f, indent=2)
        os.replace(tmp_path, json_path)

    display_queue.put(("exit", gpu_id))


# ===================================================================
# Display loop
# ===================================================================
def display_loop(queue, display, n_workers):
    exited = 0
    tests_done = 0
    per_gpu_done = {gid: 0 for gid in display.gpu_ids}

    while exited < n_workers:
        try:
            msg = queue.get(timeout=1)
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
            label = "accuracy " if phase == "accuracy" else "benchmark"
            op_display = (
                op if len(op) <= display.op_width
                else op[:display.op_width - 3] + "..."
            )
            op_col = op_display.ljust(display.op_width)
            n = per_gpu_done.get(gpu_id, 0)
            if IS_TTY:
                display.update_gpu(
                    gpu_id,
                    f"[GPU {gpu_id:2d}] ({n:>3} done)  {label} {op_col}",
                )
            else:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                display.log(f"[INFO] [{ts}][GPU {gpu_id:2d}] {label} {op_col} ...")

        elif kind == "done":
            _, gpu_id, phase, op, status, dur = msg
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            label = "accuracy " if phase == "accuracy" else "benchmark"
            op_display = (
                op if len(op) <= display.op_width
                else op[:display.op_width - 3] + "..."
            )
            op_col = op_display.ljust(display.op_width)
            status_str = _format_status(status, dur)
            log_line = (
                f"{GREEN}[INFO]{NC} [{ts}][GPU {gpu_id:2d}]"
                f" {label} {op_col} {status_str}"
            )

            tests_done += 1
            if phase == "benchmark":
                per_gpu_done[gpu_id] = per_gpu_done.get(gpu_id, 0) + 1

            ops_done = tests_done // 2
            total_ops = display.op_count
            pct = ops_done * 100 // total_ops
            if not IS_TTY:
                total_w = len(str(total_ops))
                log_line += f"  ({pct:>3}% {ops_done:>{total_w}}/{total_ops} ops)"

            display.footer[0] = display._fmt_progress(tests_done)
            display.log(log_line)


# ===================================================================
# Cleanup
# ===================================================================
def cleanup_intermediate_files():
    patterns = [
        (ROOT / "tests", "accuracy_*.json"),
        (ROOT / "benchmark", "benchmark_*.json"),
    ]
    for directory, pattern in patterns:
        for f in directory.glob(pattern):
            try:
                f.unlink()
            except OSError:
                pass

    if hasattr(CFG, "output_dir"):
        for f in CFG.output_dir.glob("summary*.tmp"):
            try:
                f.unlink()
            except OSError:
                pass


def terminate_workers():
    for p in WORKER_PROCESSES:
        if p.is_alive():
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
    for p in WORKER_PROCESSES:
        p.join(timeout=5)
        if p.is_alive():
            p.kill()


def handle_interrupt(signum, frame):
    global INTERRUPTED
    if INTERRUPTED:
        return
    INTERRUPTED = True
    if IS_TTY:
        sys.stdout.write("\n")
    pwarn("Interrupted. Cleaning up ...")
    terminate_workers()
    cleanup_intermediate_files()
    pwarn("Cleanup done.")
    sys.exit(1)


# ===================================================================
# Operator selection
# ===================================================================
def get_ops_to_test():
    op_catalog = get_ops_from_inventory()
    skip_cpu_tests = []
    for op in op_catalog:
        labels = op.get("labels", [])
        if "NoCPU" in labels:
            skip_cpu_tests.append(op["name"])
    CFG.skip_cpu_tests = skip_cpu_tests

    if OPTS.ops:
        ops = []
        for op in OPTS.ops.split(","):
            ops.append(op.strip())
        return ops

    if OPTS.op_list_file:
        lines = []
        try:
            with open(OPTS.op_list_file, "r") as f:
                lines = f.readlines()
        except Exception as e:
            perror(f"Failed reading the specified op list file: {e}")
            return []
        ops = []
        for ln in lines:
            ln = ln.strip()
            if ln.startswith("#"):
                continue
            ops.append(ln)
        return ops

    effective_stages = []
    for s in OPTS.stages.split(","):
        stage = s.strip()
        if stage not in ("alpha", "beta", "stable", "all", "removed", "experimental", "active"):
            pwarn(f"ignoring unsupported stage name '{s}'...")
            continue
        if stage == "all":
            effective_stages = ["alpha", "beta", "stable", "experimental", "active"]
            break
        effective_stages.append(stage)

    if not effective_stages:
        effective_stages = ["stable"]

    ops = []
    for op in op_catalog:
        stage_dicts = op.get("stages", [])
        if len(stage_dicts) == 0:
            continue
        stage = next(iter(stage_dicts[-1].keys()), None)
        if stage not in effective_stages:
            continue
        if OPTS.start is not None and op.get("name", "") < OPTS.start:
            continue
        ops.append(op["name"])

    return ops


# ===================================================================
# Main
# ===================================================================
def main():
    global OPTS

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", required=False, help="a comma-separated list of operator names")
    parser.add_argument("--op-list-file", required=False, help="path to operator list file")
    parser.add_argument("--start", required=False, help="the name of the first operator")
    parser.add_argument("--gpus", default="0", help="a comma-separated list of GPU IDs")
    parser.add_argument("--output-dir", default="results", help="relative path to root for test data")
    parser.add_argument(
        "--stages", required=False, default="stable",
        help="a comma-separate list of op stages",
    )
    parser.add_argument(
        "--dump-output", action="store_true", default=False,
        help="Dump stdout/stderr of each test to log files",
    )
    parser.add_argument(
        "--color", choices=["auto", "always", "never"], default="auto",
        help="Control ANSI color output: auto (TTY only), always, or never",
    )
    OPTS = parser.parse_args()
    CFG.dump_output = OPTS.dump_output
    CFG.start = OPTS.start

    # Color mode
    global USE_COLORS, RED, GREEN, YELLOW, CYAN, DIM, NC
    if OPTS.color == "always":
        USE_COLORS = True
        RED, GREEN, YELLOW, CYAN, DIM, NC = (
            "\033[31m", "\033[32m", "\033[93m", "\033[36m", "\033[2m", "\033[0m",
        )
    elif OPTS.color == "never":
        USE_COLORS = False
        RED = GREEN = YELLOW = CYAN = DIM = NC = ""

    probe_env()

    ops = get_ops_to_test()
    op_count = len(ops)
    if op_count == 0:
        pwarn("No operators to test. Please specify at least one operator.")
        sys.exit(1)
    pinfo(f"Testing {op_count} operators ...")

    CFG.ops = ops

    output_dir = Path(OPTS.output_dir)
    ensure_dir(output_dir)
    CFG.output_dir = output_dir

    gpu_list = OPTS.gpus.strip().split(",")
    if len(gpu_list) == 0:
        pwarn("Empty GPU list specified.")
        sys.exit(1)

    gpu_ids = [int(x) for x in gpu_list if x.strip()]
    gpu_count = len(gpu_ids)

    op_width = min(max(len(op) for op in ops), 40) if ops else 20

    work_queue = Queue()
    for op in ops:
        work_queue.put(op)

    display_queue = Queue()
    display = LiveDisplay(gpu_ids, op_count, op_width=op_width)

    for gpu in gpu_ids:
        p = Process(target=worker_proc, args=(gpu, work_queue, display_queue))
        p.start()
        WORKER_PROCESSES.append(p)

    display.init()
    display_loop(display_queue, display, gpu_count)

    for p in WORKER_PROCESSES:
        p.join()

    display.finish()

    # Merge per-GPU summaries
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    op_data = {}
    for gpu_id in gpu_ids:
        gpu_file = CFG.output_dir / f"summary{gpu_id}.json"
        if not gpu_file.exists():
            perror(f"GPU {gpu_id} failed to produce a summary, recovery needed.")
            continue
        with gpu_file.open("r") as f:
            try:
                result = json.load(f)
            except (json.JSONDecodeError, ValueError):
                perror(f"GPU {gpu_id} summary is invalid JSON, skipping.")
                continue
            op_data.update(result)

    final_data = {
        "timestamp": timestamp,
        "env": ENV_INFO,
        "result": op_data,
    }

    json_path = CFG.output_dir / "summary.json"
    with json_path.open("w") as f:
        json.dump(final_data, f, indent=2)

    # Print tally
    acc_pass = sum(
        1 for v in op_data.values()
        if v.get("accuracy", {}).get("status") == "Passed"
    )
    tf_pass = sum(
        1 for v in op_data.values()
        if v.get("performance", {}).get("status") == "Passed"
    )
    pinfo(f"Accuracy:    {acc_pass}/{op_count} passed")
    pinfo(f"Performance: {tf_pass}/{op_count} passed")
    pinfo(f"Results:     {json_path}")
    pinfo("Test completed.")


if __name__ == "__main__":
    main()
