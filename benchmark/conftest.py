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

"""Pytest configuration for FlagTensor benchmark (performance) tests.

Provides --mode/--level/--warmup/--iter/--record/--dtypes CLI options and
result-recording hooks aligned with FlagGems benchmark/conftest.py.
"""
import json
import logging
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flagtensor_registry import load_operator_registry
from benchmark import consts

# ---------------------------------------------------------------------------
# Re-export device info for test files
# ---------------------------------------------------------------------------
device_name = "cuda"
vendor_name = "nvidia"

record_logger = logging.getLogger("flagtensor_benchmark")
record_logger.propagate = False

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
BUILTIN_MARKS = (
    "parametrize", "skip", "skipif", "xfail", "usefixtures",
    "filterwarnings", "timeout", "tryfirst", "trylast",
)
REGISTERED_MARKS = []
TEST_RESULTS = {}
REPORT_FILE = "benchmark_result.json"
Config = None  # BenchConfig singleton, set in pytest_configure


# ---------------------------------------------------------------------------
# BenchConfig
# ---------------------------------------------------------------------------
class BenchConfig:
    def __init__(self):
        self.mode = consts.BenchMode.KERNEL
        self.bench_level = consts.BenchLevel.COMPREHENSIVE
        self.warm_up = consts.DEFAULT_WARMUP_COUNT
        self.repetition = consts.DEFAULT_ITER_COUNT
        self.record_log = False
        self.record_json = False
        self.user_desired_dtypes = None
        self.user_desired_metrics = None
        self.shape_file = os.path.join(os.path.dirname(__file__), "core_shapes.yaml")
        self.query = False
        self.parallel = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def update_result(op, data):
    if not Config.record_json:
        return
    details = TEST_RESULTS.setdefault(op, {}).setdefault("details", [])
    # Group by dtype/mode/level: merge result arrays for same (dtype, mode, level)
    target = (data.get("dtype", ""), data.get("mode", ""), data.get("level", ""))
    for existing in details:
        existing_target = (
            existing.get("dtype", ""),
            existing.get("mode", ""),
            existing.get("level", ""),
        )
        if existing_target == target:
            existing.setdefault("result", []).extend(data.get("result", []))
            return
    details.append(data)


def emit_record_logger(message: str) -> None:
    if not Config.record_log:
        return
    if record_logger.handlers:
        handler = record_logger.handlers[0]
        if getattr(handler, "stream", None) is None:
            handler.acquire()
            try:
                handler.stream = handler._open()
            finally:
                handler.release()
    record_logger.info(message)


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--mode",
        action="store",
        default="kernel",
        required=False,
        choices=["kernel", "operator", "wrapper"],
        help="Benchmark mode: kernel (device kernel), operator (end2end), wrapper (runtime)",
    )
    parser.addoption(
        "--level",
        action="store",
        default="comprehensive",
        required=False,
        choices=[level.value for level in consts.BenchLevel],
        help="Benchmark level: comprehensive or core",
    )
    parser.addoption(
        "--warmup",
        default=consts.DEFAULT_WARMUP_COUNT,
        help="Number of warmup runs",
    )
    parser.addoption(
        "--iter",
        default=consts.DEFAULT_ITER_COUNT,
        help="Number of benchmark iterations",
    )
    parser.addoption(
        "--query", action="store_true", default=False, help="Enable query mode"
    )
    parser.addoption(
        "--metrics",
        action="append",
        default=None,
        required=False,
        choices=sorted(consts.ALL_AVAILABLE_METRICS),
        help="Benchmark metrics to collect",
    )
    parser.addoption(
        "--dtypes",
        action="append",
        default=None,
        required=False,
        choices=[
            str(ele).split(".")[-1]
            for ele in consts.FLOAT_DTYPES + consts.INT_DTYPES
            + consts.BOOL_DTYPES + [torch.cfloat]
        ],
        help="Data types for benchmarks",
    )
    parser.addoption(
        "--shape_file",
        action="store",
        default=os.path.join(os.path.dirname(__file__), "core_shapes.yaml"),
        required=False,
        help="Shape configuration file for benchmarks",
    )
    try:
        parser.addoption(
            "--record",
            action="store",
            default="none",
            required=False,
            choices=["none", "log", "json"],
            help="Record benchmark results: none, log, or json",
        )
        parser.addoption(
            "--output",
            default=REPORT_FILE,
            help="Path to report file for JSON output",
        )
    except ValueError:
        pass
    parser.addoption(
        "--parallel",
        action="store",
        type=int,
        default=0,
        help="Enable multi-GPU parallel benchmark execution (0=serial)",
    )
    try:
        parser.addoption(
            "--collect-marks",
            action="store_true",
            help="Collect test marks without executing",
        )
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------
def pytest_configure(config):
    global Config, REPORT_FILE, REGISTERED_MARKS

    Config = BenchConfig()

    # Register markers
    config.addinivalue_line("markers", "performance: benchmark performance marker")
    for spec in load_operator_registry():
        config.addinivalue_line("markers", f"{spec.benchmark_mark}: benchmark")

    REGISTERED_MARKS = {
        marker.split(":")[0].strip() for marker in config.getini("markers")
    }

    Config.mode = consts.BenchMode(config.getoption("--mode"))
    Config.query = config.getoption("--query")
    Config.bench_level = consts.BenchLevel(config.getoption("--level"))
    Config.warm_up = int(config.getoption("--warmup"))
    Config.repetition = int(config.getoption("--iter"))

    types_str = config.getoption("--dtypes")
    Config.user_desired_dtypes = (
        [getattr(torch, d) for d in types_str] if types_str else types_str
    )
    Config.user_desired_metrics = config.getoption("--metrics")
    Config.shape_file = config.getoption("--shape_file")
    Config.record_log = config.getoption("--record") == "log"
    Config.record_json = config.getoption("--record") == "json"
    Config.parallel = int(config.getoption("--parallel") or 0)

    if Config.record_json:
        Config.output = config.getoption("--output")
        REPORT_FILE = Config.output

    if Config.record_log:
        cmd_args = [
            arg.replace(".py", "").replace("=", "_").replace("/", "_")
            for arg in config.invocation_params.args
        ]
        log_file = "result_{}.log".format("_".join(cmd_args)).replace("_-", "-")
        for h in list(record_logger.handlers):
            record_logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        handler = logging.FileHandler(log_file, mode="w", encoding="utf-8", delay=False)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        record_logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _bench_setup_once(request):
    if request.config.getoption("--query"):
        print("\nThis is query mode; all benchmark functions will be skipped.")


@pytest.fixture(scope="function", autouse=True)
def _bench_clear_function_cache():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="module", autouse=True)
def _bench_clear_module_cache():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture()
def extract_and_log_op_attributes(request):
    import torch as _torch

    print("")
    op_attributes = []
    for mark in request.node.iter_markers():
        if mark.name in BUILTIN_MARKS:
            continue
        rec = consts.get_recommended_shapes(mark.name, None)
        if rec:
            attri = consts.OperationAttribute(
                op_name=mark.name,
                recommended_core_shapes=rec,
                shape_desc="default",
            )
            print(attri)
            op_attributes.append(attri.to_dict())

    if request.config.getoption("--query"):
        pytest.skip("Skipping benchmark due to the query parameter.")

    yield
    if Config.record_log and op_attributes:
        emit_record_logger(json.dumps(op_attributes, indent=2))


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
def _get_reason(report):
    if hasattr(report.longrepr, "reprcrash"):
        return report.longrepr.reprcrash.message
    if isinstance(report.longrepr, tuple):
        return report.longrepr[2]
    return str(report.longrepr)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    out = yield
    report = out.get_result()
    all_marks = [mark.name for mark in item.iter_markers()]
    # Exclude only builtin marks and "performance" (category marker)
    excluded = set(BUILTIN_MARKS) | {"performance"}
    marks = [m for m in all_marks if m not in excluded]
    report.opid = marks[0] if marks else item.nodeid


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    if not Config.record_json:
        return
    op = getattr(report, "opid", report.nodeid)
    TEST_RESULTS.setdefault(op, {})
    if report.when == "setup":
        if report.outcome == "skipped":
            TEST_RESULTS[op]["result"] = "skipped"
            TEST_RESULTS[op]["reason"] = _get_reason(report)
            TEST_RESULTS[op]["test_case"] = report.nodeid
    elif report.when == "call":
        TEST_RESULTS[op]["result"] = report.outcome
        TEST_RESULTS[op]["test_case"] = report.nodeid
        if report.outcome in ("skipped", "failed"):
            TEST_RESULTS[op]["reason"] = _get_reason(report)
        else:
            TEST_RESULTS[op]["reason"] = None


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not Config.record_json:
        return
    data = TEST_RESULTS
    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "r") as f:
            try:
                existing = json.load(f)
            except (json.JSONDecodeError, ValueError):
                existing = {}
        existing.update(TEST_RESULTS)
        data = existing
    with open(REPORT_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def pytest_collection_modifyitems(session, config, items):
    if not config.getoption("--collect-marks", False):
        return
    import yaml

    report = []
    for item in items:
        data = {}
        if item.cls:
            data["class"] = item.cls.__name__
        data["test_case"] = item.name
        if item.originalname:
            data["function"] = item.originalname
        data["file"] = item.location[0]
        all_marks = list(item.iter_markers())
        excluded = set(BUILTIN_MARKS) | set(REGISTERED_MARKS) | {"performance"}
        op_marks = [mark.name for mark in all_marks if mark.name not in excluded]
        data["marks"] = op_marks
        report.append(data)

    print(yaml.dump(report, indent=2))
    items.clear()
