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

"""Pytest configuration for FlagTensor correctness tests.

Provides --ref/--record/--output CLI options and result-recording hooks
aligned with FlagGems tests/conftest.py.
"""
import fcntl
import json
import logging
import os
import sys
from datetime import datetime

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flagtensor_registry import load_operator_registry

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
BUILTIN_MARKS = {
    "filterwarnings", "parametrize", "skip", "skipif", "timeout",
    "tryfirst", "trylast", "usefixtures", "xfail",
}
REGISTERED_MARKS = []
TEST_RESULTS = {}
RUNTEST_INFO = {}
RECORD_LOG = False
RECORD_JSON = False
TO_CPU = False
QUICK_MODE = False

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT_FILE = "accuracy_result.json"


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--ref",
        action="store",
        default="cuda",
        required=False,
        choices=["cuda", "cpu"],
        help="Device to run reference tests on (default: cuda)",
    )
    parser.addoption(
        "--quick",
        action="store_true",
        help="Run tests in quick mode",
    )
    try:
        parser.addoption(
            "--record",
            action="store",
            default="none",
            required=False,
            choices=["none", "log", "json"],
            help="Record test results: none, log, or json",
        )
        parser.addoption(
            "--output",
            help="Path to the result JSON file",
        )
    except ValueError:
        pass
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
    global RECORD_LOG, RECORD_JSON, REPORT_FILE, REGISTERED_MARKS
    global RUNTEST_INFO, TO_CPU, QUICK_MODE

    # Register operator markers from registry
    for spec in load_operator_registry():
        config.addinivalue_line(
            "markers", f"{spec.correctness_mark}: correctness test"
        )

    REGISTERED_MARKS = {
        marker.split(":")[0].strip() for marker in config.getini("markers")
    }
    RECORD_LOG = config.getoption("--record") == "log"
    RECORD_JSON = config.getoption("--record") == "json"
    TO_CPU = config.getoption("--ref") == "cpu"
    QUICK_MODE = config.getoption("--quick") is True

    if RECORD_JSON:
        report_file = config.getoption("--output")
        if report_file:
            REPORT_FILE = report_file

    if RECORD_LOG:
        RUNTEST_INFO = {}
        cmd_args = [
            arg.replace(".py", "").replace("=", "_").replace("/", "_")
            for arg in config.invocation_params.args
        ]
        logging.basicConfig(
            filename="result_{}.log".format("_".join(cmd_args)).replace("_-", "-"),
            filemode="w",
            level=logging.INFO,
            format="[%(levelname)s] %(message)s",
        )


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    TEST_RESULTS[item.nodeid] = {"params": None, "result": None, "opname": None}
    param_values = {}
    request = item._request
    if hasattr(request, "node") and hasattr(request.node, "callspec"):
        param_values = request.node.callspec.params
    TEST_RESULTS[item.nodeid]["params"] = param_values

    all_marks = [mark.name for mark in item.iter_markers()]
    operator_marks = [mark for mark in all_marks if mark not in BUILTIN_MARKS]
    TEST_RESULTS[item.nodeid]["opname"] = operator_marks


def _get_reason(report):
    if hasattr(report.longrepr, "reprcrash"):
        return report.longrepr.reprcrash.message
    if isinstance(report.longrepr, tuple):
        return report.longrepr[2]
    return str(report.longrepr)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    result = TEST_RESULTS.setdefault(
        report.nodeid, {"params": None, "result": None, "opname": None}
    )
    if report.when == "setup":
        if report.outcome == "skipped":
            result["result"] = "skipped"
            result["reason"] = _get_reason(report)
    elif report.when == "call":
        result["result"] = report.outcome
        if report.outcome in ("skipped", "failed"):
            result["reason"] = _get_reason(report)
        else:
            result["reason"] = None


def pytest_runtest_teardown(item, nextitem):
    if not RECORD_LOG:
        return
    if hasattr(item, "callspec"):
        all_marks = list(item.iter_markers())
        op_marks = [
            mark.name
            for mark in all_marks
            if mark.name not in BUILTIN_MARKS and mark.name not in REGISTERED_MARKS
        ]
        if op_marks:
            params = str(item.callspec.params)
            for op_mark in op_marks:
                RUNTEST_INFO.setdefault(op_mark, []).append(params)
        else:
            logging.warning("No mark at %s", item.function.__name__)


def pytest_sessionfinish(session, exitstatus):
    if RECORD_LOG:
        logging.info(json.dumps(RUNTEST_INFO, indent=2))


def pytest_terminal_summary(terminalreporter):
    if not RECORD_JSON:
        return
    data = TEST_RESULTS
    with open(REPORT_FILE, "a+") as json_file:
        fcntl.flock(json_file, fcntl.LOCK_EX)
        json_file.seek(0)
        content = json_file.read()
        if content:
            try:
                existing = json.loads(content)
                existing.update(TEST_RESULTS)
                data = existing
            except (json.JSONDecodeError, ValueError):
                pass
        json_file.seek(0)
        json_file.truncate()
        json.dump(data, json_file, indent=2, default=str)
        json_file.flush()
        os.fsync(json_file.fileno())


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
        op_marks = [
            mark.name
            for mark in all_marks
            if mark.name not in BUILTIN_MARKS and mark.name not in REGISTERED_MARKS
        ]
        data["marks"] = op_marks
        report.append(data)

    print(yaml.dump(report, indent=2))
    items.clear()
