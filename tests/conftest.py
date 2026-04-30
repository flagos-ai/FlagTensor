import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flagtensor_registry import load_operator_registry


def pytest_configure(config):
    for spec in load_operator_registry():
        config.addinivalue_line("markers", f"{spec.correctness_mark}: correctness compatibility marker")
