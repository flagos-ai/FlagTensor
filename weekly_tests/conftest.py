import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_addoption(parser):
    parser.addoption(
        "--ref",
        action="store",
        default="cpu",
        required=False,
        choices=["cpu", "cuda"],
        help="reference device placeholder for weekly compatibility",
    )
    parser.addoption(
        "--mode",
        action="store",
        default="normal",
        required=False,
        choices=["normal", "quick"],
        help="weekly compatibility option",
    )
    parser.addoption(
        "--record",
        action="store",
        default="none",
        required=False,
        choices=["none", "log"],
        help="weekly compatibility option",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "abs: weekly compatibility marker")
    config.addinivalue_line("markers", "acos: weekly compatibility marker")
    config.addinivalue_line("markers", "acosh: weekly compatibility marker")
    config.addinivalue_line("markers", "add: weekly compatibility marker")
    config.addinivalue_line("markers", "asin: weekly compatibility marker")
    config.addinivalue_line("markers", "asinh: weekly compatibility marker")
    config.addinivalue_line("markers", "atan: weekly compatibility marker")
    config.addinivalue_line("markers", "atanh: weekly compatibility marker")
    config.addinivalue_line("markers", "ceil: weekly compatibility marker")
    config.addinivalue_line("markers", "conj: weekly compatibility marker")
    config.addinivalue_line("markers", "cos: weekly compatibility marker")
    config.addinivalue_line("markers", "cosh: weekly compatibility marker")
    config.addinivalue_line("markers", "exp: weekly compatibility marker")
    config.addinivalue_line("markers", "floor: weekly compatibility marker")
    config.addinivalue_line("markers", "identity: weekly compatibility marker")
    config.addinivalue_line("markers", "log: weekly compatibility marker")
    config.addinivalue_line("markers", "max: weekly compatibility marker")
    config.addinivalue_line("markers", "min: weekly compatibility marker")
    config.addinivalue_line("markers", "mish: weekly compatibility marker")
    config.addinivalue_line("markers", "mul: weekly compatibility marker")
    config.addinivalue_line("markers", "neg: weekly compatibility marker")
    config.addinivalue_line("markers", "rcp: weekly compatibility marker")
    config.addinivalue_line("markers", "relu: weekly compatibility marker")
    config.addinivalue_line("markers", "sigmoid: weekly compatibility marker")
    config.addinivalue_line("markers", "sin: weekly compatibility marker")
    config.addinivalue_line("markers", "sinh: weekly compatibility marker")
    config.addinivalue_line("markers", "soft_plus: weekly compatibility marker")
    config.addinivalue_line("markers", "soft_sign: weekly compatibility marker")
    config.addinivalue_line("markers", "sqrt: weekly compatibility marker")
    config.addinivalue_line("markers", "swish: weekly compatibility marker")
    config.addinivalue_line("markers", "tan: weekly compatibility marker")
    config.addinivalue_line("markers", "tanh: weekly compatibility marker")
