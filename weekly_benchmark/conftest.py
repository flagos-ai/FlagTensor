import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_addoption(parser):
    parser.addoption(
        "--mode",
        action="store",
        default="kernel",
        required=False,
        choices=["kernel", "operator", "wrapper"],
        help="weekly compatibility option",
    )
    parser.addoption(
        "--level",
        action="store",
        default="core",
        required=False,
        choices=["core", "comprehensive"],
        help="weekly compatibility option",
    )
    parser.addoption(
        "--warmup",
        action="store",
        default="5",
        required=False,
        help="weekly compatibility option",
    )
    parser.addoption(
        "--iter",
        action="store",
        default="10",
        required=False,
        help="weekly compatibility option",
    )
    parser.addoption(
        "--query",
        action="store_true",
        default=False,
        help="weekly compatibility option",
    )
    parser.addoption(
        "--metrics",
        action="append",
        default=None,
        required=False,
        help="weekly compatibility option",
    )
    parser.addoption(
        "--dtypes",
        action="append",
        default=None,
        required=False,
        help="weekly compatibility option",
    )
    parser.addoption(
        "--shape_file",
        action="store",
        default=None,
        required=False,
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
    for marker in [
        "abs",
        "acos",
        "acosh",
        "add",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "ceil",
        "conj",
        "cos",
        "cosh",
        "exp",
        "floor",
        "identity",
        "log",
        "max",
        "min",
        "mish",
        "mul",
        "neg",
        "performance",
        "rcp",
        "relu",
        "sigmoid",
        "sin",
        "sinh",
        "soft_plus",
        "soft_sign",
        "sqrt",
        "swish",
        "tan",
        "tanh",
    ]:
        config.addinivalue_line("markers", f"{marker}: weekly compatibility marker")
