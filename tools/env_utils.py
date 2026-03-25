#!/usr/bin/env python3
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def run_text(cmd, cwd=None):
    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return completed.stdout.strip()
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def module_version(name):
    """通过 import 检测模块版本（可能不准确，因为模块名和包名可能不同）"""
    try:
        module = __import__(name)
        return getattr(module, "__version__", "UNKNOWN")
    except Exception as exc:
        return f"MISSING: {type(exc).__name__}: {exc}"


def pip_package_version(name):
    """通过 importlib.metadata 检测 pip 安装的包版本（更准确）"""
    try:
        from importlib.metadata import version, PackageNotFoundError
        return version(name)
    except PackageNotFoundError:
        return f"MISSING: PackageNotFoundError: {name}"
    except Exception as exc:
        return f"MISSING: {type(exc).__name__}: {exc}"


def torch_details():
    try:
        import torch

        return {
            "version": getattr(torch, "__version__", "UNKNOWN"),
            "cuda": getattr(torch.version, "cuda", None),
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def resolve_git_commit(project_root):
    if not project_root:
        return None
    output = run_text(["git", "rev-parse", "HEAD"], cwd=project_root)
    return output if output and not output.startswith("ERROR:") else None


def build_env_payload(project_root=None):
    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            "torch": pip_package_version("torch"),
            "triton": pip_package_version("triton"),
            "flagtree": pip_package_version("flagtree"),
            "flaggems": pip_package_version("flaggems"),
            "vllm": pip_package_version("vllm"),
            "pytest": pip_package_version("pytest"),
            "matplotlib": pip_package_version("matplotlib"),
            "openpyxl": pip_package_version("openpyxl"),
        },
        "torch": torch_details(),
        "cuda": {
            "nvidia_smi": run_text(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"]),
            "ldconfig_cutensor": run_text(["bash", "-lc", "ldconfig -p | grep -i cutensor || true"]),
        },
        "git_commit": resolve_git_commit(project_root),
        "working_directory": str(project_root) if project_root else None,
    }


def write_env_json(output_path, project_root=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_env_payload(project_root=project_root)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
