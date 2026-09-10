#!/bin/bash

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

# run_tests_kunlunxin.sh — Wrapper to run FlagTensor tests on Kunlunxin XPU.
#
# This script sets up the environment variables required by the Kunlunxin
# XPU software stack (torch_xmlir plugin + XPU runtime) and then delegates
# to the standard tools/run_tests.py, so the test logic / result format /
# vendor gating logic is completely unchanged.
#
# Usage:
#   ./tools/run_tests_kunlunxin.sh [run_tests.py args...]
#
# Examples:
#   ./tools/run_tests_kunlunxin.sh --stages all --gpus 0 --output-dir results
#   ./tools/run_tests_kunlunxin.sh --ops CUTENSOR_OP_ABS,CUTENSOR_OP_ADD

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Kunlunxin XPU runtime environment ---
# XPU SDK runtime libraries (libxpurt, libxpucuda, libxpuml, ...)
export XPU_SDK_SO_DIR="${XPU_SDK_SO_DIR:-/opt/xre/so}"
# XCCL collective communication library
export XCCL_SO_DIR="${XCCL_SO_DIR:-/opt/xccl/so}"
# CUDA 11.7 compat libraries (curand, cufft, cusparse, cusolver, cupti, ...)
export CUDA_COMPAT_LIB_DIR="${CUDA_COMPAT_LIB_DIR:-/opt/cuda-libs/lib}"
export CUDA_COMPAT_CUPTI_DIR="${CUDA_COMPAT_CUPTI_DIR:-/opt/cuda-libs/cupti}"
# Python env with torch + torch_xmlir + flagtree
export KUNLUNXIN_PYTHON="${KUNLUNXIN_PYTHON:-/opt/kunlunxin-env/bin/python}"

# Build LD_LIBRARY_PATH so torch can find all the CUDA compat libs + XPU libs
TORCH_LIB_DIR="$("$KUNLUNXIN_PYTHON" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${XPU_SDK_SO_DIR}:${XCCL_SO_DIR}:${CUDA_COMPAT_LIB_DIR}:${CUDA_COMPAT_CUPTI_DIR}:${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# Build-time library search path (for triton XPU backend JIT compilation)
export LIBRARY_PATH="${XPU_SDK_SO_DIR}:${XCCL_SO_DIR}:${LIBRARY_PATH:-}"

# XPU runtime flags (required by torch_xmlir plugin)
export XPU_FORCE_USERMODE_LAUNCH=1
export CUDART_DUMMY_REGISTER=1
export CUDART_MODULE_LOADING=LAZY
export PYTHONUNBUFFERED=1

# Use operator mode for benchmark timing (triton.testing.do_bench on XPU
# has known issues with CUDA event timing; the host-loop timing used by
# operator mode is reliable and matches the FlagGems convention on
# non-NVIDIA backends).
export FLAGTENSOR_BENCHMARK_MODE=operator

# Ensure the Kunlunxin env's pytest (python 3.8 + torch_xmlir) takes
# precedence over any system-wide pytest.
export PATH="/opt/kunlunxin-env/bin:${PATH}"

cd "$PROJECT_ROOT"

# Pass through all arguments to run_tests.py
exec "$KUNLUNXIN_PYTHON" tools/run_tests.py "$@"
