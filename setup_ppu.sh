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

# FlagTensor PPU setup script — transparent, self-contained installation.
# PPU counterpart of setup.sh (which targets NVIDIA + cuTensor).
#
# Prerequisites (already satisfied on the official PPU image):
#   * Python 3.10+
#   * Alibaba PPU driver + PPU_SDK at /usr/local/PPU_SDK
#   * PPU-specific PyTorch (torch 2.9.0, CUDA-compatible build)
#   * PPU-specific Triton (triton 3.5.0+ppu2.0.0.oe)
#
# What this script does:
#   1. Verifies the PPU environment (PPU_SDK, ppu-smi, torch, triton)
#   2. Installs Python deps that the PPU image does not pre-ship
#      (matplotlib / openpyxl / PyYAML / pytest — skipped if already present)
#   3. Installs FlagTensor itself with --no-deps (the PPU torch build is
#      newer than the <2.7.0 cap declared in pyproject.toml, so we must
#      bypass pip's dependency resolver to avoid a destructive downgrade)
#   4. Smoke-tests that flagtensor imports and detects vendor=ppu
#
# What this script does NOT do (intentional):
#   * Install cuTensor — PPU has no cuTensor equivalent; the PPU baseline
#     is PyTorch native ops backed by vendor acblas/acdnn kernels.
#     See docs/ppu_adaptation.md for the full baseline selection rationale.
#   * Install FlagTree — PPU ships its own PPU-specific Triton build;
#     FlagTree (the FlagOS Triton fork) is not needed on PPU.
#   * Install cuda-bindings / nvmath-python — NVIDIA-specific, not used.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[93m'; NC='\033[0m'

step_ok()   { printf "  %s $GREEN[OK]$NC\n" "$1"; }
step_fail() { printf "  %s $RED[FAILED]$NC\n" "$1"; exit 1; }
step_skip() { printf "  %s $YELLOW[SKIP]$NC %s\n" "$1" "$2"; }

echo "=== FlagTensor PPU Setup ==="

# ---------------------------------------------------------------------------
# 1. PPU SDK + driver
# ---------------------------------------------------------------------------
printf "Checking PPU_SDK ... "
if [ -n "${PPU_SDK:-}" ] && [ -d "$PPU_SDK" ]; then
  step_ok "$PPU_SDK"
else
  step_fail "PPU_SDK env var not set or directory missing (expected /usr/local/PPU_SDK)"
fi

printf "Checking PPU version ... "
if [ -n "${PPU_VERSION:-}" ]; then
  step_ok "$PPU_VERSION"
else
  step_skip "PPU_VERSION not set" "(non-fatal, continuing)"
fi

printf "Checking ppu-smi ... "
if command -v ppu-smi &>/dev/null; then
  ppu_name=$(ppu-smi 2>/dev/null | grep -oE 'PPU-[A-Z0-9]+' | head -1)
  step_ok "${ppu_name:-detected}"
else
  step_fail "ppu-smi not on PATH (expected /usr/local/PPU_SDK/ppu-smi/bin/ppu-smi)"
fi

# ---------------------------------------------------------------------------
# 2. Python
# ---------------------------------------------------------------------------
printf "Checking Python ... "
python_version=$(python3 --version 2>/dev/null | awk '{print $NF}')
if [[ "$python_version" =~ ^3\.(10|11|12|13) ]]; then
  step_ok "$python_version"
else
  printf "  %s $RED[UNSUPPORTED]$NC (need 3.10+)\n" "$python_version"
  exit 1
fi

# ---------------------------------------------------------------------------
# 3. PyTorch (pre-installed on PPU image — verify, don't reinstall)
# ---------------------------------------------------------------------------
printf "Checking PyTorch ... "
torch_ver=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [ -z "$torch_ver" ]; then
  step_fail "PyTorch not installed (PPU image should pre-ship it)"
fi
printf "Checking torch.cuda availability ... "
cuda_ok=$(python3 -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null || echo "no")
if [ "$cuda_ok" = "yes" ]; then
  dev_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "?")
  step_ok "torch $torch_ver, $dev_name"
else
  step_fail "torch.cuda.is_available() = False (PPU driver/SDK mismatch?)"
fi

# ---------------------------------------------------------------------------
# 4. Triton (pre-installed on PPU image — verify, don't reinstall)
# ---------------------------------------------------------------------------
printf "Checking Triton ... "
triton_ver=$(python3 -c "import triton; print(triton.__version__)" 2>/dev/null || echo "")
if [ -n "$triton_ver" ]; then
  step_ok "triton $triton_ver"
else
  step_fail "Triton not installed (PPU image should pre-ship triton 3.5.0+ppu2.0.0.oe)"
fi

# ---------------------------------------------------------------------------
# 5. Upgrade pip
# ---------------------------------------------------------------------------
printf "Upgrading pip ... "
python3 -m pip install --upgrade pip -q && step_ok "done"

# ---------------------------------------------------------------------------
# 6. Python deps (install only if missing — PPU image may pre-ship some)
# ---------------------------------------------------------------------------
# NOTE: We deliberately do NOT install cuTensor, FlagTree, cuda-bindings,
# or nvmath-python here — those are NVIDIA-only. See script header.
install_if_missing() {
  local pkg="$1"
  local import_name="${2:-$1}"
  if python3 -c "import $import_name" 2>/dev/null; then
    local ver=$(python3 -c "import $import_name; print(getattr($import_name, '__version__', '?'))" 2>/dev/null || echo "?")
    step_skip "$pkg" "already installed ($ver)"
  else
    printf "  Installing %s ... " "$pkg"
    python3 -m pip install -q "$pkg" && step_ok "done" || step_fail "pip install $pkg failed"
  fi
}

echo "--- Python deps ---"
install_if_missing "PyYAML" "yaml"
install_if_missing "pytest"
install_if_missing "matplotlib"
install_if_missing "openpyxl"

# ---------------------------------------------------------------------------
# 7. FlagTensor (editable install with --no-deps)
# ---------------------------------------------------------------------------
# CRITICAL: PPU ships torch 2.9.0, but pyproject.toml declares
# "torch>=2.2.0,<2.7.0". Using `pip install -e .` without --no-deps would
# trigger pip's dependency resolver to DOWNGRADE torch to <2.7.0, which
# would destroy the PPU-specific torch build. We therefore install with
# --no-deps and rely on the pre-shipped torch/triton/PyYAML/etc.
printf "Installing FlagTensor (editable, --no-deps) ... "
if python3 -c "import flagtensor" 2>/dev/null; then
  ft_ver=$(python3 -c "import flagtensor; print(getattr(flagtensor, '__version__', '0.1.0'))" 2>/dev/null || echo "0.1.0")
  step_skip "flagtensor" "already installed ($ft_ver), reinstalling editable"
fi
python3 -m pip install -e . --no-deps -q && step_ok "done"

# ---------------------------------------------------------------------------
# 8. Smoke test: import + vendor detection
# ---------------------------------------------------------------------------
printf "Smoke-testing flagtensor import + vendor detection ... "
smoke=$(python3 -c "
import flagtensor
v = flagtensor.runtime.device.vendor_name
n = flagtensor.runtime.device.name
c = flagtensor.runtime.device.device_count
assert v == 'ppu', f'vendor expected ppu, got {v}'
assert n == 'cuda', f'device name expected cuda, got {n}'
assert c >= 1, f'device_count expected >=1, got {c}'
from flagtensor.benchmark_core import vendor_baseline_available
assert vendor_baseline_available(), 'PPU baseline should be available'
print(f'vendor={v}, device={n}, count={c}, baseline=available')
" 2>&1) && step_ok "$smoke" || step_fail "$smoke"

echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  # Run all 36 operators (stable + experimental + active):"
echo "  python tools/run_tests.py --stages all --gpus 0,1 --output-dir result --color never --dump-output"
echo ""
echo "  # Or stable-only (34 operators, default):"
echo "  python tools/run_tests.py --stages stable --gpus 0,1 --output-dir result"
echo ""
echo "See docs/ppu_adaptation.md for the full PPU adaptation report."
