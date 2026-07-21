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

# FlagTensor CI setup script — transparent, self-contained installation.
# Follows FlagOS CI convention: clone → setup.sh → run tests.
# Only assumptions: Python 3.10+, NVIDIA driver, CUDA runtime.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'

step_ok()  { printf "  %s $GREEN[OK]$NC\n" "$1"; }
step_fail(){ printf "  %s $RED[FAILED]$NC\n" "$1"; exit 1; }

echo "=== FlagTensor CI Setup ==="

# ---- Python ----
printf "Checking Python ... "
python_version=$(python3 --version 2>/dev/null | awk '{print $NF}')
if [[ "$python_version" =~ ^3\.(10|11|12) ]]; then
  step_ok "$python_version"
else
  printf "  %s $RED[UNSUPPORTED]$NC (need 3.10-3.12)\n" "$python_version"
  exit 1
fi

# ---- GPU ----
printf "Checking GPU ... "
if nvidia-smi &>/dev/null; then
  step_ok "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  step_fail "no NVIDIA GPU"
fi

# ---- Upgrade pip ----
printf "Upgrading pip ... "
python3 -m pip install --upgrade pip -q && step_ok "done"

# ---- cuTensor ----
printf "Installing cuTensor ... "
python3 -m pip install -q cutensor-cu12 && step_ok "done"
printf "Linking libcutensor.so ... "
if [ ! -f /usr/lib/x86_64-linux-gnu/libcutensor.so ]; then
  ln -sf "$(python3 -c 'import cutensor; print(cutensor.__path__[0])')/lib/libcutensor.so.2" \
    /usr/lib/x86_64-linux-gnu/libcutensor.so 2>/dev/null && step_ok "done" || step_ok "skipped (non-root)"
else
  step_ok "exists"
fi

# ---- FlagTree ----
printf "Installing FlagTree ... "
python3 -m pip install --no-cache-dir -q \
  --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple \
  --trusted-host=resource.flagos.net \
  "flagtree==0.4.0+3.3" --no-deps && step_ok "done"

# ---- Python deps ----
printf "Installing Python deps ... "
python3 -m pip install -q \
  "cuda-bindings>=12.9.6,<13" nvmath-python matplotlib setuptools PyYAML && step_ok "done"

# ---- FlagTensor ----
printf "Installing FlagTensor ... "
python3 -m pip install -e . --no-deps -q && step_ok "done"

echo "=== Setup complete ==="
