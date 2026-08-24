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

# FlagTensor CI setup script — transparent, self-contained, multi-backend.
# Follows FlagOS CI convention: clone → setup.sh → run tests.
#
# Usage:
#   ./setup.sh                          # auto-detect backend
#   ./setup.sh --backend nvidia         # NVIDIA GPU (CUDA + cuTensor baseline)
#   ./setup.sh --backend ppu            # Alibaba PPU (PPU_SDK bundled Triton)
#   ./setup.sh --backend iluvatar       # Iluvatar CoreX / 天数 (FlagTree Triton)
#
# Environment overrides (rarely needed):
#   FLAGTREE_PKG='flagtree==<spec>'     # override the FlagTree package spec
#   FLAGTENSOR_SKIP_VERIFY=1            # skip the post-install verify step
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

step_ok()  { printf "  %s $GREEN[OK]$NC\n" "$1"; }
step_fail(){ printf "  %s $RED[FAILED]$NC\n" "$1"; exit 1; }
step_warn(){ printf "  %s $YELLOW[WARN]$NC\n" "$1"; }

FLAGOS_PYPI="https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
FLAGOS_HOST="resource.flagos.net"

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
BACKEND="auto"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="${2:?--backend requires a value}"; shift 2 ;;
    --backend=*) BACKEND="${1#*=}"; shift ;;
    -h|--help)
      sed -n '17,29p' "$0"; exit 0 ;;
    *) step_fail "unknown argument: $1" ;;
  esac
done
BACKEND="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"

detect_backend() {
  if command -v nvidia-smi &>/dev/null; then echo "nvidia"; return; fi
  if command -v ppu-smi &>/dev/null || [[ -n "${PPU_SDK:-}" ]]; then echo "ppu"; return; fi
  if command -v ixsmi &>/dev/null || [[ -d /usr/local/corex ]]; then echo "iluvatar"; return; fi
  echo ""
}

if [[ "$BACKEND" == "auto" ]]; then
  BACKEND="$(detect_backend)"
  [[ -n "$BACKEND" ]] || step_fail "cannot auto-detect backend (need nvidia-smi, ppu-smi or ixsmi). Use --backend explicitly."
fi
case "$BACKEND" in
  nvidia|ppu|iluvatar) ;;
  *) step_fail "unsupported backend '$BACKEND' (expect nvidia, ppu or iluvatar)" ;;
esac

echo "=== FlagTensor CI Setup (backend: $BACKEND) ==="

# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------
printf "Checking Python ... "
python_version=$(python3 --version 2>/dev/null | awk '{print $NF}')
python_mm=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$python_version" =~ ^3\.(10|11|12) ]]; then
  step_ok "$python_version"
else
  printf "  %s $RED[UNSUPPORTED]$NC (need 3.10-3.12)\n" "$python_version"
  exit 1
fi

# ---------------------------------------------------------------------------
# Backend: GPU check + FlagTree package selection
# ---------------------------------------------------------------------------
FLAGTREE_SPEC=""
case "$BACKEND" in
  nvidia)
    printf "Checking GPU ... "
    if nvidia-smi &>/dev/null; then
      step_ok "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    else
      step_fail "no NVIDIA GPU"
    fi
    # Validated NVIDIA build: NVIDIA backend + triton.Config + triton.autotune.
    FLAGTREE_SPEC="flagtree==0.4.0+3.3"
    ;;

  ppu)
    printf "Checking GPU ... "
    if ppu-smi &>/dev/null; then
      step_ok "$(ppu-smi 2>/dev/null | grep -oE 'PPU[- ]?[A-Z0-9]+' | head -1 || echo "PPU device")"
    elif [[ -n "${PPU_SDK:-}" ]]; then
      step_ok "PPU_SDK=${PPU_SDK}"
    else
      step_fail "no PPU device (need ppu-smi or PPU_SDK)"
    fi
    # The validated PPU environment uses the Triton bundled in PPU_SDK
    # (3.5.0+ppu2.x). A FlagTree PPU build exists (flagtree==0.6.1+ppu3.6,
    # Python 3.12 only) but is NOT validated with FlagTensor yet; install it
    # explicitly via FLAGTREE_PKG if you want to experiment.
    FLAGTREE_SPEC=""
    ;;

  iluvatar)
    printf "Checking GPU ... "
    if ixsmi &>/dev/null; then
      step_ok "$(ixsmi -L 2>/dev/null | head -1 | sed 's/^GPU *[0-9]*: *//')"
    elif [[ -d /usr/local/corex ]]; then
      step_warn "ixsmi not found, assuming CoreX SDK at /usr/local/corex"
    else
      step_fail "no Iluvatar device (need ixsmi or /usr/local/corex)"
    fi
    # 0.4.0+iluvatar3.1 (cp310) and 0.6.1+iluvatar3.6 (cp312) both bundle
    # iluvatarTritonPlugin.so inside the wheel (validated on BI-V150).
    # 0.5.x iluvatar builds require an external plugin download and are
    # therefore not selected automatically.
    case "$python_mm" in
      3.10) FLAGTREE_SPEC="flagtree==0.4.0+iluvatar3.1" ;;
      3.12) FLAGTREE_SPEC="flagtree==0.6.1+iluvatar3.6" ;;
      *) step_fail "no FlagTree Iluvatar wheel for Python $python_mm (available: cp310: 0.4.0+iluvatar3.1, cp312: 0.6.1+iluvatar3.6)" ;;
    esac
    ;;
esac

# ---------------------------------------------------------------------------
# Upgrade pip
# ---------------------------------------------------------------------------
printf "Upgrading pip ... "
python3 -m pip install --upgrade pip -q && step_ok "done"

# ---------------------------------------------------------------------------
# Vendor math libraries (baseline dependency)
# ---------------------------------------------------------------------------
if [[ "$BACKEND" == "nvidia" ]]; then
  printf "Installing cuTensor ... "
  python3 -m pip install -q cutensor-cu12 && step_ok "done"
  printf "Linking libcutensor.so ... "
  if [ ! -f /usr/lib/x86_64-linux-gnu/libcutensor.so ]; then
    ln -sf "$(python3 -c 'import cutensor; print(cutensor.__path__[0])')/lib/libcutensor.so.2" \
      /usr/lib/x86_64-linux-gnu/libcutensor.so 2>/dev/null && step_ok "done" || step_ok "skipped (non-root)"
  else
    step_ok "exists"
  fi
fi
# PPU / Iluvatar baselines are PyTorch-native ops dispatched to the vendor
# libraries shipped with their SDKs — nothing extra to install.

# ---------------------------------------------------------------------------
# FlagTree (FlagOS Triton distribution)
# ---------------------------------------------------------------------------
FLAGTREE_SPEC="${FLAGTREE_PKG:-$FLAGTREE_SPEC}"
if [[ -n "$FLAGTREE_SPEC" ]]; then
  printf "Installing FlagTree (%s) ... " "$FLAGTREE_SPEC"
  python3 -m pip install --no-cache-dir -q \
    --index-url="$FLAGOS_PYPI" \
    --trusted-host="$FLAGOS_HOST" \
    "$FLAGTREE_SPEC" --no-deps && step_ok "done"
else
  step_warn "no FlagTree build installed for backend '$BACKEND'; using the SDK-bundled Triton"
fi

# ---------------------------------------------------------------------------
# Python deps
# ---------------------------------------------------------------------------
printf "Installing Python deps ... "
if [[ "$BACKEND" == "nvidia" ]]; then
  python3 -m pip install -q \
    "cuda-bindings>=12.9.6,<13" nvmath-python matplotlib setuptools PyYAML \
    pytest openpyxl && step_ok "done"
else
  python3 -m pip install -q \
    matplotlib setuptools PyYAML pytest openpyxl && step_ok "done"
fi

# ---------------------------------------------------------------------------
# FlagTensor
# ---------------------------------------------------------------------------
printf "Installing FlagTensor ... "
python3 -m pip install -e . --no-deps -q && step_ok "done"

# ---------------------------------------------------------------------------
# Iluvatar: make the FlagTree Triton shadowable by the CoreX SDK's bundled one
# ---------------------------------------------------------------------------
# The CoreX enable script (/usr/local/corex/enable) prepends the SDK's own
# dist-packages to PYTHONPATH, which would shadow the pip-installed FlagTree
# Triton. Generate a small env file that puts the pip site-packages first.
if [[ "$BACKEND" == "iluvatar" ]]; then
  printf "Generating iluvatar_env.sh ... "
  site_packages=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  cat > "$SCRIPT_DIR/iluvatar_env.sh" <<EOF
# Auto-generated by setup.sh (backend: iluvatar). Source this before running
# FlagTensor so the FlagTree Triton takes precedence over the CoreX SDK's
# bundled Triton (the SDK prepends its own dist-packages to PYTHONPATH).
#   source iluvatar_env.sh
_site="$site_packages"
case ":\${PYTHONPATH:-}:" in
  *":\$_site:"*) ;;
  *) export PYTHONPATH="\$_site\${PYTHONPATH:+:\$PYTHONPATH}" ;;
esac
# Only needed when FlagTree is installed into a non-default site directory
# (e.g. a venv); harmless for a standard system-python install.
export FLAGTREE_BACKEND_PLUGIN_LIB_DIR="\$_site/triton/_C"
EOF
  step_ok "done (source iluvatar_env.sh before run_tests.py)"
fi

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
if [[ "${FLAGTENSOR_SKIP_VERIFY:-0}" != "1" ]]; then
  printf "Verifying installation ... "
  if [[ "$BACKEND" == "iluvatar" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/iluvatar_env.sh"
  fi
  python3 - "$BACKEND" <<'PYEOF'
import sys

backend = sys.argv[1]

import flagtensor
from flagtensor.runtime import device

if device.vendor_name != backend:
    raise SystemExit(f"vendor mismatch: expected {backend}, detected {device.vendor_name}")

if backend == "iluvatar":
    import triton
    if "site-packages" not in triton.__file__:
        raise SystemExit(
            f"imported Triton is not the FlagTree one: {triton.__file__}\n"
            "Run `source iluvatar_env.sh` (or fix PYTHONPATH) before testing."
        )
    print(f"Triton (FlagTree): {triton.__version__} @ {triton.__file__}")
elif backend == "nvidia":
    from flagtensor.cutensor import CUTENSOR_AVAILABLE
    print(f"cuTensor available: {CUTENSOR_AVAILABLE}")

print(f"flagtensor OK, vendor = {device.vendor_name}, device = {device.name}")
PYEOF
  step_ok "done"
fi

echo "=== Setup complete (backend: $BACKEND) ==="
if [[ "$BACKEND" == "iluvatar" ]]; then
  echo "Next: source iluvatar_env.sh && python3 tools/run_tests.py --stages all --gpus 0"
else
  echo "Next: python3 tools/run_tests.py --stages all --gpus 0"
fi
