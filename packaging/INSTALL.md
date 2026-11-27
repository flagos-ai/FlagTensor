# Installing FlagTensor packages

FlagTensor is split into a backend-neutral Python package and NVIDIA native
packages:

- `python3-flagtensor`: Python operators and backend registration.
- `libflagtensor-nvidia`: CUDA C++ operator runtime and Triton kernel sources.
- `libflagtensor-nvidia-dev`: headers and CMake package files.
- `python3-flagtensor-nvidia`: optional Python bindings for the C++ runtime.

The native packages consume `libtriton-jit-nvidia` from the FlagOS package
repository. Their PyTorch, Triton, Python, and CUDA versions must match the
matrix used to build that runtime. The packaging build therefore uses the same
CUDA 12.8 / Python 3.12 / PyTorch 2.10 environment as libtriton_jit.
The RPM build uses the following distribution-specific environments:

| RPM distribution | Base image | Python | Build PyTorch | Triton |
| --- | --- | --- | --- | --- |
| EL9 | `nvidia/cuda:12.6.0-devel-rockylinux9` | 3.9 | 2.8.0 | 3.4.0 |
| openEuler 24.03 | `openeuler/cuda:13.0.0-oe2403lts` | 3.11 | 2.8.0+cu128 | 3.4.0 |
| Fedora 43 | `fedora:43` with CUDA 13.2 | 3.14 | 2.9.0+cu128 | 3.5.0 |

Pass the base image as the first argument to `packaging/rpm/build-flagtensor-rpm.sh`.
FlagTensor needs the CUDA-enabled torch headers and libraries during compilation.
The libtriton-jit packages must match the distribution, Python ABI and torch version.
The RPM development package is named `libflagtensor-nvidia-devel`.
The fallback package repository follows the selected distribution.
Direct Docker builds can override it with the `FLAGOS_YUM` build argument.

GPU-enabled PyTorch and Triton remain vendor runtime dependencies. They are not
replaced with distro `python3-torch`, which is generally CPU-only. On the DEB
target, matplotlib must also come from the Python 3.12 vendor environment:
Ubuntu 22.04's package pulls in a CPython 3.10 NumPy extension that is not ABI
compatible with the validated runtime.

For pre-publication validation, place the runtime and development packages from
libtriton_jit CI under `packaging/debian/local-deps/` or
`packaging/rpm/local-deps/`. These binary files are ignored by Git; normal CI
installs the same packages from FlagOS Nexus.

Fedora/openEuler builds need local libtriton-jit packages until those packages
are published to the corresponding repositories. Installation and consumer
checks without a GPU do not validate GPU execution.

On openEuler 24.03, `python3-openpyxl` is a recommended dependency because
the distribution repositories do not provide it. Native operators and CSV
output do not need it. Install `openpyxl` in the Python environment before
using XLSX report export.
