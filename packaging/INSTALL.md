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
The RPM build likewise matches its CUDA 12.6 / Python 3.9 / PyTorch 2.8 /
Triton 3.4 ABI matrix.

GPU-enabled PyTorch and Triton remain vendor runtime dependencies. They are not
replaced with distro `python3-torch`, which is generally CPU-only. On the DEB
target, matplotlib must also come from the Python 3.12 vendor environment:
Ubuntu 22.04's package pulls in a CPython 3.10 NumPy extension that is not ABI
compatible with the validated runtime.

For pre-publication validation, place the runtime and development packages from
libtriton_jit CI under `packaging/debian/local-deps/` or
`packaging/rpm/local-deps/`. These binary files are ignored by Git; normal CI
installs the same packages from FlagOS Nexus.
