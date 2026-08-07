[<img width="2182" height="602" alt="github+banner-20260130" src="https://github.com/flagos-ai/FlagGems/blob/master/.github/assets/banner-20260130.png" />](https://flagos.io/)

[中文版](./README_cn.md) | English

<div align="right">
  <a href="https://www.linkedin.com/company/flagos-community" target="_blank">
    <img src="https://github.com/flagos-ai/FlagGems/blob/master/docs/assets/Linkedin.png" alt="LinkedIn" width="32" height="32" />
  </a>

  <a href="https://www.youtube.com/@FlagOS_Official" target="_blank">
    <img src="https://github.com/flagos-ai/FlagGems/blob/master/docs/assets/youtube.png" alt="YouTube" width="32" height="32" />
  </a>

  <a href="https://x.com/FlagOS_Official" target="_blank">
    <img src="https://github.com/flagos-ai/FlagGems/blob/master/docs/assets/x.png" alt="X" width="32" height="32" />
  </a>

  <a href="https://www.facebook.com/flagosglobalcommunity/" target="_blank">
    <img src="https://github.com/flagos-ai/FlagGems/blob/master/docs/assets/Facebook.png" alt="Facebook" width="32" height="32" />
  </a>

  <a href="https://discord.com/invite/ubqGuFMTNE" target="_blank">
    <img src="https://github.com/flagos-ai/FlagGems/blob/master/docs/assets/discord.png" alt="Discord" width="32" height="32" />
  </a>
</div>

## About

FlagTensor is part of [FlagOS](https://flagos.io/), a fully open-source system software stack
designed to unify the model–system–chip layers and foster an open and collaborative ecosystem.
It enables a "develop once, run anywhere" workflow across diverse AI accelerators,
unlocking hardware performance, eliminating fragmentation among AI chipset-specific software stacks,
and substantially lowering the cost of porting and maintaining AI workloads.

FlagTensor is a high-performance tensor-primitive library implemented in [Triton](https://github.com/openai/triton) language.
It provides optimized implementations of common tensor primitives (unary, binary, and tensor contraction operations)
benchmarked against [cuTensor](https://developer.nvidia.com/cutensor) baselines, delivering reference-level
correctness with competitive performance across diverse GPU architectures.

Built on [FlagTree](https://github.com/flagos-ai/FlagTree) (a FlagOS-maintained Triton fork supporting
multiple hardware backends), FlagTensor offers a vendor-agnostic operator interface with pluggable backend support.

## Features

- Comprehensive collection of tensor primitives: unary (28 ops), binary (4 ops), contraction (6 ops)
- Hand-optimized Triton kernels with per-architecture autotune (Ampere, Hopper)
- Correctness validated against CPU-FP64 golden reference
- Performance benchmarked against cuTensor baselines
- Vendor-agnostic backend abstraction (15 vendors registered)
- Architecture-specific kernel specialization (e.g., `_nvidia/hopper/`, `_nvidia/ampere/`)
- Per-operator test infrastructure with pytest marks and JSON result recording
- Multi-GPU parallel test runner with live progress display
- CI-ready: quality gates (lint/format), correctness & performance pipelines

For a complete list of operators and their maturity stages, see [conf/operators.yaml](conf/operators.yaml).

## Getting Started

Refer to the [Environment Setup Guide](docs/environment.md) for a complete installation walkthrough.

**Quick start on NVIDIA A100:**

```bash
# 1. Install PyTorch
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 2. Install cuTensor
pip install cutensor-cu12
ln -sf $(python3 -c "import cutensor; print(cutensor.__path__[0])")/lib/libcutensor.so.2 \
  /usr/lib/x86_64-linux-gnu/libcutensor.so

# 3. Install FlagTree (Triton fork)
pip install --no-cache-dir \
  --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple \
  --trusted-host=resource.flagos.net \
  "flagtree==0.4.0+3.3" --no-deps

# 4. Install FlagTensor
pip install -e . --no-deps
```

## Usage

```python
import torch
import flagtensor

# Element-wise operations
x = torch.randn(1024, device="cuda", dtype=torch.float32)
y = flagtensor.abs(x)
z = flagtensor.relu(x)
w = flagtensor.sigmoid(x)

# Binary operations
a = torch.randn(1024, device="cuda")
b = torch.randn(1024, device="cuda")
c = flagtensor.add(a, b)

# Tensor contraction
m = torch.randn(64, 32, device="cuda")
n = torch.randn(32, 48, device="cuda")
r = flagtensor.gett(m, n)
```

## Running Tests

```bash
# Single operator correctness test
pytest tests/unary/test_abs.py -v

# Record test results as JSON (using CPU-FP64 reference)
pytest tests/unary/test_abs.py --ref cpu --record json --output results.json

# Multi-GPU test runner (from YAML registry)
python tools/run_tests.py --stages stable --gpus 0,1

# Extract operator marks
python tools/get_marks.py --stage stable --output ops.txt

# Benchmark with recording
pytest benchmark/test_unary_perf.py -m abs \
  --mode kernel --level core --record log

# Parse benchmark summary
python tools/summary_for_plot.py result-*.log
```

## Project Structure

```
FlagTensor
├── src/flagtensor/            # Python source
│   ├── ops/                   # Operator implementations (CUTENSOR_OP_*.py)
│   ├── utils/                 # Utility functions & kernel builders
│   ├── runtime/               # Runtime support
│   │   ├── backend/           # Vendor & architecture backends (_nvidia/, _ascend/, ...)
│   │   └── common.py          # Vendor enumeration & capability constants
│   ├── testing/               # Testing utilities (assertions, shapes, dtypes)
│   ├── fused/                 # Fused operators
│   └── modules/               # Module implementations
├── tests/                     # Per-operator correctness tests
│   ├── unary/test_<op>.py     # 28 unary operator tests
│   ├── binary/test_<op>.py    # 4 binary operator tests
│   ├── contraction/           # Contraction operator tests
│   └── sparse/                # Sparse operator tests
├── benchmark/                 # Performance tests
│   ├── consts.py              # Dtypes, shapes, metrics definitions
│   └── test_<category>_perf.py
├── tools/                     # CLI tooling
│   ├── run_tests.py           # Multi-GPU test runner
│   ├── get_marks.py           # Extract pytest marks from YAML
│   └── summary_for_plot.py    # Parse & aggregate benchmark logs
├── conf/
│   └── operators.yaml         # Operator registry (authoritative test entry point)
├── docs/                      # Documentation
├── .github/workflows/         # CI/CD pipelines
├── LICENSE
├── README.md
└── pyproject.toml
```

## Supported Operators

| Category | Operators | Status |
|----------|-----------|--------|
| **Unary** | abs, acos, acosh, asin, asinh, atan, atanh, ceil, conj, cos, cosh, exp, floor, identity, log, mish, neg, rcp, relu, sigmoid, sin, sinh, soft_plus, soft_sign, sqrt, swish, tan, tanh | stable |
| **Binary** | add, max, min, mul | stable |
| **Contraction** | gett, tgett, ttgt, tensor_contraction_trinary, trinary_generic | stable |
| **Sparse** | block_sparse_tensor_contraction | experimental |

## Contribution

- If you are interested in contributing to the FlagTensor project, please refer to
  the [contribution guide](./CONTRIBUTING.md). Any contributions would be highly appreciated.
- Please file an issue for feature requests or bug reports.
- Drop us an email at <a href="mailto:contact@flagos.io">contact@flagos.io</a> when you have questions or suggestions to share.

## Citation

If you find our work useful, please consider citing our project:

```bibtex
@misc{flagtensor2025,
    title={FlagOS/FlagTensor: A high-performance tensor-primitive library benchmarked against cuTensor},
    url={https://github.com/flagos-ai/FlagTensor},
    journal={GitHub},
    author={The FlagOS contributors},
    year={2025}
}
```

## Related Projects

- [FlagGems](https://github.com/flagos-ai/FlagGems) — General-purpose Triton operator library (500+ operators)
- [FlagTree](https://github.com/flagos-ai/FlagTree) — Multi-backend Triton fork maintained by FlagOS

## License

The FlagTensor project is licensed under the [Apache License (Version 2.0)](./LICENSE).
