[中文版](./CONTRIBUTING_cn.md)

# FlagTensor Contributor's Guide

Thank you for your interest in FlagTensor! We use GitHub to host code, manage issues and pull requests.
Before contributing, please read the following guidelines.

## 1. Bug Report

Please report bugs using GitHub's Issues. When reporting bugs, please provide:

- a brief summary
- steps to reproduce
- and be specific!
- sample code that triggers the issue (very helpful)

## 2. Code Contribution

In pull requests, contributors should describe what changed and why. Please also provide test cases if applicable.
Pull requests require approvals from **one member** before merging.
Additionally, they must pass continuous integration checks:

- **Quality gate** (`quality-gate.yaml`): pre-commit hooks (black, isort, flake8, clang-format), build check, registry consistency
- **Correctness tests**: All per-operator tests must pass against CPU-FP64 reference
- **Performance tests**: Benchmark should not regress from baseline

## 3. Adding a New Operator

When adding a new operator, you need to create or update:

1. **Operator implementation**: `src/flagtensor/ops/CUTENSOR_OP_<NAME>.py`
2. **Correctness test**: `tests/<category>/test_<name>.py` (one file per operator)
3. **Performance test**: update or create `benchmark/test_<category>_perf.py`
4. **Register in YAML**: add entry to `conf/operators.yaml`

### Test file template

```python
import pytest
import torch
from tests.accuracy_utils import gems_assert_close, to_reference
from tests.accuracy_utils import POINTWISE_SHAPES, FLOAT_DTYPES
from flagtensor import your_op

@pytest.mark.your_op
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_your_op_correctness(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = torch.randn(shape, device="cuda", dtype=dtype)
    ref = to_reference(x, upcast=True)
    ref_out = torch.<reference_op>(ref)
    y = your_op(x)
    gems_assert_close(y, ref_out, dtype)
```

### YAML entry template

```yaml
- id: your_op
  name: your_op
  category: unary
  for: [your_op]
  labels: [cuTensor, pointwise]
  kind: [Math]
  stages:
    - stable: '1.0'
  description: Brief description of what the operator does.
  python_api: flagtensor.your_op
  impl_file: src/flagtensor/ops/CUTENSOR_OP_YOUR_OP.py
  correctness_test: tests/unary/test_your_op.py
  benchmark_test: benchmark/test_unary_perf.py
  correctness_mark: your_op
  benchmark_mark: your_op
  benchmark_modes:
    - operator
  status: stable
```

## 4. Adding a New Hardware Backend

FlagTensor uses a plugin-based backend architecture. To add a new vendor:

1. Create `src/flagtensor/runtime/backend/_<vendor>/__init__.py`:

```python
from backend_utils import VendorInfoBase

vendor_info = VendorInfoBase(
    vendor_name="your_vendor",
    device_name="your_device",      # torch.{device_name}
    device_query_cmd="your-smi",    # hardware detection command
    dispatch_key=None,              # PyTorch dispatch key (or "PrivateUse1")
)
```

2. (Optional) Create architecture-specific directories and operator overrides under the same vendor directory.

## 5. Development Setup

```bash
# Clone and install
git clone https://github.com/flagos-ai/FlagTensor.git
cd FlagTensor
pip install -e . --no-deps

# Enable pre-commit hooks
pip install pre-commit
pre-commit install
```

## Project Structure

```
FlagTensor
├── src/flagtensor/            # Python source code
│   ├── ops/                   # Single operator implementations
│   ├── utils/                 # Python utilities
│   ├── runtime/               # Runtime support & backend abstraction
│   ├── testing/               # Testing utilities
│   ├── fused/                 # Fused operators
│   └── modules/               # Module implementations
├── tests/                     # Per-operator correctness tests
├── benchmark/                 # Performance tests
├── tools/                     # CLI tooling (run_tests, get_marks, etc.)
├── conf/
│   └── operators.yaml         # Operator registry (authoritative list)
├── docs/                      # Documentation
├── LICENSE
├── README.md
├── CONTRIBUTING.md
└── pyproject.toml
```

## License

Any contributions you make will be under the [Apache License (Version 2.0)](./LICENSE).
