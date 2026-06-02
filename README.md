[English|[中文版](./README_cn.md)]

## Introduction

FlagTensor is part of [FlagOS](https://flagos.io/). FlagTensor is a tensor-primitive library oriented toward multiple hardware backends. It provides high-performance implementations of common tensor primitives (for example, unary, binary, and contraction operations), and supports correctness and performance comparisons against cuTensor baselines.

FlagTensor is a high-performance tensor-primitive library implemented with the [Triton programming language](https://github.com/openai/triton) launched by OpenAI.

## CI workflows

This repository provides two GitHub Actions workflows under `.github/workflows`:

- `flagtensor-ci`: split into `correctness` and `perf` jobs for smoke-style automated validation.
- `flagtensor-weekly`: runs the weekly correctness and benchmark pipeline from an operator list.

## Operator registry

The authoritative operator list lives in `conf/operators.yaml`.

It is used to track:

- operator category
- implementation path
- correctness / benchmark entry points
- supported benchmark modes
- blocked operators and skip reasons

By default, the local CI and weekly runners discover operators from this registry.

## Development quality gates

Install and enable pre-commit locally:

```bash
pip install pre-commit
pre-commit install
```

The repository ships a `.pre-commit-config.yaml` with YAML, formatting, import ordering, lint, and C/C++ formatting hooks.

### Benchmark modes

Both workflows support the benchmark `mode` input:

- `kernel`
- `operator`

The default mode is `kernel`.

### How to use

- Trigger `flagtensor-ci` from `workflow_dispatch` when you want a quick automated check of the currently covered operators.
- Trigger `flagtensor-weekly` from `workflow_dispatch` when you want to run the weekly-style multi-operator pipeline.
- For `flagtensor-weekly`, you can optionally provide a custom operator list file; otherwise the workflow generates one from the discovered tests.

Run CI correctness locally:

```bash
python tools/run_flagtensor_ci.py --smoke --run-correctness --exclude-op tensor_contraction_trinary --mode kernel --results-dir ci_results_correctness
```

Run CI perf locally in kernel mode:

```bash
python tools/run_flagtensor_ci.py --smoke --run-perf --exclude-op tensor_contraction_trinary --mode kernel --results-dir ci_results_perf
```

Run CI perf locally in operator mode:

```bash
python tools/run_flagtensor_ci.py --smoke --run-perf --exclude-op tensor_contraction_trinary --mode operator --results-dir ci_results_perf_operator
```

Run weekly locally in kernel mode:

```bash
python tools/run_flagtensor_weekly.py --project-root . --gpus 0 --mode kernel --results-dir weekly_results_ci
```

Run weekly locally in operator mode:

```bash
python tools/run_flagtensor_weekly.py --project-root . --gpus 0 --mode operator --results-dir weekly_results_ci_operator
```

Run weekly with an explicit operator list (optional; generated from registry if omitted):

```bash
python tools/run_flagtensor_weekly.py --project-root . --op-list my_ops.txt --gpus 0 --mode kernel --results-dir weekly_results_ci
```

## Features

- Tensor primitives have undergone performance tuning
- Triton kernel call optimization
- Flexible multi-backend support mechanism
- Support for common tensor primitives

## Quick Installation

### Install Dependencies

```shell
pip install -U pip setuptools wheel
pip install torch triton pytest pyyaml matplotlib openpyxl
```

### Install FlagTensor

```shell
git clone https://github.com/flagos-ai/FlagTensor.git
cd FlagTensor
pip install -e .
```

## Usage Example

```python
import torch
import flagtensor

# Create a tensor
x = torch.randn(1024, device="cuda", dtype=torch.float32)

# Apply ReLU operator
y = flagtensor.relu(x)
```

This project is licensed under the [Apache (Version 2.0) License](./LICENSE).
