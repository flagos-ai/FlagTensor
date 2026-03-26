[English|[中文版](./README_cn.md)]

## Introduction

FlagTensor is part of [FlagOS](https://flagos.io/). FlagTensor is a tensor-primitive library oriented toward multiple hardware backends. It provides high-performance implementations of common tensor primitives (for example, unary, binary, and contraction operations), and supports correctness and performance comparisons against cuTensor baselines.

FlagTensor is a high-performance tensor-primitive library implemented with the [Triton programming language](https://github.com/openai/triton) launched by OpenAI.

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