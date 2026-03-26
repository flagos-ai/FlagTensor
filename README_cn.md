[中文版|[English](./README.md)]

## 介绍

FlagTensor 是 [FlagOS](https://flagos.io/) 的一部分。FlagTensor 是一个面向 GPU 后端的张量原语计算库，提供常见张量原语（如一元、二元与张量缩并）的高性能实现，并支持与 cuTensor baselines 的正确性和性能对比。

FlagTensor 是一个使用 OpenAI 推出的 [Triton 编程语言](https://github.com/openai/triton) 实现的高性能张量原语库。

## 特性

- 张量原语性能调优
- Triton kernel 调用优化
- 多后端支持
- 支持常见张量原语（张量一元、二元运算，待支持张量缩并等操作）

## 快速安装

### 安装依赖

```shell
pip install -U pip setuptools wheel
pip install torch triton pytest pyyaml matplotlib openpyxl
```

### 安装 FlagTensor

```shell
git clone https://github.com/flagos-ai/FlagTensor.git
cd FlagTensor
pip install -e .
```

## 使用示例

```python
import torch
import flagtensor

# 创建张量
x = torch.randn(1024, device="cuda", dtype=torch.float32)

# 应用 ReLU 算子
y = flagtensor.relu(x)
```

本项目采用 [Apache (Version 2.0) License](./LICENSE) 授权许可。