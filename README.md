# FlagTensor 快速上手

FlagTensor 是一个高性能张量运算算子仓库，支持：

- 基于智源FlagTree开源编译器实现的张量运算，包括常见的张量一元、二元运算，以及tensor contraction（待支持）等操作
- 与 cuTensor baseline 的正确性/性能对比
- CI/Weekly 自动化测试与报告生成

## 1. 环境与安装

要求：

- Python 3.10+
- NVIDIA GPU + CUDA
- PyTorch / Triton / pytest
- 可选：`libcutensor.so`（用于 baseline 对比）

安装：

```bash
cd /path/to/FlagTensor
pip install -e .
```

## 2. 一键跑 CI（推荐入口）

```bash
python tools/run_flagtensor_ci.py --op-list tools/ci_op_list.txt
```

快速冒烟：

```bash
python tools/run_flagtensor_ci.py --op-list tools/ci_op_list.txt --smoke
```

结果默认在 `ci_results/`，包含每个算子的日志和汇总文件（`summary.json` / `summary.md`）。

## 3. 生成 HTML 报告

```bash
python tools/generate_flagtensor_html_report.py \
  --smoke-results ci_results \
  --libtuner-results ci_results \
  --output ci_results/FlagTensor_CI_report.html \
  --title "FlagTensor CI 测试报告"
```

## 4. 跑单算子

正确性：

```bash
cd ctests
pytest -vs test_CUTENSOR_OP_RELU.py
```

性能：

```bash
cd benchmark
pytest -vs test_CUTENSOR_OP_RELU_perf.py
```

## 5. 常用目录

```text
src/flagtensor/   核心实现（ops/runtime/libtuner/cutensor）
ctests/           正确性测试
benchmark/        性能测试
tools/            CI/Weekly 脚本与报告生成
```

## 6. 新增算子最小流程

1. 在 `src/flagtensor/ops/` 新增算子实现
2. 在 `src/flagtensor/__init__.py` 导出函数
3. 在 `ctests/` 加正确性测试
4. 在 `benchmark/` 加性能测试

---

