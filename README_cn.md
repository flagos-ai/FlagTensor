<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

[<img width="2182" height="602" alt="github+banner-20260130" src="https://github.com/flagos-ai/FlagGems/blob/master/.github/assets/banner-20260130.png" />](https://flagos.io/)

中文版 | [English](./README.md)

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

## 介绍

FlagTensor 是 [FlagOS](https://flagos.io/) 的一部分。
FlagOS 是一个面向多元AI芯片的开源、统一系统软件栈，旨在打通模型、系统与芯片层，
培育开放协作的生态系统。它支持"一次开发，多芯运行"的工作流，兼容多样化的 AI 加速芯片，
释放硬件性能潜力，消除各类 AI 芯片专用软件栈之间的碎片化问题，
并大幅降低大模型在多种 AI 硬件移植与维护的成本。

FlagTensor 是一个使用 [Triton 编程语言](https://github.com/openai/triton) 实现的高性能张量原语库。
它提供了常见张量原语（一元运算、二元运算、张量缩并）的优化实现，
并以 [cuTensor](https://developer.nvidia.com/cutensor) 为基线进行正确性和性能对比，
在不同 GPU 架构上提供参考级别的精度和具有竞争力的性能。

FlagTensor 基于 [FlagTree](https://github.com/flagos-ai/FlagTree)（FlagOS 维护的支持多后端的 Triton 分支），
提供了与厂商无关的算子接口和可插拔的后端支持。

## 特性

- 全面的张量原语覆盖：一元算子（28 个）、二元算子（4 个）、张量缩并（6 个）
- 手工优化的 Triton kernel，支持按架构自动调优（Ampere、Hopper）
- 以 CPU-FP64 为基准的正确性验证
- 以 cuTensor 为基线的性能对比
- 厂商无关的后端抽象（已注册 15 个厂商）
- 架构专用 kernel 特化（如 `_nvidia/hopper/`、`_nvidia/ampere/`）
- 每个算子独立测试文件，支持 pytest 标记和 JSON 结果录制
- 多 GPU 并行测试运行器，带实时进度显示
- CI 就绪：质量门禁（lint/format）、正确性与性能流水线

完整算子列表及各算子的成熟度阶段，参见 [conf/operators.yaml](conf/operators.yaml)。

## 快速入门

详细安装步骤请参阅 [环境搭建指南](docs/environment.md)。

**在 NVIDIA A100 上快速开始：**

```bash
# 1. 安装 PyTorch
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 2. 安装 cuTensor
pip install cutensor-cu12
ln -sf $(python3 -c "import cutensor; print(cutensor.__path__[0])")/lib/libcutensor.so.2 \
  /usr/lib/x86_64-linux-gnu/libcutensor.so

# 3. 安装 FlagTree（Triton 分支）
pip install --no-cache-dir \
  --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple \
  --trusted-host=resource.flagos.net \
  "flagtree==0.4.0+3.3" --no-deps

# 4. 安装 FlagTensor
pip install -e . --no-deps
```

## 使用示例

```python
import torch
import flagtensor

# 一元运算
x = torch.randn(1024, device="cuda", dtype=torch.float32)
y = flagtensor.abs(x)
z = flagtensor.relu(x)
w = flagtensor.sigmoid(x)

# 二元运算
a = torch.randn(1024, device="cuda")
b = torch.randn(1024, device="cuda")
c = flagtensor.add(a, b)

# 张量缩并
m = torch.randn(64, 32, device="cuda")
n = torch.randn(32, 48, device="cuda")
r = flagtensor.gett(m, n)
```

## 运行测试

```bash
# 单个算子正确性测试
pytest tests/unary/test_abs.py -v

# 录制测试结果为 JSON（使用 CPU-FP64 参考基准）
pytest tests/unary/test_abs.py --ref cpu --record json --output results.json

# 多 GPU 测试运行器（从 YAML 注册表读取）
python tools/run_tests.py --stages stable --gpus 0,1

# 提取算子标记
python tools/get_marks.py --stage stable --output ops.txt

# 性能测试并录制
pytest benchmark/test_unary_perf.py -m abs \
  --mode kernel --level core --record log

# 解析性能汇总
python tools/summary_for_plot.py result-*.log
```

## 项目结构

```
FlagTensor
├── src/flagtensor/            # Python 源码
│   ├── ops/                   # 算子实现（CUTENSOR_OP_*.py）
│   ├── utils/                 # 工具函数与 kernel 构建器
│   ├── runtime/               # 运行时支持
│   │   ├── backend/           # 厂商与架构后端（_nvidia/、_ascend/ 等）
│   │   └── common.py          # 厂商枚举与能力常量
│   ├── testing/               # 测试工具（断言、shape、dtype）
│   ├── fused/                 # 融合算子
│   └── modules/               # 模块实现
├── tests/                     # 每个算子的正确性测试
│   ├── unary/test_<op>.py     # 28 个一元算子测试
│   ├── binary/test_<op>.py    # 4 个二元算子测试
│   ├── contraction/           # 张量缩并测试
│   └── sparse/                # 稀疏算子测试
├── benchmark/                 # 性能测试
│   ├── consts.py              # dtype、shape、指标定义
│   └── test_<category>_perf.py
├── tools/                     # 命令行工具
│   ├── run_tests.py           # 多 GPU 测试运行器
│   ├── get_marks.py           # 从 YAML 提取 pytest 标记
│   └── summary_for_plot.py    # 解析并汇总 benchmark 日志
├── conf/
│   └── operators.yaml         # 算子注册表（测试的统一入口）
├── docs/                      # 文档
├── .github/workflows/         # CI/CD 流水线
├── LICENSE
├── README.md
└── pyproject.toml
```

## 支持的算子

| 类别 | 算子 | 状态 |
|------|------|------|
| **一元** | abs, acos, acosh, asin, asinh, atan, atanh, ceil, conj, cos, cosh, exp, floor, identity, log, mish, neg, rcp, relu, sigmoid, sin, sinh, soft_plus, soft_sign, sqrt, swish, tan, tanh | stable |
| **二元** | add, max, min, mul | stable |
| **缩并** | gett, tgett, ttgt, tensor_contraction_trinary, trinary_generic | stable |
| **稀疏** | block_sparse_tensor_contraction | experimental |

## 贡献代码

- 欢迎大家参与 FlagTensor 的算子开发并贡献代码，
  详情请参考[贡献指南](./CONTRIBUTING_cn.md)。
- 欢迎提交问题报告（Issue）或者特性请求（Feature Request）。
- 关于项目的疑问或建议，可发送邮件至 <a href="mailto:contact@flagos.io">contact@flagos.io</a>。

## 引用

欢迎引用我们的项目：

```bibtex
@misc{flagtensor2025,
    title={FlagOS/FlagTensor: A high-performance tensor-primitive library benchmarked against cuTensor},
    url={https://github.com/flagos-ai/FlagTensor},
    journal={GitHub},
    author={The FlagOS contributors},
    year={2025}
}
```

## 相关项目

- [FlagGems](https://github.com/flagos-ai/FlagGems) — 通用 Triton 算子库（500+ 算子）
- [FlagTree](https://github.com/flagos-ai/FlagTree) — FlagOS 维护的多后端 Triton 分支

## 许可证

本项目采用 [Apache License (Version 2.0)](./LICENSE) 授权许可。
