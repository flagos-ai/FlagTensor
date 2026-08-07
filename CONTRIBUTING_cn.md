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

[English](./CONTRIBUTING.md)

# FlagTensor 贡献者指南

感谢您对 FlagTensor 的兴趣！我们使用 GitHub 来托管代码、管理问题和处理拉取请求。在贡献之前，请阅读以下指南。

## 1. 错误报告

请使用 GitHub 的 Issues 来报告错误。在报告错误时，请提供：

- 简单摘要
- 复现步骤
- 确保描述具体且准确！
- 如果可以提供一些示例代码将会帮助开发者快速定位问题

## 2. 代码贡献

在提交拉取请求时，贡献者应描述所做的更改以及更改的原因。
如果可以设计测试用例，请提供相应测试。拉取请求在合并前需要 **一位** 成员的批准，
而且需要通过代码的持续集成检查：

- **质量门禁** (`quality-gate.yaml`): pre-commit 检查（black、isort、flake8、clang-format）、构建检查、注册表一致性
- **正确性测试**: 所有 per-operator 测试必须基于 CPU-FP64 参考基准通过
- **性能测试**: Benchmark 结果不得低于基线水平

## 3. 添加新算子

添加新算子时，需要创建或更新以下文件：

1. **算子实现**: `src/flagtensor/ops/CUTENSOR_OP_<NAME>.py`
2. **正确性测试**: `tests/<category>/test_<name>.py`（每个算子一个测试文件）
3. **性能测试**: 更新或创建 `benchmark/test_<category>_perf.py`
4. **YAML 注册**: 在 `conf/operators.yaml` 中添加条目

### 测试文件模板

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

### YAML 条目模板

```yaml
- id: your_op
  name: your_op
  category: unary
  for: [your_op]
  labels: [cuTensor, pointwise]
  kind: [Math]
  stages:
    - stable: '1.0'
  description: 算子的简要描述。
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

## 4. 添加新硬件后端

FlagTensor 采用插件式后端架构。添加新厂商只需：

1. 创建 `src/flagtensor/runtime/backend/_<vendor>/__init__.py`：

```python
from backend_utils import VendorInfoBase

vendor_info = VendorInfoBase(
    vendor_name="your_vendor",
    device_name="your_device",      # torch.{device_name}
    device_query_cmd="your-smi",    # 硬件检测命令
    dispatch_key=None,              # PyTorch 分发键（或 "PrivateUse1"）
)
```

2. （可选）在同一厂商目录下创建架构专用目录和算子重写。

## 5. 开发环境搭建

```bash
# 克隆并安装
git clone https://github.com/flagos-ai/FlagTensor.git
cd FlagTensor
pip install -e . --no-deps

# 启用 pre-commit 钩子
pip install pre-commit
pre-commit install
```

## FlagTensor 项目结构

```
FlagTensor
├── src/flagtensor/            # Python 源码
│   ├── ops/                   # 独立算子实现
│   ├── utils/                 # Python 工具
│   ├── runtime/               # 运行时支持与后端抽象
│   ├── testing/               # 测试工具
│   ├── fused/                 # 融合算子
│   └── modules/               # 模块实现
├── tests/                     # per-operator 正确性测试
├── benchmark/                 # 性能测试
├── tools/                     # 命令行工具（run_tests、get_marks 等）
├── conf/
│   └── operators.yaml         # 算子注册表（权威列表）
├── docs/                      # 文档
├── LICENSE
├── README.md
├── CONTRIBUTING.md
└── pyproject.toml
```

## FlagTensor 许可证

FlagTensor 使用 [Apache License (Version 2.0)](./LICENSE) 许可证。
