# FlagTensor 环境搭建指南

## 1. 基础环境要求

| 组件 | 要求 |
|------|------|
| GPU | NVIDIA A100/H100/H20 等（CUDA compute capability ≥ 8.0） |
| CUDA 驱动 | ≥ 12.2 |
| Python | 3.10+ |
| OS | Ubuntu 22.04 |

## 2. 安装步骤

### 2.1 安装 PyTorch + CUDA 工具链

```bash
# PyTorch 2.6.0 (CUDA 12.4)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# cuTensor
pip install cutensor-cu12
ln -sf $(python3 -c "import cutensor; print(cutensor.__path__[0])")/lib/libcutensor.so.2 \
  /usr/lib/x86_64-linux-gnu/libcutensor.so

# NVIDIA 数学库
pip install "cuda-bindings>=12.9.6,<13" nvmath-python matplotlib PyYAML openpyxl
```

### 2.2 安装 FlagTree（Triton 分支）

```bash
# 从内部源安装（需要 flagos 仓库访问权限）
pip install --no-cache-dir \
  --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple \
  --trusted-host=resource.flagos.net \
  "flagtree==0.4.0+3.3" --no-deps
```

> **说明**：FlagTree 是 flagos-ai 维护的 Triton 分支，`0.4.0+3.3` 版本对应 Triton 3.3 内核。
> 该版本包含 `triton.Config` 和 `triton.autotune`，且同时包含 NVIDIA 和 AMD 后端。

### 2.3 安装 FlagTensor

```bash
pip install -e /path/to/FlagTensor --no-deps
```

使用 `--no-deps` 避免自动升级 PyTorch/triton 版本。

### 2.4 验证安装

```bash
# 验证 cuTensor 可用
python3 -c "
import torch
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorAdd
print('CUTENSOR_AVAILABLE:', CUTENSOR_AVAILABLE)
x, y = torch.randn(100, device='cuda'), torch.randn(100, device='cuda')
print('OK:', CuTensorAdd()(x, y).shape)
"

# 运行精度测试
pytest tests/unary/test_abs.py -v

# 运行所有 stable 算子
python tools/run_tests.py --stages stable --gpus 0
```

## 3. 使用清华镜像源（加速安装）

若 PyPI 下载较慢，添加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple cutensor-cu12 cuda-bindings nvmath-python matplotlib PyYAML openpyxl
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e /path/to/FlagTensor --no-deps
```

## 4. 版本锁定说明

| 包 | 版本 | 原因 |
|---|------|------|
| `torch` | `2.6.0+cu124` | 与 CUDA 12.2 驱动兼容 |
| `flagtree` | `0.4.0+3.3` | 唯一同时提供 NVIDIA 后端 + `triton.Config` 的内部版本 |
| `cuda-bindings` | `≥12.9.6, <13` | nvmath-python 依赖 |
| `cutensor-cu12` | `2.6.0` | pip 安装，需做 libcutensor.so 软链接 |

> **注意**：PyTorch 版本不能超过 2.6.x，否则可能需要更新 NVIDIA 驱动（驱动 535 只支持到 CUDA 12.2，而 PyTorch 2.7+ 需要 CUDA 12.6+）。

## 5. 多厂商适配（预留接口）

FlagTensor 目前已预留多厂商后端接口，只需新建对应厂商模块即可接入：

```
src/flagtensor/runtime/backend/
├── _nvidia/        # 当前唯一实现
│   ├── __init__.py          # vendor_info 定义
│   ├── ampere/              # A100 架构 autotune
│   └── hopper/              # H100 架构 autotune
└── _<vendor>/      # 新厂商模块（如 _ascend/_cambricon 等）
    └── __init__.py          # 只需 5 行 vendor_info 即可接入
```

新厂商接入示例（以华为昇腾为例，仅作预留）：

```python
# 新建 src/flagtensor/runtime/backend/_ascend/__init__.py
from backend_utils import VendorInfoBase

vendor_info = VendorInfoBase(
    vendor_name="ascend",
    device_name="npu",
    device_query_cmd="npu-smi info",
    dispatch_key="PrivateUse1",
)
```

厂商自动检测优先级：
1. 环境变量 `GEMS_VENDOR`、`FLAGGEMS_VENDOR`
2. PyTorch 属性检测（`torch.npu`、`torch.mlu` 等）
3. 系统命令检测（`nvidia-smi`、`npu-smi info` 等）
