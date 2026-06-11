# FlagTensor 环境搭建指南

## 已验证环境

| 组件 | 版本 | 说明 |
|------|------|------|
| GPU | NVIDIA A100-SXM4-40GB | CUDA compute capability 8.0 (Ampere) |
| CUDA 驱动 | 535.161.08 | CUDA 12.2 |
| Python | 3.10.12 | |
| OS | Ubuntu 22.04.5 LTS | x86_64 |
| PyTorch | 2.6.0+cu124 | |
| FlagTree | 0.4.0+3.3 | FlagOS 维护的 Triton 分支 |
| cuTensor | cutensor-cu12 2.6.0 | pip 安装 |
| cuda-bindings | 12.9.7 | nvmath-python 依赖 |
| nvmath-python | 0.9.0 | NVIDIA 数学库 Python 绑定 |
| matplotlib | 3.10.1 | 可视化 |
| PyYAML | 6.0+ | 算子注册表 |

## 安装步骤

### 1. 安装 PyTorch

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

> PyTorch 版本不能超过 2.6.x。驱动 535 只支持到 CUDA 12.2，PyTorch 2.7+ 需要 CUDA 12.6+，会导致 `torch.cuda.is_available()` 返回 False。

### 2. 安装 cuTensor

```bash
pip install cutensor-cu12
ln -sf $(python3 -c "import cutensor; print(cutensor.__path__[0])")/lib/libcutensor.so.2 \
  /usr/lib/x86_64-linux-gnu/libcutensor.so
```

> FlagTensor 通过 `ctypes.CDLL("libcutensor.so")` 加载 cuTensor，pip 安装后需手动创建软链接。

### 3. 安装 FlagTree

```bash
pip install --no-cache-dir \
  --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple \
  --trusted-host=resource.flagos.net \
  "flagtree==0.4.0+3.3" --no-deps
```

> **为什么是 0.4.0+3.3？**
>
> 内部源上 FlagTree 有多个版本，但只有 `0.4.0+3.3` 同时满足：
> - NVIDIA 后端（`triton/backends/nvidia/`）
> - `triton.Config` 类（`@triton.autotune` 装饰器需要）
> - `triton.language.extra.cuda.libdevice`（算子的数学函数实现需要）
>
> `0.5.0` 系列目前只有 mthreads 变体（如 `0.5.1+mthreads3.6`），缺少 NVIDIA CUDA 后端支持。
> `0.5.0+aipu3.3` 是 AIPU（寒武纪）变体，也没有 CUDA 后端。

### 4. 安装 Python 依赖

```bash
pip install "cuda-bindings>=12.9.6,<13" nvmath-python matplotlib PyYAML
```

### 5. 安装 FlagTensor

```bash
cd /path/to/FlagTensor
pip install -e . --no-deps
```

> 使用 `--no-deps` 避免 pip 自动升级 PyTorch 和 triton 版本。

## 验证

```bash
# 基础验证
python3 -c "import torch; assert torch.cuda.is_available(); print('torch:', torch.__version__)"
python3 -c "import triton; assert hasattr(triton, 'Config'); print('triton OK')"
python3 -c "from flagtensor.cutensor import CUTENSOR_AVAILABLE; assert CUTENSOR_AVAILABLE; print('cuTensor OK')"

# 冒烟测试
python3 -c "
from flagtensor.cutensor import CuTensorAdd
import torch
x, y = torch.randn(100, device='cuda'), torch.randn(100, device='cuda')
z = CuTensorAdd()(x, y)
print('OK:', z.shape)
"

# 单算子测试
pytest tests/unary/test_abs.py -v

# 全量 Stable 算子测试
python tools/run_tests.py --stages stable --gpus 0
```

## 镜像加速

若 PyPI 下载较慢，使用清华源：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple cutensor-cu12 matplotlib PyYAML
```

## 已知限制

- **PyTorch 版本上限**：驱动 535 只能到 CUDA 12.2，PyTorch 需 ≤ 2.6.x
- **FlagTree 版本锁定**：仅 0.4.0+3.3 同时有 NVIDIA 后端 + Config + autotune
- **cuTensor 软链接**：pip 安装后需手动创建 libcutensor.so 链接
- **FP8 不支持**：A100 (Ampere) 不支持 FP8 数据类型，benchmark 已自动排除
