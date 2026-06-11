# FlagTensor Docker 镜像构建指南

## 环境差异说明

当前实际验证通过的环境与之前交付的 `flagtensor_docker.md` 有以下关键差异：

| 组件 | 旧环境（之前文档） | 当前验证通过的环境 | 变化原因 |
|------|-------------------|-------------------|---------|
| 基础镜像 | `nvcr.io/nvidia/pytorch:25.05-py3` | `nvidia/cuda:12.4.0-devel-ubuntu22.04` | 避免 PyTorch 版本过高导致 CUDA 驱动不兼容 |
| Python | 3.12 | 3.10 | 基础镜像自带 |
| PyTorch | 25.05 镜像自带（≥2.7，cu128） | `2.6.0+cu124` | 驱动 535.161.08 最高支持 CUDA 12.2，PyTorch 2.7+ 需要 CUDA 12.6+，会导致 `torch.cuda.is_available()` 返回 False |
| FlagTree | `0.5.0`（提供 Triton 3.6） | `0.4.0+3.3` | `0.5.0` 系列在内部源上只有 mthreads 变体（`0.5.1+mthreads3.6`），没有 NVIDIA CUDA 后端。`0.4.0+3.3` 是唯一同时具备 NVIDIA 后端、`triton.Config` 和 `triton.autotune` 的版本 |
| cuTensor | 未知 | `cutensor-cu12 2.6.0`（pip） | pip 安装更简单，无需系统包管理器 |

> **核心结论**：驱动版本（CUDA 12.2）是硬约束，决定了 PyTorch 不能超过 2.6.x。FlagTree 版本选择受限于内部源的可用变体，`0.4.0+3.3` 是当前唯一可用选择。

---

## 1. Dockerfile

文件位于 `docker/Dockerfile`，内容如下：

```dockerfile
# FlagTensor Docker 镜像
# 基于当前验证通过的运行环境构建：
#   Python 3.10 | PyTorch 2.6.0+cu124 | FlagTree 0.4.0+3.3 | cuTensor 2.6.0

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# ---- 系统依赖 ----
RUN apt-get update -qq && apt-get install -y -qq \
    python3 python3-pip git wget curl \
    && rm -rf /var/lib/apt/lists/*

# ---- PyTorch 2.6.0 (CUDA 12.4，兼容驱动 535 / CUDA 12.2) ----
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ---- cuTensor (pip 安装 + 软链接) ----
# FlagTensor 通过 ctypes.CDLL("libcutensor.so") 加载，
# pip 安装后需创建软链接到系统库路径
RUN python3 -m pip install --no-cache-dir cutensor-cu12 \
    && ln -sf /usr/local/lib/python3.10/dist-packages/cutensor/lib/libcutensor.so.2 \
       /usr/lib/x86_64-linux-gnu/libcutensor.so

# ---- FlagTree 0.4.0+3.3 (FlagOS 维护的 Triton 分支) ----
# 必须从此内部源安装，公共 PyPI 上没有此包
RUN python3 -m pip install --no-cache-dir \
    --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple \
    --trusted-host=resource.flagos.net \
    "flagtree==0.4.0+3.3" --no-deps

# ---- Python 运行时依赖 ----
RUN python3 -m pip install --no-cache-dir \
    "cuda-bindings>=12.9.6,<13" \
    nvmath-python \
    matplotlib \
    PyYAML

# ---- 抑制 NGC/PyTorch 启动横幅 (基于 nvidia/cuda 镜像可能不存在此目录) ----
RUN if [ -d /opt/nvidia/entrypoint.d ]; then \
      touch /opt/nvidia/entrypoint.d/10-banner.txt 2>/dev/null || true; \
      printf '#!/bin/bash\n# suppressed\n' > /opt/nvidia/entrypoint.d/12-banner.sh 2>/dev/null || true; \
    fi

# ---- 安装 FlagTensor ----
COPY . /workspace
WORKDIR /workspace
RUN python3 -m pip install -e . --no-deps

# ---- 构建阶段验证 (导入检查，GPU 在运行时才可用) ----
RUN python3 -c "import torch; print('torch:', torch.__version__)" \
    && python3 -c "import flagtensor; print('flagtensor OK')" \
    && python3 -c "import triton; assert hasattr(triton, 'Config'), 'triton.Config missing'; print('triton OK')" \
    && python3 -c "import cutensor; print('cuTensor OK')" \
    && echo "=== FlagTensor Docker 镜像构建成功 ==="
```

---

## 2. 构建镜像

```bash
cd FlagTensor
docker build -f docker/Dockerfile -t flagtensor-a100:latest .
```

构建时间约 5-10 分钟（取决于网络速度，FlagTree 包约 328MB）。

---

## 3. 运行容器

```bash
# 交互式终端
docker run -it --gpus all --shm-size=16g \
  -v $(pwd):/workspace -w /workspace \
  flagtensor-a100:latest bash

# 运行全量测试
docker run --gpus all --shm-size=16g \
  -v $(pwd):/workspace -w /workspace \
  flagtensor-a100:latest \
  python3 tools/run_tests.py --stages stable --gpus 0

# 运行单个算子测试
docker run --gpus all --shm-size=16g \
  -v $(pwd):/workspace -w /workspace \
  flagtensor-a100:latest \
  python3 -m pytest tests/unary/test_abs.py -v
```

---

## 4. 导出 / 加载镜像

```bash
# 导出（构建机器上执行）
docker save -o flagtensor-a100.tar flagtensor-a100:latest

# 加载（目标机器上执行）
docker load -i flagtensor-a100.tar
```

---

## 5. 容器内验证

进入容器后执行以下命令，确认环境正常：

```bash
# 检查 PyTorch + CUDA
python3 -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# 检查 FlagTree (Triton)
python3 -c "import triton; print('triton:', triton.__version__, 'Config:', hasattr(triton,'Config'))"

# 检查 cuTensor
python3 -c "from flagtensor.cutensor import CUTENSOR_AVAILABLE; print('cuTensor:', CUTENSOR_AVAILABLE)"

# 冒烟测试
python3 -c "
from flagtensor.cutensor import CuTensorAdd
import torch
x, y = torch.randn(100, device='cuda'), torch.randn(100, device='cuda')
z = CuTensorAdd()(x, y)
print('cuTensor Add OK:', z.shape)
"

# 跑几个算子验证
python3 tools/run_tests.py --ops abs,exp,add --gpus 0
```

---

## 6. 旧版本文档对比

| 旧文档 (flagtensor_docker.md) | 新文档 |
|------------------------------|--------|
| 基础镜像 `nvcr.io/nvidia/pytorch:25.05-py3` | `nvidia/cuda:12.4.0-devel-ubuntu22.04` |
| FlagTree 0.5.0 | FlagTree 0.4.0+3.3 |
| 需下载 `.tar.gz` 并 `docker load` | 使用公共 Docker Hub 基础镜像 + pip 安装 |
| 多阶段构建（v1 → v2） | 单阶段构建 |
| 含 curand 测试脚本 | 含 FlagTensor 导入 + API 冒烟测试 |
| 构建两次 | 构建一次即可 |
