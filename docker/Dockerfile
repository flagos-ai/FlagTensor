# FlagTensor Docker 镜像
# 构建: docker build -t flagtensor-a100:latest .
# 运行: docker run -it --gpus all --shm-size=16g -v $(pwd):/workspace -w /workspace flagtensor-a100:latest bash
# 测试: docker run --gpus all --shm-size=16g -v $(pwd):/workspace -w /workspace flagtensor-a100:latest python3 tools/run_tests.py --stages stable --gpus 0

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 系统依赖
RUN apt-get update -qq && apt-get install -y -qq \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.6.0 (CUDA 12.4)
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# cuTensor (pip 安装后需软链接到系统库路径，供 ctypes.CDLL 加载)
RUN python3 -m pip install --no-cache-dir cutensor-cu12 \
    && ln -sf $(python3 -c "import cutensor; print(cutensor.__path__[0])")/lib/libcutensor.so.2 \
       /usr/lib/x86_64-linux-gnu/libcutensor.so

# FlagTree 0.4.0+3.3 — FlagOS 维护的 Triton 分支
# 含 NVIDIA 后端 + triton.Config + triton.autotune，从内部源安装
RUN python3 -m pip install --no-cache-dir \
    --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple \
    --trusted-host=resource.flagos.net \
    "flagtree==0.4.0+3.3" --no-deps

# Python 运行时依赖
RUN python3 -m pip install --no-cache-dir \
    "cuda-bindings>=12.9.6,<13" \
    nvmath-python \
    matplotlib \
    PyYAML

# 安装 FlagTensor
COPY . /workspace
WORKDIR /workspace
RUN python3 -m pip install -e . --no-deps

# 构建阶段验证
RUN python3 -c "import torch; print('torch:', torch.__version__)" \
    && python3 -c "import flagtensor; print('flagtensor OK')" \
    && python3 -c "import triton; assert hasattr(triton, 'Config'), 'triton.Config missing'; print('triton OK')" \
    && python3 -c "import cutensor; print('cuTensor OK')" \
    && echo "=== FlagTensor 镜像构建成功 ==="
