FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV GEMS_VENDOR=nvidia

# 系统依赖
RUN chmod 1777 /tmp \
    && apt-get update -qq && apt-get install -y -qq \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.6.0 + CUDA 12.4（运行时 GPU 透传，兼容 A100/H100/H800）
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# cuTensor（pip 安装 + 软链接。FlagTensor 通过 ctypes.CDLL 加载 libcutensor.so）
RUN python3 -m pip install --no-cache-dir cutensor-cu12 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && ln -sf $(python3 -c "import cutensor; print(cutensor.__path__[0])")/lib/libcutensor.so.2 \
       /usr/lib/x86_64-linux-gnu/libcutensor.so

# FlagTree 0.4.0+3.3（FlagOS 维护的 Triton 分支，从内部源安装）
RUN python3 -m pip install --no-cache-dir \
    --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple \
    --trusted-host=resource.flagos.net \
    "flagtree==0.4.0+3.3" --no-deps

# 运行时依赖（cuda-bindings + nvmath 是 FlagTensor 硬依赖）
RUN python3 -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "cuda-bindings>=12.9.6,<13" \
    nvmath-python \
    matplotlib \
    PyYAML \
    pytest \
    openpyxl

# FlagTensor
COPY . /workspace
WORKDIR /workspace
RUN python3 -m pip install -e . --no-deps

# 构建阶段验证（仅验证不依赖 GPU 的包；flagtensor 需要 GPU 驱动，运行时验证）
RUN python3 -c "import torch; print('torch:', torch.__version__)" \
    && python3 -c "import triton; assert hasattr(triton, 'Config'), 'triton OK'" \
    && python3 -c "import cutensor; print('cuTensor OK')" \
    && echo "=== FlagTensor 镜像构建成功（运行时需 docker run --gpus=all） ==="
