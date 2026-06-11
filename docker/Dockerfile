# FlagTensor Docker 镜像

# 构建（在 FlagTensor 项目根目录下执行）:
#   docker build -f docker/Dockerfile -t flagtensor:latest .
#
# 运行（GPU 透传，兼容 A100 / H100 / H800 等所有 NVIDIA GPU）:
#   docker run -it --gpus all --shm-size=16g -v $(pwd):/workspace -w /workspace flagtensor:latest bash
#
# 验收测试（多卡）:
#   docker run --gpus all --shm-size=16g -v $(pwd):/workspace -w /workspace \
#     flagtensor:latest python3 tools/run_tests.py --gpus 0,1,2,3,4,5,6,7 --stages all --dump-output --output logs_results

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 系统依赖
RUN apt-get update -qq && apt-get install -y -qq python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.6.0 + CUDA 12.4（要求宿主机 NVIDIA 驱动 >= 525）
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# cuTensor（pip 安装 + 软链接，FlagTensor 通过 ctypes.CDLL 加载 libcutensor.so）
RUN python3 -m pip install --no-cache-dir cutensor-cu12 \
    && ln -sf $(python3 -c "import cutensor; print(cutensor.__path__[0])")/lib/libcutensor.so.2 \
       /usr/lib/x86_64-linux-gnu/libcutensor.so

# FlagTree 0.4.0+3.3（FlagOS 维护的 Triton 分支，含 NVIDIA 后端 + triton.Config + triton.autotune）
# 从内部源安装，公共 PyPI 上无此包
RUN python3 -m pip install --no-cache-dir \
    --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple \
    --trusted-host=resource.flagos.net \
    "flagtree==0.4.0+3.3" --no-deps

# Python 运行时依赖
RUN python3 -m pip install --no-cache-dir \
    "cuda-bindings>=12.9.6,<13" nvmath-python matplotlib PyYAML

# FlagTensor
COPY . /workspace
WORKDIR /workspace
RUN python3 -m pip install -e . --no-deps

# 构建阶段导入验证（GPU 在运行时才可用）
RUN python3 -c "import torch; print('torch:', torch.__version__)" \
    && python3 -c "import flagtensor; print('flagtensor OK')" \
    && python3 -c "import triton; assert hasattr(triton, 'Config'), 'triton OK'" \
    && python3 -c "import cutensor; print('cuTensor OK')" \
    && echo "=== FlagTensor 镜像构建成功 ==="
