# FlagTensor Docker 交付指南

## 跨 GPU 兼容原理

Docker 镜像本身**不绑定 GPU 型号**。关键机制是：

1. **基础镜像** `nvidia/cuda:12.4.0-devel-ubuntu22.04`：提供 CUDA 12.4 工具链
2. **运行时 GPU 透传**：`docker run --gpus all` 将宿主机的 GPU 和驱动透传给容器
3. **PyTorch 自适应**：容器内 PyTorch 在首次调用时自动检测 GPU 架构（Ampere/Hopper/Blackwell 等）
4. **CUDA 兼容性**：CUDA 12.4 要求宿主机 NVIDIA 驱动 >= 525.60.13

已验证通过的硬件：

| GPU | 架构 | 状态 |
|-----|------|------|
| NVIDIA A100-SXM4-40GB | Ampere (SM80) | ✅ 已验证 |
| NVIDIA H800 | Hopper (SM90) | ✅ 甲方验收环境 |

**FP8 自动处理**：Ampere (A100) 不支持 FP8 运算，benchmark 代码已自动跳过 FP8 dtype。Hopper (H100/H800) 支持 FP8，会自动启用。

## 环境版本表

| 组件 | 版本 | 说明 |
|------|------|------|
| 基础镜像 | `nvidia/cuda:12.4.0-devel-ubuntu22.04` | CUDA 12.4，Python 3.10 |
| PyTorch | 2.6.0+cu124 | 驱动 >= 525 即可运行 |
| FlagTree | 0.4.0+3.3 | FlagOS Triton 分支，从内部源安装 |
| cuTensor | cutensor-cu12 2.6.0 | pip 安装 + 软链接 |
| cuda-bindings | >=12.9.6, <13 | nvmath 依赖 |
| nvmath-python | 0.9.0 | NVIDIA 数学库 Python 绑定 |
| Python 其他 | matplotlib, PyYAML | 可视化 + YAML 解析 |

> **为什么 FlagTree 是 0.4.0+3.3 而不是 0.5.0？**
>
> 内部源上 FlagTree 有 15+ 个变体。0.5.0 系列目前只有 mthreads 变体（`0.5.1+mthreads3.6`），缺少 NVIDIA 后端。
> `0.4.0+3.3` 是唯一同时具备 NVIDIA 后端 + `triton.Config` + `triton.autotune` 的版本。

## 1. 构建镜像

```bash
cd FlagTensor
docker build -f docker/Dockerfile -t flagtensor:latest .
```

## 2. 运行容器

```bash
# 单 GPU
docker run --gpus all --shm-size=16g -v $(pwd):/workspace -w /workspace \
  flagtensor:latest \
  python3 tools/run_tests.py --stages all --gpus 0

# 多 GPU（例如 8 卡 H800）
docker run --gpus all --shm-size=16g -v $(pwd):/workspace -w /workspace \
  flagtensor:latest \
  python3 tools/run_tests.py --gpus 0,1,2,3,4,5,6,7 --stages all --dump-output --output logs_results

# 只跑稳定算子
docker run --gpus all --shm-size=16g -v $(pwd):/workspace -w /workspace \
  flagtensor:latest \
  python3 tools/run_tests.py --stages stable --gpus 0 --dump-output --output logs_results
```

## 3. 验收流程

甲方的标准操作步骤：

```bash
# 1. 克隆代码
git clone https://github.com/flagos-ai/FlagTensor.git
cd FlagTensor

# 2. 构建镜像
docker build -f docker/Dockerfile -t flagtensor:latest .

# 3. 运行验收测试（8 卡全量）
docker run --gpus all --shm-size=16g -v $(pwd):/workspace -w /workspace \
  flagtensor:latest \
  python3 tools/run_tests.py --gpus 0,1,2,3,4,5,6,7 --stages all --dump-output --output logs_results

# 4. 查看结果
cat logs_results/summary.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
acc = sum(1 for v in d['result'].values() if v['accuracy']['status']=='Passed')
perf = sum(1 for v in d['result'].values() if v['performance']['status']=='Passed')
print(f'Accuracy: {acc}/{len(d[\"result\"])}')
print(f'Performance: {perf}/{len(d[\"result\"])}')
"
```

## 4. 导出 / 加载

```bash
# 导出
docker save -o flagtensor.tar flagtensor:latest

# 加载
docker load -i flagtensor.tar
```

## 5. 容器内验证

```bash
docker run -it --gpus all --shm-size=16g -v $(pwd):/workspace -w /workspace flagtensor:latest bash

# 在容器内执行：
python3 -c "import torch; print('GPU:', torch.cuda.get_device_name(), 'CUDA OK:', torch.cuda.is_available())"
python3 -c "import triton; print('triton:', triton.__version__, 'Config:', hasattr(triton,'Config'))"
python3 -c "from flagtensor.cutensor import CUTENSOR_AVAILABLE; print('cuTensor:', CUTENSOR_AVAILABLE)"
python3 tools/run_tests.py --ops abs,exp --gpus 0
```

## 6. 旧版本文档对照

| 旧文档 | 新文档 |
|--------|--------|
| `nvcr.io/nvidia/pytorch:25.05-py3` | `nvidia/cuda:12.4.0-devel-ubuntu22.04`（CUDA 12.4） |
| FlagTree 0.5.0（Triton 3.6） | FlagTree 0.4.0+3.3 |
| 多阶段构建 | 单阶段构建 |
| Docker Hub 闭源镜像，需下载 .tar.gz | 公共镜像 + pip + 内部源 |
