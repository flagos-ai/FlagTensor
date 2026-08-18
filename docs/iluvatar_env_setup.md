# FlagTensor 天数 (Iluvatar CoreX) 环境搭建指南 — 基于 FlagTree

> 适用对象: 从 0 搭建 FlagTensor 测试环境 (BI-V150 已验证)
> 前提: 机器已安装 Iluvatar CoreX SDK (驱动 + 工具链, 由天数提供)
> 验证环境: CoreX SDK 4.4.0, Ubuntu 22.04, Python 3.10, torch 2.7.1 (CoreX 版)

## 1. 组件说明

| 组件 | 版本 | 来源 | 说明 |
|------|------|------|------|
| CoreX SDK | 4.4.0 (例子) | 天数提供 | 提供驱动库、ixsmi、CoreX 版 PyTorch |
| FlagTree | 0.4.0+iluvatar3.1 | FlagOS 内部源 | FlagOS 维护的 Triton 发行版 (Iluvatar 变体) |
| FlagTensor | 本仓库 | GitHub | `pip install -e .` |

**为什么 FlagTree 选 `0.4.0+iluvatar3.1`**: 该 wheel 把 Iluvatar 后端插件
(`iluvatarTritonPlugin.so`) 直接打包在 wheel 内, pip 安装即可用。
更新的 `0.5.x+iluvatar3.1` 需要从 GitHub Releases 额外下载插件
(内网环境常不可达); `0.6.x+iluvatar3.6` 只有 cp312 wheel 且未经验证。
0.4.0 代际也与 NVIDIA 交付所选的 `0.4.0+3.3` 对齐。

## 2. 方式一 (推荐): setup.sh 裸机/虚机从 0 搭建

`setup.sh` 已重构为多后端脚本 (nvidia / ppu / iluvatar), 会自动检测后端:

```bash
git clone https://github.com/flagos-ai/FlagTensor.git
cd FlagTensor

# 自动检测 (有天数卡时识别为 iluvatar); 也可显式指定
./setup.sh                      # = ./setup.sh --backend iluvatar
```

脚本依次完成: Python 版本检查 → ixsmi 设备检查 → pip 升级 →
FlagTree 安装 (flagos 内部源) → 运行依赖 (matplotlib/PyYAML/pytest/openpyxl)
→ FlagTensor 安装 → 生成 `iluvatar_env.sh` → 安装校验。

### 2.1 重要: 每次跑测试前先 source 环境文件

CoreX SDK 的 `/usr/local/corex/enable` 会把 SDK 自带的 Triton 目录放进
`PYTHONPATH` 最前面, 从而**遮蔽** pip 安装的 FlagTree Triton。setup.sh
生成的 `iluvatar_env.sh` 把 pip site-packages 提到最前:

```bash
source iluvatar_env.sh
python3 tools/run_tests.py --stages all --gpus 0,1,2,3 --output-dir results
```

建议把 `source <repo>/iluvatar_env.sh` 加入 `~/.bashrc`。

### 2.2 如何判断当前用的是不是 FlagTree 的 Triton

```bash
python3 -c "import triton; print(triton.__file__)"
# 期望输出包含 site-packages (FlagTree):
#   /usr/local/lib/python3.10/site-packages/triton/__init__.py
# 如果输出是 corex-4.4.0/.../dist-packages/triton, 说明被 SDK 自带版遮蔽,
# 请先 source iluvatar_env.sh
```

## 3. 方式二: Docker 镜像

```bash
cd FlagTensor
# --build-arg 指定天数 CoreX 基础镜像 (向天数索取)
docker build -f docker/Dockerfile.iluvatar \
  --build-arg COREX_BASE=<天数corex基础镜像> \
  --build-arg COREX_VERSION=4.4.0 \
  -t flagtensor:iluvatar .

# 运行 (按 CoreX 容器化文档透传设备)
docker run --rm -it --privileged -v /dev:/dev --ipc=host --shm-size=16g \
  -v $(pwd):/workspace -w /workspace \
  flagtensor:iluvatar \
  python3 tools/run_tests.py --stages all --gpus 0,1,2,3
```

Dockerfile 内已处理 PYTHONPATH 顺序 (pip site-packages 在前)、
`GEMS_VENDOR=iluvatar` 检测钉死、`FLAGTREE_BACKEND_PLUGIN_LIB_DIR` 兜底。
构建期跳过设备校验 (`FLAGTENSOR_SKIP_VERIFY=1`), 运行时验证。

## 4. 验证清单 (setup 完成后)

```bash
source iluvatar_env.sh   # 裸机方式

python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python3 -c "import triton; print(triton.__version__, triton.__file__); assert hasattr(triton, 'Config')"
python3 -c "import flagtensor; from flagtensor.runtime import device; print(device.vendor_name)"  # 期望 iluvatar
python3 tools/run_tests.py --ops CUTENSOR_OP_ABS --gpus 0   # 单算子冒烟
python3 tools/run_tests.py --stages all --gpus 0 --output-dir results  # 全量
```

期望: `Accuracy: 36/36 passed`, `Performance: 36/36 passed`
(参考数据见 `docs/iluvatar_performance_report.md`)。

## 5. 常见问题 (FAQ)

**Q1: `import triton` 报 `iluvatarTritonPlugin.so: cannot open shared object file`**

FlagTree 的 `libtriton.so` 按裸名 dlopen 插件库。标准系统 Python 安装
(site-packages) 下插件在同目录可自动找到。如在 venv/conda 中遇到, 设置:

```bash
export FLAGTREE_BACKEND_PLUGIN_LIB_DIR=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/triton/_C
export LD_LIBRARY_PATH=$FLAGTREE_BACKEND_PLUGIN_LIB_DIR:$LD_LIBRARY_PATH
```

**Q2: 跑起来用的好像不是 FlagTree 的 Triton**

见 2.2 — PYTHONPATH 顺序问题, `source iluvatar_env.sh`。

**Q3: Python 3.12 环境怎么办**

`flagtree==0.5.1+iluvatar3.1` 有 cp312 wheel (setup.sh 会自动选择), 但需
额外下载后端插件并设置 `FLAGTREE_BACKEND_PLUGIN_LIB_DIR`
(插件 URL 见 FlagTree releases; 内网环境建议改用 Python 3.10 方案)。
`0.6.1+iluvatar3.6` (cp312) 未经验证, 如需尝试:
`FLAGTREE_PKG='flagtree==0.6.1+iluvatar3.6' ./setup.sh --backend iluvatar`。

**Q4: 没有外网/内部源不通**

`resource.flagos.net` 必须可达 (FlagTree 唯一下载渠道)。可提前在有网
机器 `pip download flagtree==0.4.0+iluvatar3.1 --no-deps -d wheels/`
后拷贝 wheel 离线安装。

**Q5: benchmark 里 Contraction 相关算子准确吗 / cuTensor 报错**

天数 CoreX SDK 的 `libcutensor.so` 是占位 stub, FlagTensor 已自动识别并
切换到 PyTorch 原生 baseline, 不用安装任何 cuTensor 包。
