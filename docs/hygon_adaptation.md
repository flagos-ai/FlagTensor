# FlagTensor 海光 (Hygon DCU) 平台适配报告

> 适配时间: 2026-08
> 适配目标: Hygon DCU (Deep Computing Unit, CUDA 兼容, torch_hcu 插件)
> 适配结果: 36 个算子全部可在 Hygon DCU 上运行 accuracy + performance 测试

## 1. 背景

FlagTensor 此前已支持 NVIDIA (cuTensor 作为 vendor baseline)、Alibaba PPU、Iluvatar
CoreX 与 Kunlunxin XPU (三者也以 PyTorch 原生 op 作为 baseline, 见
`docs/ppu_adaptation.md` / `docs/iluvatar_adaptation.md`)。迁移到海光 DCU 平台时面临的
核心问题与 PPU / Iluvatar / Kunlunxin 类似:

1. **海光 DCU 没有可用的 cuTensor 库作为 baseline**: SDK 不提供 `libcutensor.so`,
   也没有等价的 generalized tensor-contraction / elementwise-trinary 库。
2. **Hygon DCU 未被 backend 抽象识别**: `runtime/backend/` 下不存在 `_hygon/` 模块,
   `DeviceDetector` 虽然能通过 `torch.__hcu_version__` 标记属性识别 vendor, 但
   `get_vendor_module('hygon')` 因找不到 `_hygon/` 目录而抛 `ModuleNotFoundError`,
   最后 fallback 到 nvidia, 导致 baseline 选型错误、benchmark 全量 skip。
. **`run_tests.py` 把 hygon 当作 pilot vendor**: 既有 vendor 白名单只包含
   `("nvidia", "ppu", "unknown")`, 海光被当作 pilot 阶段 vendor, 默认只跑 8 个算子,
   这就是「全量算子效果太差」的直接原因。

本报告描述 FlagTensor 为支持海光 DCU 平台所做的全部适配工作, 包括 vendor 抽象接入、
设备检测补全、benchmark harness vendor 白名单扩展, 以及对 NVIDIA / PPU / Iluvatar /
Kunlunxin 路径零影响的兼容性保证。

## 2. Baseline 选型

与 PPU / Iluvatar / Kunlunxin 的结论一致: **PyTorch 原生 op 是海光 DCU 上唯一可行且
正确的 baseline**。`torch.matmul` / `torch.einsum` / elementwise aten op 通过
`torch_hcu` 符号改写 hook 分发到海光 DCU 的 vendor 库 (`libdtkblas.so`、
`libdtkdnn.so` 等), 其在海光 DCU 上的地位等同于 NVIDIA 上的 cuTensor。这不是
fallback, 而是 vendor 原生优化路径。

海光 DCU 基于 ROCm 兼容的 gfx9-class 架构, 支持 FP64 / BF16 / INT64 (故未出现在
`UNSUPPORT_FP64` / `UNSUPPORT_BF16` / `UNSUPPORT_INT64` 集合中), accuracy 测试会
按完整 dtype 集合执行。

## 3. 适配内容

### 3.1 Hygon vendor 后端模块 (`src/flagtensor/runtime/backend/_hygon/`)

按 PPU / Iluvatar / Kunlunxin 建立的 vendor 接入模式新增:

```
_hygon/
├── __init__.py                  # vendor_info(hygon-smi), ARCH_MAP, BASELINE_AVAILABLE=True
├── baseline.py                  # PyTorch-native baseline 类 + BASELINE_CLASSES registry
├── tolerances.yaml              # Hygon 容差配置
├── heuristics_config_utils.py   # elementwise heuristics (与 PPU/Iluvatar/Kunlunxin 相同)
└── tune_configs.yaml            # autotune 配置 (复用 nvidia 的 Ampere/Hopper 配置)
```

- `vendor_info`: `vendor_name='hygon'`, `device_name='cuda'` (Hygon DCU 通过
  `torch_hcu` 兼容层接入 `torch.cuda`), `device_query_cmd='hygon-smi'`。
- `ARCH_MAP = {'8': 'ampere', '9': 'hopper'}`: DCU 通过 CUDA 兼容层上报 compute
  capability, K100 / K100 AI 变体上报 9.0, 旧变体上报 8.0。复用 NVIDIA Ampere /
  Hopper 的 arch 特化配置, autotune 配置与 kernel tuning 因 target 同一 compute
  capability family 而完全一致。
- `baseline.py`: 复用 `flagtensor.torch_baseline` 的 6 个基类, 为 36 个算子注册
  `BASELINE_CLASSES`, 并导出 `elementwise_trinary` / `_get_trinary_executor` 函数式
  API (与 `_ppu/baseline.py` / `_kunlunxin/baseline.py` 同构)。
- 同时导出 `CuTensor{SlugCamelCase}` 别名 (如 `CuTensorAbs`、`CuTensorContraction`),
  指向对应的 `Baseline*` 类, 让 `Benchmark._get_baseline_instance()` 通过
  `getattr(baseline_module, "CuTensorAbs")` 解析时在 Hygon 上透明工作。
- `tolerances.yaml`: vendor 级 `benchmark_verify_floor: atol=1e-3, rtol=1e-3`。
  海光 baseline (DCU vendor GEMM) 与 Triton kernel 使用不同的 GEMM summation order,
  contraction-family 算子在近零输出元素上会放大绝对误差, 故采用与 PPU / Iluvatar /
  Kunlunxin 一致的 1e-3 floor。
- `tune_configs.yaml`: 直接复用 `_nvidia/tune_configs.yaml` 的内容 (因 Hygon 复用
  Ampere/Hopper arch 特化)。

### 3.2 设备检测 (`src/flagtensor/runtime/backend/device.py`)

`_get_vendor_from_quick_cmd()` 的 Layer 2 快速检测新增两条 Hygon 分支 (插在
Kunlunxin `torch_xmlir` 检测之后、NVIDIA `torch.cuda` fallback 之前):

1. **`torch_hcu` 模块可导入** → 返回 `"hygon"` (与 Kunlunxin 检测 `torch_xmlir` 同构,
   是主信号)。
2. **`torch.__hcu_version__` 标记属性存在** → 返回 `"hygon"` (作为
   `torch_hcu` 模块懒加载场景的兜底, 该属性已在 `common._VENDOR_TORCH_ATTR` 中注册)。

Layer 2 的 CUDA 设备名回退分支也新增一条 Hygon 识别 (插在 Kunlunxin 之后、NVIDIA 之前):

3. **设备名含 `"HYGON"` 或以 `"DCU"` 开头** → 返回 `"hygon"` (作为
   `torch_hcu` 模块懒加载场景的次级兜底)。

NVIDIA 设备名不含 "HYGON" / "DCU", PPU 设备名以 "PPU" 开头先行命中, Iluvatar 设备名
含 "ILUVATAR" 先行命中, Kunlunxin 设备名含 "XPU" / "KUNLUN" 先行命中 — 各分支互不干扰。

### 3.3 Benchmark harness vendor 白名单 (`src/flagtensor/benchmark_core.py`)

`Benchmark._baseline_module()` 与 `vendor_baseline_available()` 的 vendor 白名单从

```python
if vendor_name in ("ppu", "iluvatar", "kunlunxin"):
```

扩展为

```python
if vendor_name in ("ppu", "iluvatar", "kunlunxin", "hygon"):
```

让 benchmark harness 在 `CUTENSOR_AVAILABLE=False` 且 `torch_npu_available()=False`
时, 能加载 `_hygon.baseline` 作为 baseline 模块, 并通过 `CuTensor*` 别名解析到对应
的 `Baseline*` 类 (PyTorch-native 实现)。

### 3.4 全量算子 vendor 白名单 (`tools/run_tests.py`)

`_PILOT_VENDOR_OPS` 过滤的 vendor 白名单从

```python
if _vendor not in ("nvidia", "ppu", "unknown"):
```

扩展为

```python
if _vendor not in ("nvidia", "ppu", "iluvatar", "kunlunxin", "hygon", "unknown"):
```

让海光 DCU 默认跑全量 36 个算子, 而不是只跑 8 个 pilot 算子。这是「全量算子效果太差」
问题的直接修复点。顺带把已存在完整 backend 的 Iluvatar / Kunlunxin 也纳入全量白名单
(此前它们也只能通过 `--ops` / `--op-list-file` 显式指定才能跑全量)。

## 4. 现有 vendor 兼容性保证

本次适配对 NVIDIA / PPU / Iluvatar / Kunlunxin 路径**完全零影响**, 验证如下:

### 4.1 device.py

- 新增的 Hygon 检测分支是独立的新分支 (`torch_hcu` 模块检查 + `__hcu_version__`
  属性检查 + `HYGON`/`DCU` 名字检查), 与既有分支互斥。
- NVIDIA / PPU / Iluvatar / Kunlunxin 的检测分支原样保留, 顺序不变。
- `_VENDOR_TORCH_ATTR["hygon"] = "__hcu_version__"` 既有逻辑自动生效 (该 entry 早已
  在 `common.py` 中注册, 无需改动)。

### 4.2 benchmark_core.py

- vendor 白名单是**严格扩展** (新增 `"hygon"` 到 tuple), 不影响其他 vendor 的判定。
- `_baseline_module()` / `vendor_baseline_available()` 的 fallback 链顺序不变:
  `CUTENSOR_AVAILABLE` → `torch_npu_baseline` → vendor 白名单 fallback。
- NVIDIA+cuTensor 路径: `CUTENSOR_AVAILABLE=True` 直接返回 `flagtensor.cutensor`,
  白名单分支根本不执行。
- Ascend 路径: `torch_npu_available()=True` 直接返回 `flagtensor.torch_npu_baseline`,
  白名单分支不执行。
- PPU / Iluvatar / Kunlunxin 路径: 仍在白名单内, 行为不变。

### 4.3 tools/run_tests.py

- vendor 白名单是**严格扩展** (新增 `"iluvatar"`、`"kunlunxin"`、`"hygon"` 到 tuple)。
- NVIDIA / PPU / unknown 路径仍在白名单内, 跑全量算子的行为不变。
- 其他 vendor (Ascend / T-Head 等) 仍走 pilot 过滤, 行为不变。

### 4.4 _hygon/ 模块

- 全部为新增文件, 不修改任何既有 vendor 的 `_<vendor>/` 目录。
- `tolerances.yaml` / `tune_configs.yaml` 是独立的新文件, 不影响其他 vendor 的 yaml
  加载 (各 vendor 的 yaml 由 `_<vendor>/` 路径独立解析)。

## 5. 适配文件清单

### 新增文件

| 文件 | 作用 |
|------|------|
| `src/flagtensor/runtime/backend/_hygon/__init__.py` | Hygon vendor 模块 (vendor_info / ARCH_MAP / BASELINE_AVAILABLE / get_baseline_class) |
| `src/flagtensor/runtime/backend/_hygon/baseline.py` | Hygon baseline 类 + CuTensor* 别名 + BASELINE_CLASSES registry |
| `src/flagtensor/runtime/backend/_hygon/tolerances.yaml` | Hygon 容差配置 (1e-3 floor) |
| `src/flagtensor/runtime/backend/_hygon/heuristics_config_utils.py` | elementwise heuristics (复用 nvidia 配置) |
| `src/flagtensor/runtime/backend/_hygon/tune_configs.yaml` | autotune 配置 (复用 nvidia Ampere/Hopper 配置) |
| `docs/hygon_adaptation.md` | 本文档 |

### 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/flagtensor/runtime/backend/device.py` | 新增 Hygon 检测分支 (torch_hcu 模块 + __hcu_version__ 属性 + HYGON/DCU 名字) |
| `src/flagtensor/benchmark_core.py` | `_baseline_module()` / `vendor_baseline_available()` vendor 白名单加 `"hygon"` |
| `tools/run_tests.py` | 全量算子 vendor 白名单加 `"iluvatar"`、`"kunlunxin"`、`"hygon"` |

### 未修改的文件 (零影响确认)

| 文件 | 说明 |
|------|------|
| `src/flagtensor/runtime/common.py` | `vendors.HYGON = 6` 早已注册, `_VENDOR_TORCH_ATTR["hygon"]` 早已存在, `UNSUPPORT_*` 集合不含 HYGON |
| `src/flagtensor/cutensor.py` | cuTensor stub 检测逻辑不变, Hygon 上 `libcutensor.so` 不存在故 `CUTENSOR_AVAILABLE=False` |
| `src/flagtensor/torch_baseline.py` | 6 个基类不变, Hygon baseline 直接复用 |
| `src/flagtensor/torch_npu_baseline.py` | 不变, `CuTensorTrinary` 在 Hygon 上仍可作为 elementwise-trinary 的 fallback (该类只用 torch.* op, 不依赖 torch_npu) |
| `src/flagtensor/testing/assertions.py` | `get_tolerance()` 已通过 `vendor_name != "nvidia"` 处理非 NVIDIA vendor, Hygon 自动适用 |
| `src/flagtensor/runtime/dtype_capability.py` | Hygon 不在 `_VENDOR_DTYPE_SUPPORT` 中, fallback 到 `_DEFAULT_SUPPORTED = {f16, f32, bf16}`, 测试 dtype 集合由 `UNSUPPORT_*` 决定 (Hygon 不在其中, 故支持 fp64/bf16/int64) |
| `src/flagtensor/runtime/backend/_nvidia/` | 不变 |
| `src/flagtensor/runtime/backend/_ppu/` | 不变 |
| `src/flagtensor/runtime/backend/_iluvatar/` | 不变 |
| `src/flagtensor/runtime/backend/_kunlunxin/` | 不变 |

## 6. 运行方式

### 6.1 一键测试命令

```bash
# 全部 36 个算子 (stable + experimental + active)
python tools/run_tests.py --stages all --gpus 0 --output-dir results_hygon

# 仅 stable 算子 (默认)
python tools/run_tests.py --stages stable --gpus 0

# 强制指定 vendor (绕过自动检测)
GEMS_VENDOR=hygon python tools/run_tests.py --stages stable --gpus 0
```

### 6.2 单算子测试

```bash
# accuracy 测试
pytest tests/unary/test_CUTENSOR_OP_ABS.py -v --ref cpu --record json --output results.json

# performance 测试
pytest benchmark/test_CUTENSOR_OP_ABS_perf.py -m CUTENSOR_OP_ABS --mode kernel --level core
```

## 7. ElementwiseTrinary 测试说明

`benchmark/test_ElementwiseTrinary_perf.py` 的 `_resolve_baseline_executor()` 在
`CUTENSOR_AVAILABLE=False` 时 fallback 到 `flagtensor.torch_npu_baseline.CuTensorTrinary`。
该类虽然名字带 `npu`, 但实现只用 `torch.add` / `torch.mul` / `torch.maximum` 等
aten op, **不依赖 torch_npu 模块本身** — 因此在 Hygon 上也能正常工作, 会通过
`torch_hcu` 分发到 DCU vendor kernel。这一既有 fallback 机制让 Hygon 无需修改
`test_ElementwiseTrinary_perf.py` 即可运行 trinary benchmark, 与 Kunlunxin 行为一致。

## 8. 后续扩展指南

### 8.1 容差调优

如果某个算子在 Hygon 上有数值容差问题:

1. 在 `_hygon/tolerances.yaml` 新增 `benchmark_verify_floor_by_op` 段, 列出该 op 的
   override (同时列出 operators.yaml name slug 和 OP_NAME slug 两个 alias)。
2. accuracy 测试 (`tests/`, `--ref cpu`) 仍用 `DEFAULT_CORRECTNESS_TOLERANCES` 的严格
   容差, 不受影响。

### 8.2 Arch 特化

如果未来需要为特定 DCU 变体做 arch 特化 (例如 K100 AI 的特殊 tile 配置):

1. 在 `_hygon/` 下新建 `<arch>/` 子目录, 包含 `heuristics_config_utils.py` 和
   `tune_configs.yaml`。
2. 在 `_hygon/__init__.py` 的 `ARCH_MAP` 把对应 compute capability major 映射到该
   arch 名, `BackendArchEvent` 会自动加载该子目录的配置。
