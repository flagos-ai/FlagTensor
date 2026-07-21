# FlagTensor PPU 平台适配报告

> 适配时间: 2026-07
> 适配目标: Alibaba PPU (PPU-ZW810E, sm80, CUDA 12.9 兼容 SDK)
> 适配结果: 36 个算子全部通过 accuracy + performance 测试，geomean 加速比 2.15x

## 1. 背景

FlagTensor 此前完全基于 NVIDIA GPU + cuTensor 实现：Triton kernel 作为被测对象，cuTensor
作为 vendor baseline。当迁移到 Alibaba PPU 平台时，面临两个核心挑战：

1. **PPU 没有 cuTensor 等价物**：既不存在 `libcutensor`，PPU 原生 SDK 也没有提供
   `libacTensor` 这类 generalized tensor-contraction / elementwise-trinary 库
2. **PPU 设备未被识别**：原 `DeviceDetector` 把 `PPU_SDK` 环境变量误判为 T-Head，
   导致 backend 模块加载失败

本报告描述 FlagTensor 为支持 PPU 平台所做的全部适配工作，包括 baseline 选型调研、
vendor 抽象重构、容差配置体系，以及对 NVIDIA 路径零影响的兼容性保证。

## 2. PPU 环境 baseline 选型调研

### 2.1 PPU SDK 库清单

对 `/usr/local/PPU_SDK` 完整扫描，确认 PPU 提供的 vendor 库如下：

| 库名 | 类型 | 是否可作为 cuTensor 替代 |
|------|------|------------------------|
| `libcutensor` | — | ❌ 不存在 |
| `libacTensor` | — | ❌ 不存在（PPU 原生也无 tensor 库） |
| `libacblas` / `libcublas` (shim) | BLAS | ⚠️ 仅 BLAS，无 elementwise-trinary API；且 ctypes 直调失败 |
| `libacdnn` / `libcudnn` (shim) | DNN | ❌ 不是 contraction/elementwise 库 |
| `libacsparse` / `libcusparse` (shim) | Sparse | ❌ 仅 sparse，无通用 contraction |
| `libacfft` / `libcufft` (shim) | FFT | ❌ 与 tensor primitive 无关 |
| `libacsolver` / `libcusolver` (shim) | Solver | ❌ 与 tensor primitive 无关 |

**结论：PPU 上没有任何与 cuTensor 1:1 对应的库。**

### 2.2 直接 cuBLAS ctypes 不可行

`libcublas.so` 是个 shim，转发到 `libacblas.so`。直接 `ctypes.CDLL('libcublas.so')` 调用
`cublasSgemm_v2` 在 PPU 上失败：

```
[ALINPU ERROR]: HGGC_ERROR_ILLEGAL_ADDRESS at gemm_rtc.h:689
Cannot Find HGresult Map Match, will return ACBLAS_STATUS_INTERNAL_ERROR 700
cublasSgemm status: 14
```

原因：acblas shim 依赖 PyTorch 内部完成的设备上下文初始化，从 Python 端难以复现。

### 2.3 PyTorch 原生 op 走 vendor kernel 的证据

通过 PPU RTC 日志可以确认，PyTorch 原生 op 在 PPU 上确实调用 vendor 优化 kernel：

```
[ALINPU INFO]: Saved_file: /usr/local/PPU_SDK/rtccache/PPU0010/2.0.0-715aa1_*_gemm_*_tile128x256x64*.bin
```

- `torch.matmul` / `torch.addmm` 触发 acblas 生成优化 GEMM kernel（tile 128×256×64，
  缓存在 `/usr/local/PPU_SDK/rtccache/`）
- elementwise op 通过 aten dispatcher 调用 acdnn / acsfu kernel
- batched matmul / einsum 同样走 acblas 路径

### 2.4 Baseline 选型结论

**PyTorch 原生 op 是 PPU 上唯一可行的、正确的 baseline 选择**：

- 它通过 PyTorch C++ dispatch 调用 acblas/acdnn vendor kernel，是真正的 vendor 优化路径
- 与 FlagGems（FlagOS 姊妹项目）的做法一致：对没有 vendor 库的算子用 `torch.*` 作为 baseline
- **这并非 fallback**——PPU 上 PyTorch 原生 op 就是 PPU 的原生 baseline，等同于 NVIDIA 上 cuTensor 的地位

## 3. 适配架构：把 PPU 提升为正式 vendor

为了避免 "fallback 思路"，我们把 PPU 作为正式 vendor 接入 FlagTensor 的 backend 抽象，
与 NVIDIA 平级。每个 vendor 自带：
- `__init__.py`：vendor 元信息 + `BASELINE_AVAILABLE` 标志 + `get_baseline_class()` 工厂
- `baseline.py`：该 vendor 的原生 baseline 实现
- `tolerances.yaml`：该 vendor 的容差配置
- `tune_configs.yaml`：autotune 配置（PPU 复用 nvidia 的 Ampere 配置）

### 3.1 vendors 枚举扩展

`src/flagtensor/runtime/common.py`：

```python
class vendors(Enum):
    NVIDIA = 0
    ...
    THEAD = 14
    PPU = 15      # ← 新增
```

### 3.2 设备检测修正

`src/flagtensor/runtime/backend/device.py`：

- `PPU_SDK` 环境变量 → 返回 `"ppu"`（不再误判为 thead，也不 fallback 到 nvidia）
- 设备名以 `"PPU"` 开头 → 返回 `"ppu"`
- NVIDIA 检测逻辑（`"NVIDIA" in upper_name`）原样保留，互不干扰

### 3.3 PPU backend 模块

新建 `src/flagtensor/runtime/backend/_ppu/`：

```
_ppu/
├── __init__.py                  # vendor_info, ARCH_MAP={'8':'ampere'}, BASELINE_AVAILABLE=True
├── baseline.py                  # PyTorch-native baseline 类（PPU 原生 baseline）
├── tolerances.yaml              # PPU 容差配置
└── heuristics_config_utils.py   # 复用 nvidia 的 heuristics
```

- `ARCH_MAP = {'8': 'ampere'}`：PPU 是 sm80（Ampere-class），复用 NVIDIA Ampere 的
  autotune 配置（`tune_configs.yaml` 通过 `backend_utils.get_tune_config` 的 fallback
  机制从 `_nvidia/` 加载）
- `BASELINE_AVAILABLE = True`：PPU baseline 始终可用（PyTorch 原生 op）

### 3.4 NVIDIA backend 模块对称扩展

`src/flagtensor/runtime/backend/_nvidia/` 同步加入：
- `__init__.py` 新增 `BASELINE_AVAILABLE = CUTENSOR_AVAILABLE` 和 `get_baseline_class()`
- `baseline.py`：将原 `CuTensor*` class 通过 `BASELINE_CLASSES` 字典重新导出
- `tolerances.yaml`：声明 NVIDIA 容差（保持历史值 1e-4）

## 4. Baseline 实现细节

### 4.1 PyTorch-native baseline 模块

`src/flagtensor/torch_baseline.py` 实现了 6 个基类，完整对齐 `CuTensor*` 的公开接口
（`prepare` / `__call__` / `build_kernel_callable`）：

| 基类 | 对应的 CuTensor 类 | 语义 |
|------|-------------------|------|
| `TorchUnaryBaseline` | `CuTensorUnary` | `y = alpha * op(x)` |
| `TorchBinaryBaseline` | `CuTensorBinary` | `y = op_AB(alpha * x, gamma * y)` |
| `TorchTrinaryBaseline` | `CuTensorTrinary` | `y = op_ABC(op_AB(alpha*op_A(x), beta*op_B(y)), gamma*op_C(z))` |
| `TorchContractionBaseline` | `CuTensorContraction` | `d = alpha * einsum(a, b, modes) + beta * c` |
| `TorchContractionTrinaryBaseline` | `CuTensorContractionTrinary` | `e = alpha * (a@b@c) + beta * d` |
| `TorchBlockSparseContractionBaseline` | `CuTensorBlockSparseContraction` | dense matmul + sparsity mask |

所有类都支持 mode permutation（高维 indexed 场景），并通过 `torch.matmul` /
`torch.einsum` / elementwise op 调用 PPU vendor kernel。

### 4.2 PPU baseline registry

`_ppu/baseline.py` 为每个算子提供一个具名子类，注册到 `BASELINE_CLASSES` 字典：

```python
BASELINE_CLASSES = {
    'abs': BaselineAbs,
    'add': BaselineAdd,
    ...
    'contraction': BaselineContraction,
    'contraction_trinary': BaselineContractionTrinary,
    'elementwise_trinary': BaselineElementwiseTrinary,
    'block_sparse_contraction': BaselineBlockSparseContraction,
}
```

同时导出 `elementwise_trinary` 函数和 `_get_trinary_executor` 工厂，对齐
`flagtensor.cutensor` 的函数式 API（`test_ElementwiseTrinary_perf.py` 使用）。

### 4.3 NVIDIA baseline registry

`_nvidia/baseline.py` 将原 `CuTensor*` class 通过相同结构重新导出：

```python
_C = (lambda slug, cls: cls) if CUTENSOR_AVAILABLE else (lambda slug, cls: None)

BASELINE_CLASSES = {
    'abs': _C('abs', CuTensorAbs),
    ...
    'contraction': _C('contraction', CuTensorContraction),
    ...
}
```

当 cuTensor 不可用时，所有条目解析为 `None`，benchmark 测试 skip with "baseline
unavailable"——**与重构前的 NVIDIA 行为完全一致**。

### 4.4 BlockSparseTensorContraction 修复

`src/flagtensor/cutensor.py` 中 `BlockSparseTensorContraction.__call__` 原本在
cuTensor 不可用时仍调用 `CuTensorBlockSparseContraction`（崩溃）和 `contraction()`
（依赖 cuTensor）。修复为：

- cuTensor 可用：走原 cuTensor 路径（NVIDIA 行为不变）
- cuTensor 不可用：走 dense `torch.matmul` + sparsity mask 路径（PPU 路径）

通过 lazy import 的 `_get_torch_contraction_baseline_cls()` /
`_get_torch_block_sparse_baseline_cls()` 避免循环依赖。

## 5. 容差配置体系

### 5.1 问题背景

PPU 上 acblas 与 Triton kernel 使用不同的 GEMM summation order，对 contraction-family
算子产生两类误差：

1. **常规误差**：~1e-3 绝对误差（dtype default 是 1.3e-6，需要放宽）
2. **近零元素放大误差**：当输出元素 |out| < 1 时，cancellation 把绝对误差放大到
   ~3e-3，超过 1e-3 的常规 floor

NVIDIA 上 cuTensor 和 Triton 共享相同 GEMM tiling，不存在此问题。

### 5.2 Vendor+Op 双层容差配置

每个 vendor 在 `_<vendor>/tolerances.yaml` 声明容差：

```yaml
# vendor-wide default floor
benchmark_verify_floor:
  atol: 1.0e-3
  rtol: 1.0e-3

# per-op-category override (optional)
benchmark_verify_floor_by_op:
  contraction:
    atol: 5.0e-3
    rtol: 1.0e-3
  contraction_trinary:
    atol: 5.0e-3
    rtol: 1.0e-3
  block_sparse_contraction:
    atol: 5.0e-3
    rtol: 1.0e-3
```

### 5.3 各 vendor 容差配置

| Vendor | 默认 atol/rtol | Contraction 类 atol | 说明 |
|--------|---------------|---------------------|------|
| NVIDIA | 1e-4 | 1e-4 | 历史值，完全保留 |
| PPU | 1e-3 | 5e-3 | acblas summation order 差异 |
| 未知 vendor | 1e-4 | 1e-4 | 向后兼容 fallback |

### 5.4 加载机制

`src/flagtensor/testing/assertions.py` 的 `get_vendor_benchmark_floor(vendor, op_slug)`：

1. 读取 `_<vendor>/tolerances.yaml`（带缓存）
2. 优先返回 `benchmark_verify_floor_by_op.<op_slug>` 覆盖
3. 否则返回 `benchmark_verify_floor` 默认值
4. yaml 缺失时 fallback 到 (1e-4, 1e-4) 保持向后兼容

`Benchmark.verify()` 调用时传入 `op_slug=self._get_op_slug()`，让 per-op override
自动生效。

### 5.5 Op-slug 别名处理

`ContractionTrinary` 算子在 operators.yaml 中名为 `ContractionTrinary`（slug
`contraction_trinary`），但其 `OP_NAME` 是 `CUTENSOR_OP_TENSOR_CONTRACTION_TRINARY`
（slug `tensor_contraction_trinary`）。yaml 中同时列出两个 alias key，保证无论从哪个
入口计算 slug 都能命中 override：

```yaml
benchmark_verify_floor_by_op:
  contraction_trinary:           # operators.yaml name slug
    atol: 5.0e-3
    rtol: 1.0e-3
  tensor_contraction_trinary:    # OP_NAME slug
    atol: 5.0e-3
    rtol: 1.0e-3
```

## 6. Benchmark harness vendor-aware 改造

### 6.1 benchmark_core.py

- 新增 `_get_vendor_baseline_class(op_slug)`：通过 `flagtensor.runtime.device.vendor_name`
  加载对应 vendor 的 `BASELINE_CLASSES` 字典
- 新增 `get_baseline_class(op_name)`：公开 API，支持 CamelCase → snake_case 转换
  （`ElementwiseTrinary` → `elementwise_trinary`）
- 新增 `get_baseline_module()`：返回当前 vendor 的 baseline 模块（用于函数式 API）
- 新增 `vendor_baseline_available()`：公开 API，供 benchmark test 文件做 skip 判断
- `Benchmark.cutensor_available` 改为 `_vendor_baseline_available()` 计算
- `Benchmark.verify()` 改用 vendor+op-aware 容差
- `_get_baseline_instance()` 改用 `get_baseline_class(slug)` 加载

### 6.2 benchmark test 文件

36 个 benchmark test 文件统一改造：

```python
# 改造前
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorAbs
...
if not CUTENSOR_AVAILABLE:
    pytest.skip("cuTensor unavailable")
...
baseline = CuTensorAbs(dtype=x.dtype)

# 改造后
from flagtensor.benchmark_core import (
    Benchmark, BenchmarkConfig,
    get_baseline_class, vendor_baseline_available,
)
...
if not vendor_baseline_available():
    pytest.skip("baseline unavailable")
...
baseline = get_baseline_class(OP_NAME)(dtype=x.dtype)
```

- `test_ElementwiseTrinary_perf.py`：函数式 API 改用 `get_baseline_module().elementwise_trinary`
  和 `get_baseline_module()._get_trinary_executor`
- `test_ElementwiseTrinary_perf.py`：顺带修复预存的拼写错误
  `cutensor_elementwise_elementwise_trinary` → `cutensor_elementwise_trinary`
  （之前 cuTensor skip 掩盖了这个 bug）

## 7. NVIDIA 兼容性保证

本次适配对 NVIDIA 路径**完全零影响**，验证如下：

### 7.1 cutensor.py

- 完全恢复原状，0 处 `_delegate` 残留
- 所有 `if not CUTENSOR_AVAILABLE: return` 逻辑保持不变
- `BlockSparseTensorContraction` 修复仅在 `CUTENSOR_AVAILABLE=False` 时生效
  （`_supports_cutensor` 返回 False 才走新 dense fallback）

### 7.2 device.py

- PPU 检测是新增的独立分支（`upper_name.startswith("PPU")` → "ppu"）
- NVIDIA 检测原样保留（`"NVIDIA" in upper_name` → "nvidia"）
- 两个分支互不干扰

### 7.3 assertions.py

- NVIDIA 容差完全保留（1e-4，从 `_nvidia/tolerances.yaml` 加载）
- PPU 用独立的 1e-3 / 5e-3 容差
- 未知 vendor fallback 到 1e-4（向后兼容）

### 7.4 benchmark_core.py

- `vendor_baseline_available()` 在 NVIDIA+cuTensor 上返回 True（同原 `CUTENSOR_AVAILABLE=True`）
- 在 NVIDIA 无 cuTensor 上返回 False（同原行为，benchmark skip）
- `get_baseline_class()` 在 NVIDIA 上返回 `CuTensorXxx`（同原 class）

### 7.5 benchmark test 文件

- `vendor_baseline_available()` 在 NVIDIA+cuTensor 上为 True，不 skip（同原行为）
- `get_baseline_class(OP_NAME)` 在 NVIDIA 上返回 `CuTensorXxx`（同原 class）
- skip 消息从 "cuTensor unavailable" 改为 "baseline unavailable"（仅文案变化）

## 8. 适配文件清单

### 新增文件

| 文件 | 作用 |
|------|------|
| `src/flagtensor/torch_baseline.py` | PyTorch-native baseline 6 个基类 |
| `src/flagtensor/runtime/backend/_ppu/__init__.py` | PPU vendor 模块 |
| `src/flagtensor/runtime/backend/_ppu/baseline.py` | PPU baseline 类 + registry |
| `src/flagtensor/runtime/backend/_ppu/tolerances.yaml` | PPU 容差配置 |
| `src/flagtensor/runtime/backend/_ppu/heuristics_config_utils.py` | 复用 nvidia heuristics |
| `src/flagtensor/runtime/backend/_nvidia/baseline.py` | NVIDIA baseline registry |
| `src/flagtensor/runtime/backend/_nvidia/tolerances.yaml` | NVIDIA 容差配置 |
| `docs/ppu_adaptation.md` | 本文档 |

### 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/flagtensor/runtime/common.py` | `vendors` 枚举加 `PPU = 15` |
| `src/flagtensor/runtime/backend/device.py` | PPU 检测返回 "ppu"（不再 fallback） |
| `src/flagtensor/runtime/backend/backend_utils.py` | `get_tune_config` fallback 到 nvidia |
| `src/flagtensor/runtime/backend/_nvidia/__init__.py` | 加 `BASELINE_AVAILABLE` + `get_baseline_class` |
| `src/flagtensor/cutensor.py` | `BlockSparseTensorContraction` cuTensor 不可用时的 dense fallback |
| `src/flagtensor/testing/assertions.py` | 加 `get_vendor_benchmark_floor(vendor, op_slug)` |
| `src/flagtensor/benchmark_core.py` | vendor-aware baseline 加载 + verify 容差 |
| `benchmark/test_*_perf.py` (36 个) | 改用 `get_baseline_class` + `vendor_baseline_available` |

## 9. 测试结果

### 9.1 一键测试命令

```bash
# 全部 36 个算子（stable + experimental + active）
python tools/run_tests.py --stages all --gpus 0,1 --output-dir results

# 仅 stable 算子（默认）
python tools/run_tests.py --stages stable --gpus 0,1
```

### 9.2 PPU 上的最终结果

```
Device:    PPU-ZW810E (Alibaba PPU, sm80, CUDA-compatible)
PyTorch:   2.9.0
Triton:    3.5.0+ppu2.0.0.oe
Baseline:  PyTorch native ops (vendor acblas/acdnn on PPU)

Accuracy:    36/36 passed
Performance: 36/36 passed
GEOMEAN:     2.15x speedup vs PPU-native baseline
```

### 9.3 各算子加速比

| 算子类别 | 算子数 | 加速比范围 | 说明 |
|---------|-------|-----------|------|
| Elementwise Trinary | 1 | 4.57x | `op_ABC(op_AB(...))` 三元融合 |
| Unary | 28 | 1.15x ~ 3.62x | pointwise math ops |
| Binary | 4 | 2.65x ~ 2.75x | add/mul/max/min |
| Contraction | 1 | 1.04x | Triton GEMM vs acblas GEMM，接近持平 |
| ContractionTrinary | 1 | 1.76x | chain matmul (A@B)@C |
| BlockSparseContraction | 1 | 1.82x | block-sparse GEMM |

### 9.4 重点观察

- **Elementwise 算子加速最明显**（1.15x ~ 4.57x）：Triton kernel 融合度高于
  PyTorch elementwise dispatch
- **Contraction 接近持平**（1.04x）：Triton GEMM 与 acblas GEMM 都使用 sm80
  Tensor Core，性能相当
- **Identity 仅 1.15x**：纯拷贝算子受 memory bandwidth 限制，加速空间有限

## 10. 后续扩展指南

### 10.1 新增 vendor 接入

接入新 vendor（如 AMD、Cambricon）只需 4 步：

1. `src/flagtensor/runtime/common.py` 的 `vendors` 枚举加新条目
2. 新建 `src/flagtensor/runtime/backend/_<vendor>/` 目录，包含：
   - `__init__.py`：`vendor_info`、`ARCH_MAP`、`BASELINE_AVAILABLE`、`get_baseline_class()`
   - `baseline.py`：该 vendor 的原生 baseline 类 + `BASELINE_CLASSES` registry
   - `tolerances.yaml`：该 vendor 的容差配置
   - `tune_configs.yaml`（可选）：autotune 配置
3. `src/flagtensor/runtime/backend/device.py` 的检测逻辑加该 vendor 的识别分支
4. （可选）`src/flagtensor/runtime/backend/_<vendor>/<arch>/` 加 arch-specific 配置

### 10.2 新增算子

新增算子时，每个 vendor 的 `baseline.py` 需要在 `BASELINE_CLASSES` 字典加一行：

```python
BASELINE_CLASSES = {
    ...
    '<new_op_slug>': <NewOpBaselineClass>,
}
```

benchmark test 文件用 `get_baseline_class(OP_NAME)` 加载即可，无需关心 vendor 差异。

### 10.3 容差调优

如果新算子在某个 vendor 上有数值容差问题：

1. 在 `_<vendor>/tolerances.yaml` 的 `benchmark_verify_floor_by_op` 加该 op 的 override
2. 同时列出 operators.yaml name slug 和 OP_NAME slug 两个 alias
3. accuracy 测试（`tests/`）仍用 `DEFAULT_CORRECTNESS_TOLERANCES` 的严格容差，不受影响

## 11. 参考资料

- FlagGems baseline 约定：https://github.com/flagos-ai/FlagGems
- cuTensor 文档：https://docs.nvidia.com/cuda/cutensor/
- PPU SDK 文档：`/usr/local/PPU_SDK/` 内的 README 与 samples
- 项目内文档：
  - `docs/acceptance/benchmark_policy.md`：benchmark 容差策略
  - `docs/acceptance/accuracy_policy.md`：accuracy 容差策略
  - `conf/operators.yaml`：算子注册表（36 个算子）
