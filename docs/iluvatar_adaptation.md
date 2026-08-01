# FlagTensor Iluvatar (天数智芯) 平台适配报告

> 适配时间: 2026-07
> 适配目标: Iluvatar CoreX BI-V150 (CUDA 兼容, CoreX SDK 4.4.0)
> 适配结果: 36 个算子全部通过 accuracy + performance 测试, geomean 加速比 1.95x

## 1. 背景

FlagTensor 此前支持 NVIDIA (cuTensor 作为 vendor baseline) 和 Alibaba PPU
(PyTorch 原生 op 作为 baseline, 见 `docs/ppu_adaptation.md`)。迁移到 Iluvatar
天数智芯平台时面临的核心问题与 PPU 类似但有新的变化:

1. **天数没有可用的 cuTensor 库作为 baseline**: CoreX SDK 在
   `/usr/local/corex/lib64/` 下提供了一个**占位性质**的 `libcutensor.so` —
   它能被 `ctypes.CDLL` 成功加载, `cutensorCreate` 也返回成功, 但缺少
   `CUTENSOR_COMPUTE_DESC_16F/16BF/64F` 等数据符号, `cutensorGetVersion`
   返回无意义的 `1`。它是一个 stub, 不能作为 baseline 使用。
2. **该 stub 会导致 `import flagtensor` 直接崩溃**: 原 `cutensor.py` 在
   `ctypes.CDLL` 成功后立即通过 `c_void_p.in_dll` 绑定
   `CUTENSOR_COMPUTE_DESC_*` 符号, 在 Iluvatar 上抛
   `ValueError: undefined symbol`, 使整个包无法导入。
3. **性能数据必须完整**: 尽管没有 cuTensor baseline, benchmark 仍要产出
   完整的 base/gems/speedup 数据, 且 `tools/run_tests.py` 的逻辑与结果
   导出方式保持不变。

## 2. Baseline 选型

与 PPU 的结论一致: **PyTorch 原生 op 是 Iluvatar 上唯一可行且正确的
baseline**。`torch.matmul` / `torch.einsum` / elementwise aten op 通过 CoreX
软件栈分发到天数 vendor 库的优化 kernel, 其在 Iluvatar 上的地位等同于
NVIDIA 上的 cuTensor。这不是 fallback, 而是 vendor 原生优化路径。

实测确认 PyTorch 原生路径正确可用 (`torch.matmul`/`einsum`/elementwise 均
在 BI-V150 上正常执行), Triton 3.1.0 kernel 也可正常编译运行。

## 3. 适配内容

### 3.1 cuTensor stub 检测 (`src/flagtensor/cutensor.py`)

在 `ctypes.CDLL("libcutensor.so")` 成功后、符号绑定之前, 增加 stub 检测:
逐一校验本模块 import 时需要的 4 个 `CUTENSOR_COMPUTE_DESC_*` 数据符号与
`cutensorCreate`/`cutensorDestroy` 入口是否存在; 任一缺失即判定为占位库,
置 `CUTENSOR_AVAILABLE = False`。

对 NVIDIA **零影响**: 这些符号正是原代码在 import 时无条件绑定的符号,
NVIDIA 完整版 cuTensor 全部具备, 守卫是 no-op; 对 PPU (SDK 无
libcutensor) 同样零影响 — `CDLL` 本来就失败。

`CUTENSOR_AVAILABLE = False` 后, `BlockSparseTensorContraction` 等自动走
PPU 适配时引入的 dense `torch.matmul` 路径, 无需额外修改。

### 3.2 Iluvatar vendor 后端模块 (`src/flagtensor/runtime/backend/_iluvatar/`)

按 PPU 建立的 vendor 接入模式新增:

```
_iluvatar/
├── __init__.py                  # vendor_info(ixsmi), ARCH_MAP={}, BASELINE_AVAILABLE=True
├── baseline.py                  # PyTorch-native baseline 类 + BASELINE_CLASSES registry
├── tolerances.yaml              # Iluvatar 容差配置
└── heuristics_config_utils.py   # elementwise heuristics (与 PPU/NVIDIA 相同)
```

- `vendor_info`: `vendor_name='iluvatar'`, `device_name='cuda'` (Iluvatar 通过
  `torch.cuda` 驱动), `device_query_cmd='ixsmi'` (CoreX SDK 自带)。
- `ARCH_MAP = {}`: 不做 arch 特化; vendor 级 tune configs 经
  `backend_utils.get_tune_config` 的既有 fallback 机制复用 NVIDIA 配置。
- `baseline.py`: 复用 `flagtensor.torch_baseline` 的 6 个基类, 为 36 个算子
  注册 `BASELINE_CLASSES`, 并导出 `elementwise_trinary` /
  `_get_trinary_executor` 函数式 API (与 `_ppu/baseline.py` 同构)。
- `vendors` 枚举已有 `ILUVATAR = 3`, `_VENDOR_TORCH_ATTR` 已有
  `"iluvatar": "corex"`, 均无需改动。

### 3.3 设备检测 (`src/flagtensor/runtime/backend/device.py`)

- Layer 2 快速检测: `torch.corex` 属性命中 → `"iluvatar"` (既有逻辑, 自动生效)。
- CUDA 设备名回退分支新增: 设备名含 `"ILUVATAR"` (如 "Iluvatar BI-V150")
  → `"iluvatar"`, 插在 PPU 分支之后、NVIDIA 分支之前。NVIDIA 设备名不含
  "ILUVATAR", PPU 设备名以 "PPU" 开头先行命中, 两个分支互不干扰。

### 3.4 容差配置 (`_iluvatar/tolerances.yaml`)

实测 BI-V150 上 baseline (CoreX vendor GEMM) 与 Triton kernel 的差异
(fp32, 取 benchmark 最大 shape):

| 算子 | shape | max_abs | max_rel |
|------|-------|---------|---------|
| Contraction | 512x256 @ 256x512 | 7.6e-6 | 2.3e-4 |
| ContractionTrinary | 512x256 @ 256x384 @ 384x512 | 2.4e-4 | 1.5e-5 |

结论: 采用 vendor 级 `benchmark_verify_floor: atol=1e-3, rtol=1e-3`
(对最差观测值保留 ~4x 余量, 同时足够紧以捕获真实回归)。与 PPU 不同,
**不需要** contraction 家族的 per-op override — 天数 GEMM 与 Triton
GEMM 的 summation order 差异远小于 PPU acblas。

accuracy 测试 (`tests/`, `--ref cpu`) 仍使用严格容差 (fp32: 1.3e-6), 不受影响。

### 3.5 并发修复: `_gen_*.py` 原子写入 (`src/flagtensor/utils/unary_pointwise.py`)

`tools/run_tests.py --gpus 0,1,2,3` 每 GPU 一个 worker 进程。 unary 算子的
Triton kernel 在 import 时生成并**覆写**共享的
`utils/__kernels__/_gen_<op>.py`; 多进程并发时, 一个进程可能读到另一个
进程写了一半的文件, 导致 `@triton.jit` 内 `inspect.getsourcelines` 抛
`OSError: could not get source code` (全量测试中 CUTENSOR_OP_CEIL 曾因此
benchmark 失败)。

修复: 生成内容不变时跳过写入; 需要写入时先写临时文件再 `os.replace`
原子替换。生成内容是确定性的, 各后端行为完全不变。该 bug 为平台无关的
预存并发缺陷, 在 4 卡 Iluvatar 机器上被稳定复现并修复。

## 4. NVIDIA / PPU 兼容性保证

| 文件 | 改动 | 对 NVIDIA/PPU 的影响 |
|------|------|---------------------|
| `cutensor.py` | stub 符号检测 | NVIDIA: 完整库符号齐全, 守卫 no-op; PPU: 无库, 原本就不可用 |
| `device.py` | `"ILUVATAR"` 名称分支 | NVIDIA/PPU 名称不命中该分支, 路由不变 (已用模拟名称验证) |
| `unary_pointwise.py` | 原子写入 | 生成内容逐字节不变, 仅写入方式变为原子 |
| `_iluvatar/` | 新增目录 | 仅当 vendor 检测为 iluvatar 时加载 |
| `ops/CUTENSOR_OP_GETT.py` | iluvatar 门控的 extra configs + frozen runner | 门控关闭时: config 列表与 launcher 代码路径逐字节不变 |
| `benchmark/test_ContractionTrinary_perf.py` | fp32 分支计时 two-step (对齐 op 实际派发) | **测量值**在 NVIDIA 上亦变化 (反映生产路径而非已禁用的 fused kernel); 运行时功能不变 |
| `tools/run_tests.py` | **未改动** | 逻辑与结果导出方式完全不变 |

已验证: `_nvidia` / `_ppu` vendor 模块均可正常加载; NVIDIA 无 cuTensor 时
`BASELINE_AVAILABLE=False`、baseline class 为 `None` (与历史行为一致)。

## 5. 测试结果

### 5.1 一键测试命令 (与 NVIDIA/PPU 完全相同)

```bash
# 全部 36 个算子
python tools/run_tests.py --stages all --gpus 0,1,2,3 --output-dir results

# 仅 stable 算子 (默认)
python tools/run_tests.py --gpus 0
```

### 5.2 Iluvatar 上的最终结果

```
Device:    Iluvatar BI-V150 x4 (CoreX SDK 4.4.0, CUDA-compatible)
PyTorch:   2.7.1 (torch.corex)
Triton:    3.1.0
Baseline:  PyTorch native ops (CoreX vendor libraries)

Accuracy:    36/36 passed
Performance: 36/36 passed
GEOMEAN:     1.95x speedup vs Iluvatar-native baseline
```

### 5.3 各类算子加速比

| 算子类别 | 算子数 | 加速比 | 说明 |
|---------|-------|--------|------|
| Unary | 28 | 1.08x ~ 3.34x | pointwise math ops |
| Binary | 4 | 2.31x ~ 2.41x | add/mul/max/min |
| ElementwiseTrinary | 1 | 3.97x | 三元融合 |
| Contraction | 1 | 1.66x (2D shapes 1.92x) | 调优后, 详见第 6 节 |
| ContractionTrinary | 1 | 2.21x (fp32 block) | two-step GETT, 详见第 6 节 |
| BlockSparseContraction | 1 | 1.75x | block-sparse GEMM |

### 5.4 重点观察

- **Elementwise 算子加速显著** (1.08x ~ 3.97x): Triton kernel 融合度高于
  PyTorch elementwise dispatch。
- **Contraction 家族经调优后全部反超 baseline** (1.66x / 2.21x): 初版适配时
  Contraction 仅 0.51x — 根因不是 kernel GPU 时间而是 CoreX Triton fork 的
  启动开销, 详见第 6 节。
- **Identity 仅 1.08x**: 纯拷贝算子受 memory bandwidth 限制。
- 唯一仍低于 1x 的 shape: Contraction 的 `(256,1024)x(1024,128)` (K=1024
  深约减, 0.91x) — 该 shape 接近 memory-bound, vendor GEMM 有优势。

## 6. Contraction 性能调优 (2026-07 第二轮)

初版适配中 Contraction 0.51x / ContractionTrinary 0.65x, 是仅有的两个
低于 1x 的算子。第二轮调优将其提升至 1.66x / 2.21x, 全部改动仅在
iluvatar vendor 激活时生效 (NVIDIA/PPU 行为逐字节不变)。

### 6.1 根因分析

**(a) 2D GEMM 配置不适配 BI-V150。** `_gett_kernel` 的 12 个 autotune
配置面向 NVIDIA (Ampere/Hopper) 设计。BI-V150 只有 16 个 SM、
warp_size=64、128KB smem/SM, 独立扫描全部 benchmark shapes 后确认
32x32 小 tile + 少 pipeline stage 的组合几乎全面占优
(如 `(32,32,32,G8,W4,S2)` 在 `(256,128)x(128,256)` 上比原最优快 1.36x)。

**(b) "batched" shapes 的 0.47x 是启动开销假象。** 这些 shape 走
`contraction()` 的 ND reshape 路径 (单次 2D kernel 启动), 但 benchmark
对其使用 operator 模式 (host 墙钟计时)。实测分解
(shape `(8,64,32)x(32,48)`):

| 组成 | 耗时 |
|------|------|
| contraction() 完整调用 | 0.236 ms |
| 其中 kernel GPU 时间 | 0.009 ms |
| 其中 JITFunction.run 派发 (CoreX fork) | ~0.15 ms |
| 其中 Python glue | ~0.06 ms |
| baseline (einsum, host 计时) | 0.062 ms |

CoreX Triton 3.1 fork 的 `JITFunction.run` 每次调用重新做 binder、
字符串化 cache key、特化计算, 单次 ~0.15 ms, 是 GPU 时间的 17 倍。
do_bench (kernel 模式) 用 CUDA event 测量可以掩盖该开销, host 计时
(operator 模式) 则完全暴露。

**(c) ContractionTrinary benchmark 计时的是已禁用的 fused kernel。**
op 内 `_supports_fused_triton_trinary()` 恒为 False (fused 在中大尺寸
严重回退, 生产路径是 two-step GETT), 但 benchmark 的 fp32 分支仍计时
`_launch_fused_trinary_kernel` — 测量的是死代码路径。

### 6.2 方案选型: 为什么是 CompiledKernel runner 而不是 CUDA graph

| 方案 | host 开销 | do_bench (kernel 模式) | 结论 |
|------|----------|------------------------|------|
| 原 JITFunction.run 派发 | 0.148 ms | 0.0073 ms (GPU 时间) | host 路径不可接受 |
| CUDA graph replay | 0.036 ms | 0.0689 ms | CoreX 驱动 graph 启动有 ~0.06 ms GPU 侧固定开销, kernel 模式回退 10x |
| CompiledKernel runner | 0.012 ms | 0.0073 ms | 两种计时模式下均最优 |

### 6.3 落地改动 (`src/flagtensor/ops/CUTENSOR_OP_GETT.py`, 全部 iluvatar 门控)

1. **`_GETT_ILUVATAR_EXTRA_CONFIGS`**: 4 个 BI-V150 标定配置
   (`32x32x32/G8/W4/S2`, `32x32x32/G4/W4/S3`, `64x64x32/G8/W4/S2`,
   `128x128x32/G8/W8/S2`), 仅 iluvatar 激活时追加进 autotune 空间;
   其他 vendor 的 config 列表逐字节不变。
2. **Prepared launcher 冻结机制**: 首次调用走标准 autotuner (完成调优并
   经 `best_config` 读出胜方), 随即将该 config 以
   `JITFunction.run(warmup=True)` 编译并包装成 `CompiledKernel[grid]`
   runner, 后续调用 ~0.012 ms/次。runner 按指针 16B 对齐特化键缓存;
   任何异常永久回退标准 autotuner 路径。CUDA graph 方案经实测被否决
   (见 6.2)。
3. **`benchmark/test_ContractionTrinary_perf.py`**: fp32 分支从 fused
   kernel 改为计时 two-step GETT — 与 op 的实际派发路径、与该文件
   fp16 分支完全一致。**注意**: 该改动同样影响 NVIDIA 上此算子的
   测量值 (使其反映生产路径而非死代码), 但不改变任何后端的运行时
   功能。

### 6.4 调优后数据 (fp32, vs PyTorch-native baseline)

| 算子 | 调优前 | 调优后 |
|------|--------|--------|
| Contraction — 12 个 2D shapes 平均 | ~1.5x (最低 0.58x) | 1.92x (最低 0.91x) |
| Contraction — 9 个 batched shapes 平均 | 0.47x | 1.66x |
| ContractionTrinary — fp32 block 平均 | 0.65x | 2.21x |
| 全量 GEOMEAN (36 ops) | 1.82x | 1.95x |

正确性: 36/36 accuracy + 36/36 performance 全过; 另对 frozen-runner
路径做了 8 轮"原地修改输入后复算"的专项验证 (2D / batched / trinary,
含 alpha/beta 变键) 全部正确。

## 7. 适配文件清单

### 新增文件

| 文件 | 作用 |
|------|------|
| `src/flagtensor/runtime/backend/_iluvatar/__init__.py` | Iluvatar vendor 模块 |
| `src/flagtensor/runtime/backend/_iluvatar/baseline.py` | baseline 类 + registry |
| `src/flagtensor/runtime/backend/_iluvatar/tolerances.yaml` | 容差配置 (1e-3 floor) |
| `src/flagtensor/runtime/backend/_iluvatar/heuristics_config_utils.py` | elementwise heuristics |
| `docs/iluvatar_adaptation.md` | 本文档 |

### 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/flagtensor/cutensor.py` | 占位 libcutensor (缺符号) 判定为不可用 |
| `src/flagtensor/runtime/backend/device.py` | 设备名含 "ILUVATAR" → "iluvatar" |
| `src/flagtensor/utils/unary_pointwise.py` | `_gen_*.py` 原子写入, 修复多进程竞态 |
| `src/flagtensor/ops/CUTENSOR_OP_GETT.py` | iluvatar 门控: 4 个 BI-V150 标定 GEMM config + prepared-launcher 冻结 CompiledKernel runner |
| `benchmark/test_ContractionTrinary_perf.py` | fp32 分支计时 two-step GETT (对齐 op 实际派发路径) |

`tools/run_tests.py` 与结果导出格式 **零改动**。

## 8. 后续工作建议

- **Contraction 单 shape 微调**: `(256,1024)x(1024,128)` (K=1024) 仍为
  0.91x, 可针对深 K 场景增补 config (如更大 BLOCK_K / split-K)。
- **Python glue 优化**: batched shapes 的 host 计时中 `contraction()` 的
  mode 规范化仍有 ~0.06 ms, 可加 plan cache (参照 trinary 的
  `_trinary_plan_key` 模式) 进一步压缩。
- 新增算子时, 在 `_iluvatar/baseline.py` 的 `BASELINE_CLASSES` 加一行注册
  (见 `docs/ppu_adaptation.md` 第 10 节扩展指南)。

## 9. 参考资料

- PPU 适配报告: `docs/ppu_adaptation.md`
- CoreX SDK: `/usr/local/corex/` (设备查询: `ixsmi`)
- `docs/acceptance/benchmark_policy.md`, `docs/acceptance/accuracy_policy.md`
