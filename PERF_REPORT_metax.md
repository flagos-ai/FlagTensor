# FlagTensor 性能测试报告 — MetaX C550

**测试时间**: 2026-08-16 17:23:39  
**硬件**: MetaX C550 × 8 (64GB/卡)  
**后端**: MetaX MACA SDK (vendor=metax, device=cuda)  
**PyTorch**: 2.10.0+cpu (torch+metax plugin)  
**Triton**: 3.6.0+metax  
**FlagTensor**: 0.1.0  
**Python**: 3.10.12 | **OS**: Linux 5.15.0-119-generic | **arch**: x86_64

## 1. 总览

| 指标 | 结果 |
|---|---|
| 算子总数 | 36 |
| Accuracy 通过 | **36/36** ✅ |
| Performance 通过 | **36/36** ✅ |
| 失败 / 跳过 / 错误 | 0 / 0 / 0 |
| Baseline | PyTorch 原生算子 (经 MACA 派发到 mcblas/mcdnn 厂商库) |
| 测试启动 | `python3 tools/run_tests.py --ops <36 ops> --gpus 0,1,2,3,4,5,6,7` |
| 总耗时 | ~4 分钟 (8卡并行, wall-clock) |

## 2. 性能汇总（按类别）

| 类别 | 算子数 | 平均加速比 | 加速比范围 |
|---|---|---|---|
| unary | 28 | 1.28x | [1.01, 1.77] |
| binary | 4 | 1.44x | [1.43, 1.46] |
| contraction | 3 | 1.79x | [0.97, 2.77] |
| sparse | 1 | 2.13x | [2.13, 2.13] |
| **合计** | **36** | **1.37x** | **[0.97, 2.77]** |

## 3. 全算子性能明细

> 每个算子取**最大 shape** 的代表数据点（各 dtype 分别列出）。`avg` 为该算子所有 shape/dtype 的平均加速比。

| 算子 | 类别 | dtype | 代表 shape | Triton (ms) | Baseline (ms) | 加速比 | 算子平均 |
|---|---|---|---|---|---|---|---|
| ElementwiseTrinary | contraction | bf16 | 1048576 | 0.0230 | 0.0932 | 4.04x | 2.77x |
|  |  | fp16 | 1048576 | 0.0230 | 0.0914 | 3.97x |  |
|  |  | fp32 | 1048576 | 0.0274 | 0.1111 | 4.06x |  |
| BlockSparseContraction | sparse | fp32 | 256,128 | 0.2749 | 0.8881 | 3.23x | 2.13x |
| CUTENSOR_OP_MISH | unary | bf16 | 8388608 | 0.0620 | 0.1743 | 2.81x | 1.77x |
|  |  | fp16 | 8388608 | 0.0576 | 0.1659 | 2.88x |  |
|  |  | fp32 | 8388608 | 0.0632 | 0.2404 | 3.80x |  |
| CUTENSOR_OP_SOFT_SIGN | unary | bf16 | 8388608 | 0.0440 | 0.1262 | 2.87x | 1.70x |
|  |  | fp16 | 8388608 | 0.0407 | 0.1221 | 3.00x |  |
|  |  | fp32 | 8388608 | 0.0599 | 0.2255 | 3.76x |  |
| ContractionTrinary | contraction | fp32 | 512,256 | 0.0961 | 0.1547 | 1.61x | 1.63x |
| CUTENSOR_OP_MAX | binary | bf16 | 8388608 | 0.0486 | 0.0986 | 2.03x | 1.46x |
|  |  | fp16 | 8388608 | 0.0481 | 0.0968 | 2.01x |  |
|  |  | fp32 | 8388608 | 0.0812 | 0.1792 | 2.21x |  |
| CUTENSOR_OP_MIN | binary | bf16 | 8388608 | 0.0499 | 0.0996 | 1.99x | 1.45x |
|  |  | fp16 | 8388608 | 0.0492 | 0.0978 | 1.99x |  |
|  |  | fp32 | 8388608 | 0.0822 | 0.1807 | 2.20x |  |
| CUTENSOR_OP_MUL | binary | bf16 | 8388608 | 0.0492 | 0.0986 | 2.01x | 1.43x |
|  |  | fp16 | 8388608 | 0.0489 | 0.0975 | 1.99x |  |
|  |  | fp32 | 8388608 | 0.0819 | 0.1777 | 2.17x |  |
| CUTENSOR_OP_ADD | binary | bf16 | 8388608 | 0.0497 | 0.0980 | 1.97x | 1.43x |
|  |  | fp16 | 8388608 | 0.0484 | 0.0970 | 2.01x |  |
|  |  | fp32 | 8388608 | 0.0817 | 0.1777 | 2.18x |  |
| CUTENSOR_OP_SOFT_PLUS | unary | bf16 | 8388608 | 0.0433 | 0.1037 | 2.40x | 1.41x |
|  |  | fp16 | 8388608 | 0.0407 | 0.0993 | 2.44x |  |
|  |  | fp32 | 8388608 | 0.0609 | 0.1224 | 2.01x |  |
| CUTENSOR_OP_ASINH | unary | bf16 | 8388608 | 0.0589 | 0.1078 | 1.83x | 1.36x |
|  |  | fp16 | 8388608 | 0.0512 | 0.1032 | 2.02x |  |
|  |  | fp32 | 8388608 | 0.0620 | 0.1262 | 2.04x |  |
| CUTENSOR_OP_SIN | unary | bf16 | 8388608 | 0.0509 | 0.0737 | 1.45x | 1.33x |
|  |  | fp16 | 8388608 | 0.0466 | 0.0696 | 1.49x |  |
|  |  | fp32 | 8388608 | 0.0614 | 0.1119 | 1.82x |  |
| CUTENSOR_OP_SINH | unary | bf16 | 8388608 | 0.0415 | 0.0832 | 2.01x | 1.33x |
|  |  | fp16 | 8388608 | 0.0389 | 0.0788 | 2.03x |  |
|  |  | fp32 | 8388608 | 0.0604 | 0.1106 | 1.83x |  |
| CUTENSOR_OP_COS | unary | bf16 | 8388608 | 0.0530 | 0.0750 | 1.42x | 1.32x |
|  |  | fp16 | 8388608 | 0.0481 | 0.0704 | 1.46x |  |
|  |  | fp32 | 8388608 | 0.0620 | 0.1124 | 1.81x |  |
| CUTENSOR_OP_TAN | unary | bf16 | 8388608 | 0.0527 | 0.0771 | 1.46x | 1.32x |
|  |  | fp16 | 8388608 | 0.0486 | 0.0727 | 1.49x |  |
|  |  | fp32 | 8388608 | 0.0663 | 0.1116 | 1.68x |  |
| CUTENSOR_OP_RELU | unary | bf16 | 8388608 | 0.0364 | 0.0637 | 1.75x | 1.32x |
|  |  | fp16 | 8388608 | 0.0358 | 0.0622 | 1.74x |  |
|  |  | fp32 | 8388608 | 0.0599 | 0.1088 | 1.82x |  |
| CUTENSOR_OP_LOG | unary | bf16 | 8388608 | 0.0376 | 0.0724 | 1.93x | 1.31x |
|  |  | fp16 | 8388608 | 0.0381 | 0.0686 | 1.80x |  |
|  |  | fp32 | 8388608 | 0.0599 | 0.1085 | 1.81x |  |
| CUTENSOR_OP_COSH | unary | bf16 | 8388608 | 0.0410 | 0.0753 | 1.84x | 1.26x |
|  |  | fp16 | 8388608 | 0.0387 | 0.0717 | 1.85x |  |
|  |  | fp32 | 8388608 | 0.0602 | 0.1083 | 1.80x |  |
| CUTENSOR_OP_ATANH | unary | bf16 | 8388608 | 0.0486 | 0.0814 | 1.67x | 1.25x |
|  |  | fp16 | 8388608 | 0.0458 | 0.0783 | 1.71x |  |
|  |  | fp32 | 8388608 | 0.0602 | 0.1108 | 1.84x |  |
| CUTENSOR_OP_SIGMOID | unary | bf16 | 8388608 | 0.0466 | 0.0776 | 1.66x | 1.25x |
|  |  | fp16 | 8388608 | 0.0433 | 0.0732 | 1.69x |  |
|  |  | fp32 | 8388608 | 0.0599 | 0.1075 | 1.79x |  |
| CUTENSOR_OP_ACOSH | unary | bf16 | 8388608 | 0.0525 | 0.0783 | 1.49x | 1.25x |
|  |  | fp16 | 8388608 | 0.0492 | 0.0737 | 1.50x |  |
|  |  | fp32 | 8388608 | 0.0622 | 0.1114 | 1.79x |  |
| CUTENSOR_OP_EXP | unary | bf16 | 8388608 | 0.0387 | 0.0650 | 1.68x | 1.24x |
|  |  | fp16 | 8388608 | 0.0376 | 0.0630 | 1.67x |  |
|  |  | fp32 | 8388608 | 0.0599 | 0.1073 | 1.79x |  |
| CUTENSOR_OP_RCP | unary | bf16 | 8388608 | 0.0417 | 0.0660 | 1.58x | 1.24x |
|  |  | fp16 | 8388608 | 0.0364 | 0.0609 | 1.68x |  |
|  |  | fp32 | 8388608 | 0.0589 | 0.1062 | 1.80x |  |
| CUTENSOR_OP_NEG | unary | bf16 | 8388608 | 0.0374 | 0.0622 | 1.66x | 1.24x |
|  |  | fp16 | 8388608 | 0.0366 | 0.0609 | 1.66x |  |
|  |  | fp32 | 8388608 | 0.0596 | 0.1073 | 1.80x |  |
| CUTENSOR_OP_ABS | unary | bf16 | 8388608 | 0.0371 | 0.0622 | 1.68x | 1.23x |
|  |  | fp16 | 8388608 | 0.0376 | 0.0614 | 1.63x |  |
|  |  | fp32 | 8388608 | 0.0602 | 0.1075 | 1.79x |  |
| CUTENSOR_OP_SQRT | unary | bf16 | 8388608 | 0.0417 | 0.0709 | 1.70x | 1.23x |
|  |  | fp16 | 8388608 | 0.0399 | 0.0660 | 1.65x |  |
|  |  | fp32 | 8388608 | 0.0599 | 0.1065 | 1.78x |  |
| CUTENSOR_OP_TANH | unary | bf16 | 8388608 | 0.0492 | 0.0749 | 1.52x | 1.22x |
|  |  | fp16 | 8388608 | 0.0445 | 0.0707 | 1.59x |  |
|  |  | fp32 | 8388608 | 0.0617 | 0.1080 | 1.75x |  |
| CUTENSOR_OP_CEIL | unary | bf16 | 8388608 | 0.0387 | 0.0627 | 1.62x | 1.22x |
|  |  | fp16 | 8388608 | 0.0376 | 0.0617 | 1.64x |  |
|  |  | fp32 | 8388608 | 0.0596 | 0.1075 | 1.80x |  |
| CUTENSOR_OP_FLOOR | unary | bf16 | 8388608 | 0.0381 | 0.0620 | 1.62x | 1.21x |
|  |  | fp16 | 8388608 | 0.0379 | 0.0609 | 1.61x |  |
|  |  | fp32 | 8388608 | 0.0602 | 0.1073 | 1.78x |  |
| CUTENSOR_OP_ACOS | unary | bf16 | 8388608 | 0.0489 | 0.0735 | 1.50x | 1.21x |
|  |  | fp16 | 8388608 | 0.0451 | 0.0681 | 1.51x |  |
|  |  | fp32 | 8388608 | 0.0604 | 0.1070 | 1.77x |  |
| CUTENSOR_OP_ATAN | unary | bf16 | 8388608 | 0.0476 | 0.0724 | 1.52x | 1.20x |
|  |  | fp16 | 8388608 | 0.0440 | 0.0681 | 1.55x |  |
|  |  | fp32 | 8388608 | 0.0609 | 0.1078 | 1.77x |  |
| CUTENSOR_OP_SWISH | unary | bf16 | 8388608 | 0.0476 | 0.0650 | 1.37x | 1.19x |
|  |  | fp16 | 8388608 | 0.0448 | 0.0630 | 1.41x |  |
|  |  | fp32 | 8388608 | 0.0602 | 0.1070 | 1.78x |  |
| CUTENSOR_OP_ASIN | unary | bf16 | 8388608 | 0.0504 | 0.0740 | 1.47x | 1.19x |
|  |  | fp16 | 8388608 | 0.0466 | 0.0701 | 1.51x |  |
|  |  | fp32 | 8388608 | 0.0614 | 0.1078 | 1.75x |  |
| CUTENSOR_OP_CONJ | unary | cf64 | 8388608 | 0.1551 | 0.1974 | 1.27x | 1.11x |
|  |  | torch.complex128 | 8388608 | 0.5880 | 0.3784 | 0.64x |  |
| CUTENSOR_OP_IDENTITY | unary | bf16 | 8388608 | 0.0364 | 0.0369 | 1.01x | 1.01x |
|  |  | fp16 | 8388608 | 0.0361 | 0.0364 | 1.01x |  |
|  |  | fp32 | 8388608 | 0.0591 | 0.0591 | 1.00x |  |
| Contraction | contraction | fp32 | 32,256,128 | 0.0821 | 0.1170 | 1.43x | 0.97x |

## 4. Top 5 / Bottom 5（按算子平均加速比）

### Top 5

| 算子 | 类别 | 平均加速比 | 最大加速比 | 数据点数 |
|---|---|---|---|---|
| ElementwiseTrinary | contraction | 2.77x | 4.06x | 15 |
| BlockSparseContraction | sparse | 2.13x | 3.23x | 8 |
| CUTENSOR_OP_MISH | unary | 1.77x | 3.80x | 66 |
| CUTENSOR_OP_SOFT_SIGN | unary | 1.70x | 3.76x | 66 |
| ContractionTrinary | contraction | 1.63x | 1.61x | 9 |

### Bottom 5

| 算子 | 类别 | 平均加速比 | 最小加速比 | 数据点数 |
|---|---|---|---|---|
| Contraction | contraction | 0.97x | 1.43x | 9 |
| CUTENSOR_OP_IDENTITY | unary | 1.01x | 1.00x | 66 |
| CUTENSOR_OP_CONJ | unary | 1.11x | 0.64x | 44 |
| CUTENSOR_OP_ASIN | unary | 1.19x | 1.47x | 66 |
| CUTENSOR_OP_SWISH | unary | 1.19x | 1.37x | 66 |

## 5. 测试环境与配置

### 5.1 硬件

- **GPU**: MetaX C550 × 8 (每卡 64GB HBM, MACA 3.7.1 SDK)
- **驱动**: mxdriver (MACA KMD 3.8.1)
- **SMI**: mx-smi 2.3.1

### 5.2 软件栈

| 组件 | 版本 | 说明 |
|---|---|---|
| PyTorch | 2.10.0+cpu | stock torch + torch_fl 0.1.0+metax3.8.1 插件 (torch.cuda 走 MACA) |
| Triton | 3.6.0+metax3.8.1 | MetaX 原生 fork (metax backend on Triton 3.6 base) |
| FlagTensor | 0.1.0 | 本仓库 (editable install) |
| Python | 3.10.12 | |
| MACA SDK | 3.7.1 (/opt/maca) | `MACA_PATH` 环境变量 |

### 5.3 Baseline 选择

MACA SDK 不提供 `libcutensor`，无 cuTensor 等价库。MetaX 的 baseline = **PyTorch 原生算子**（经 MACA 派发到 mcblas/mcdnn/mcfft 厂商库），与 cuTensor 在 NVIDIA 上的角色对应。实现位于 `src/flagtensor/runtime/backend/_metax/baseline.py`。

### 5.4 容差配置

- **Accuracy 测试**: 严格 per-dtype 容差（fp32: 1.3e-6，非 NVIDIA 后端自动放宽到 1e-4 以吸收 aclnn/aten 路径差异）
- **Performance benchmark verify**: 厂商 `tolerances.yaml` floor——MetaX 默认 1e-3，contraction 系（Contraction/ContractionTrinary/BlockSparseContraction）5e-3（吸收 mcblas 与 Triton kernel 的 GEMM 求和顺序差异）

## 6. 算子清单（36 个）

| # | 算子 | 类别 | Accuracy | Performance |
|---|---|---|---|---|
| 1 | BlockSparseContraction | sparse | Passed | Passed |
| 2 | CUTENSOR_OP_ABS | unary | Passed | Passed |
| 3 | CUTENSOR_OP_ACOS | unary | Passed | Passed |
| 4 | CUTENSOR_OP_ACOSH | unary | Passed | Passed |
| 5 | CUTENSOR_OP_ADD | binary | Passed | Passed |
| 6 | CUTENSOR_OP_ASIN | unary | Passed | Passed |
| 7 | CUTENSOR_OP_ASINH | unary | Passed | Passed |
| 8 | CUTENSOR_OP_ATAN | unary | Passed | Passed |
| 9 | CUTENSOR_OP_ATANH | unary | Passed | Passed |
| 10 | CUTENSOR_OP_CEIL | unary | Passed | Passed |
| 11 | CUTENSOR_OP_CONJ | unary | Passed | Passed |
| 12 | CUTENSOR_OP_COS | unary | Passed | Passed |
| 13 | CUTENSOR_OP_COSH | unary | Passed | Passed |
| 14 | CUTENSOR_OP_EXP | unary | Passed | Passed |
| 15 | CUTENSOR_OP_FLOOR | unary | Passed | Passed |
| 16 | CUTENSOR_OP_IDENTITY | unary | Passed | Passed |
| 17 | CUTENSOR_OP_LOG | unary | Passed | Passed |
| 18 | CUTENSOR_OP_MAX | binary | Passed | Passed |
| 19 | CUTENSOR_OP_MIN | binary | Passed | Passed |
| 20 | CUTENSOR_OP_MISH | unary | Passed | Passed |
| 21 | CUTENSOR_OP_MUL | binary | Passed | Passed |
| 22 | CUTENSOR_OP_NEG | unary | Passed | Passed |
| 23 | CUTENSOR_OP_RCP | unary | Passed | Passed |
| 24 | CUTENSOR_OP_RELU | unary | Passed | Passed |
| 25 | CUTENSOR_OP_SIGMOID | unary | Passed | Passed |
| 26 | CUTENSOR_OP_SIN | unary | Passed | Passed |
| 27 | CUTENSOR_OP_SINH | unary | Passed | Passed |
| 28 | CUTENSOR_OP_SOFT_PLUS | unary | Passed | Passed |
| 29 | CUTENSOR_OP_SOFT_SIGN | unary | Passed | Passed |
| 30 | CUTENSOR_OP_SQRT | unary | Passed | Passed |
| 31 | CUTENSOR_OP_SWISH | unary | Passed | Passed |
| 32 | CUTENSOR_OP_TAN | unary | Passed | Passed |
| 33 | CUTENSOR_OP_TANH | unary | Passed | Passed |
| 34 | Contraction | contraction | Passed | Passed |
| 35 | ContractionTrinary | contraction | Passed | Passed |
| 36 | ElementwiseTrinary | contraction | Passed | Passed |

## 7. 复现方法

```bash
# 环境变量
export MACA_PATH=/opt/maca
export PATH=/tmp/opencode/venv-fl/bin:$PATH   # 或你的 venv
export PYTHONPATH=src

# 全集（36 算子，8 卡并行）
OPS=$(python3 -c "import yaml; print(','.join(o['name'] for o in yaml.safe_load(open('conf/operators.yaml'))['ops']))")
python3 tools/run_tests.py --ops "$OPS" --gpus 0,1,2,3,4,5,6,7 --output-dir results

# 正式（pilot 8 算子，单卡）
python3 tools/run_tests.py --gpus 0 --output-dir results

# 单算子调试
python3 tools/run_tests.py --ops CUTENSOR_OP_SQRT --gpus 0 --output-dir results --dump-output
```

## 8. 备注

- **FlagTree 说明**: 当前环境跑的是 MetaX 原生 `triton-3.6.0+metax3.8.1`（API 与 FlagTree 一致，metax backend 同源）。FlagTree 官方 metax wheel 目前基于 Triton 3.1（`flagtree==0.5.1+metax3.1`），FlagTensor 的 kernel 需要 Triton ≥3.6，在 3.1 下会报 MLIR 编译错误。详见 `setup_muxi.sh` 注释。
- **Contraction 平均加速比 <1**: GETT contraction 在小 shape 上 Triton kernel 的 launch overhead 占比高，baseline（mcblas）在小矩阵上更快；大 shape 上 Triton 反超（见 Contraction 各 shape 明细）。
- **数据完整性**: 36/36 accuracy + 36/36 performance 全通过，无 skip/error/timeout。原始 JSON 在 `summary.json`（433KB，含每个 shape 的 triton_ms/baseline_ms/speedup）。

---
*报告生成自 `/tmp/metax_full2/summary.json`，由 `tools/run_tests.py` 在 MetaX C550 上实测产出。*