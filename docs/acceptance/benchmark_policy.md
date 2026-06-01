# FlagTensor Benchmark Policy

## Scope

This document defines the acceptance-facing benchmark policy for FlagTensor performance validation.

## Benchmark Goals

- Compare FlagTensor kernels against cuTensor baselines.
- Produce reproducible operator-level and category-level benchmark artifacts.
- Support smoke and acceptance-level execution modes.
- Report benchmark results with mode-aware output handling.

## Benchmark Modes

| Mode | Meaning | Typical Use |
| --- | --- | --- |
| `kernel` | Kernel-focused measurement | Low-level performance analysis |
| `operator` | Operator-level measurement | Default acceptance reporting |
| `wrapper` | Wrapper/API-path measurement | Integration-level validation |

## Execution Levels

### Smoke Benchmark

- Reduced shape set
- Reduced dtype set
- Intended for CI turnaround speed
- Triggered via `tools/run_flagtensor_ci.py --smoke --run-perf`

### Acceptance Benchmark

- Full configured shape coverage
- Full supported dtype coverage for the selected operator set
- Intended for release and acceptance review
- Triggered via `tools/run_flagtensor_ci.py --run-perf`

### Weekly Benchmark

- Registry-driven scheduled or manual regression execution
- Intended for broader drift tracking across operators and GPUs
- Triggered via `tools/run_flagtensor_weekly.py`

## Shape and Dtype Policy

- Benchmark shapes should be centrally managed where possible.
- Category-level benchmark entry points are the formal acceptance interface.
  Legacy per-operator benchmark files are retained as debugging and migration compatibility shims,
  not as an acceptance requirement.
- Benchmark dtypes default to `float16` and `float32` unless the operator requires a specialized dtype set.

## Timing Policy

- Warmup count and repetition count must be explicit and reproducible.
- Current defaults are controlled through environment variables and runner flags.
- Future consolidation should move shared timing policy into a centralized benchmark utility layer.

## Reporting Policy

- Benchmark CSV selection must be mode-aware.
- Reports should distinguish `kernel`, `operator`, and `wrapper` outputs.
- Acceptance reporting should include pass/fail status and speedup statistics.
- HTML and XLSX reporting are supported.

## Category Benchmark Entry Points (Acceptance Interface)

Benchmark execution uses category-level files as the formal acceptance interface.
Individual operators are selected via `pytest -m <op>` markers.

Current category entry points (all four complete):

- `benchmark/test_unary_perf.py` — 28 unary operators
- `benchmark/test_binary_perf.py` — 4 binary operators
- `benchmark/test_contraction_perf.py` — 5 contraction operators
- `benchmark/test_sparse_perf.py` — 1 sparse operator

Legacy per-operator benchmark files (`benchmark/test_CUTENSOR_OP_*_perf.py`) are retained as
implementation details for debugging and migration compatibility, but they are not part of the
formal acceptance interface.

## Source of Truth

- Registry metadata: `conf/operators.yaml`
- Benchmark strategy overview: `docs/benchmark_strategy.md`
- CI runner: `tools/run_flagtensor_ci.py`
- Weekly runner: `tools/run_flagtensor_weekly.py`
