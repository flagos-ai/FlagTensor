# FlagTensor Benchmark Strategy

FlagTensor performance testing is pytest-based and uses cuTensor baselines to measure Triton operator performance.

## Benchmark goals

- Compare Triton implementations against cuTensor-backed baselines.
- Preserve per-shape and per-dtype detail for diagnosis.
- Provide stable summary artifacts for CI, weekly runs, and acceptance reporting.

## Benchmark modes

FlagTensor currently supports multiple benchmark modes:

- `kernel`
  - Measures the kernel-facing callable when available.
  - Used for low-level performance diagnosis.
- `operator`
  - Measures the public operator path.
  - Used for end-to-end operator overhead analysis.
- `wrapper`
  - Reserved for wrapper-path timing where the benchmark supplies a dedicated wrapper callable.

Acceptance reports should clearly label which mode is being summarized.

## Smoke vs full runs

- Smoke mode is a reduced benchmark configuration for CI turnaround.
- Full mode uses the benchmark's default dtypes and full configured shapes.
- Smoke mode currently reduces warmup, repetitions, and shape count, and may restrict dtypes if configured.

## Metrics

Default metrics:

- `latency_base`
- `latency`
- `speedup`

Additional operator-specific metrics may be added where meaningful.

## Timing requirements

- Include warmup before measurement.
- Use repeated measurements and aggregate with stable summary statistics.
- Exclude input construction time from the timed region.
- Synchronize device execution where needed to avoid async timing artifacts.

## Output artifacts

Each benchmark should preserve:

- raw console log
- per-case CSV output
- mode-specific CSV when multiple modes are emitted
- summary JSON/Markdown/XLSX from runner scripts when applicable

## Registry alignment

- The operator registry is the default source of truth for what benchmark exists.
- `benchmark_mark` and `benchmark_modes` in `conf/operators.yaml` define how CI and weekly tools discover executable benchmark coverage.
- Runner scripts should prefer mode-aware CSV selection such as `benchmark_kernel.csv` or `benchmark_operator.csv` over a hard-coded single-mode assumption.
