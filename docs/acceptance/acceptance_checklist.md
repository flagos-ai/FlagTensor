# FlagTensor Acceptance Checklist

This checklist tracks the current compliance status against the operator library acceptance standards.

## Structure & Organization

| Item | Status | Notes |
| --- | --- | --- |
| Unified operator registry exists | **Done** | `conf/operators.yaml` |
| Registry is CI/weekly/entry point | **Done** | `tools/run_flagtensor_ci.py` and `run_flagtensor_weekly.py` use it |
| `tests/` directory exists as correctness entry | **Mostly Done** | All categories migrated: unary (26/27, exp proxied due to float64 issue), binary (4/4 complete), contraction (3/5, trinary_generic and tensor_contraction_trinary proxied), sparse (1/1, float32/float64 only) |
| `benchmark/` supports category-level execution | **Partial** | `test_unary_perf.py`, `test_binary_perf.py`, `test_contraction_perf.py` exist; `test_sparse_perf.py` and shared perf utilities are still missing |
| Pre-commit configuration exists | **Done** | `.pre-commit-config.yaml` |
| pyproject.toml has tool configs | **Done** | black, isort, flake8, pytest markers |

## Testing Framework

| Item | Status | Notes |
| --- | --- | --- |
| Pytest-based correctness tests | **Done** | `ctests/` and `tests/` |
| Shared tolerance/assertion helpers | **Done** | centralized in `src/flagtensor/testing/` package with `assertions.py`, `shapes.py`, `dtypes.py` modules |
| `tests/accuracy_utils.py` compatibility layer | **Done** | Re-exports from `flagtensor.testing` |
| Dtype-aware tolerance policy | **Done** | float16, float32, float64, bfloat16, complex |
| Reference selection documented | **Done** | `docs/testing_strategy.md` |
| Shape coverage policy documented | **Done** | `docs/testing_strategy.md` |

## Performance Testing

| Item | Status | Notes |
| --- | --- | --- |
| Benchmark against cuTensor baselines | **Done** | Existing benchmark suite |
| Kernel/operator/wrapper modes defined | **Done** | `docs/benchmark_strategy.md` |
| Smoke vs full run distinction | **Done** | `run_flagtensor_ci.py --smoke` |
| Warmup/repetition/timing standards | **Partial** | Environment variables exist; consolidation needed |
| Mode-aware CSV selection | **Done** | `benchmark_csv_path()` in CI runner |
| HTML/XLSX report generation | **Partial** | HTML report tooling exists; XLSX acceptance output is not yet standardized |

## CI/CD & Automation

| Item | Status | Notes |
| --- | --- | --- |
| Correctness CI job | **Done** | `ci.yaml` correctness-smoke |
| Performance CI job | **Done** | `ci.yaml` perf-smoke |
| Weekly regression workflow | **Done** | `weekly.yaml` (registry-driven) |
| Quality gate (pre-commit) | **Done** | `quality-gate.yaml` |
| Registry consistency check | **Done** | `quality-gate.yaml` registry-consistency job |
| Build/package check | **Done** | `quality-gate.yaml` build-check job |
| Artifact upload and summary | **Done** | Artifacts + GITHUB_STEP_SUMMARY in CI |
| Acceptance-level CI workflow | **Done** | `acceptance.yaml` with category/mode filtering |
| CI matrix documentation | **Done** | `docs/acceptance/ci_matrix.md` |

## Documentation & Release

| Item | Status | Notes |
| --- | --- | --- |
| README with usage examples | **Done** | `README.md` |
| Testing strategy document | **Done** | `docs/testing_strategy.md` |
| Benchmark strategy document | **Done** | `docs/benchmark_strategy.md` |
| Acceptance checklist | **Done** | This file |
| Operator coverage matrix | **Done** | `docs/acceptance/operator_coverage.md` |
| FlagTensor-specific accuracy/benchmark policies exist | **Done** | `docs/acceptance/accuracy_policy.md`, `docs/acceptance/benchmark_policy.md`; 已回填验收规范 TODO |
| Known issues list | **Done** | `docs/acceptance/known_issues.md` |
| Standard acceptance commands | **Done** | `docs/acceptance/standard_commands.md` |
| Release note template | **Done** | `docs/acceptance/release_notes_template.md` |

## Known Issues

| Operator | Issue | Registry Status |
| --- | --- | --- |
| `tensor_contraction_trinary` | float64 correctness path not stable in CI | `blocked` |
