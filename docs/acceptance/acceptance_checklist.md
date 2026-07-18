<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# FlagTensor Acceptance Checklist

This checklist tracks the current compliance status against the operator library acceptance standards.

## Structure & Organization

| Item | Status | Notes |
| --- | --- | --- |
| Unified operator registry exists | **Done** | `conf/operators.yaml` |
| Registry is CI/weekly/entry point | **Done** | `tools/run_flagtensor_ci.py` and `run_flagtensor_weekly.py` use it |
| `tests/` directory exists as correctness entry | **Done** | Category-based organization: `tests/{unary,binary,contraction,sparse}/`. All categories migrated (28 unary, 4 binary, 5 contraction, 1 sparse). Legacy per-op `ctests/` files retained as compatibility shims. |
| Benchmark dtype coverage | **Done** | float16, float32 per `DEFAULT_BENCHMARK_DTYPES` |
| Correctness dtype coverage | **Done** | float16, float32, bfloat16 per `DEFAULT_CORRECTNESS_DTYPES` |
| Benchmark shape coverage | **Done** | unary/binary: 22 shapes (14 1D pow2 + 8 multi-dimensional); contraction: 4 shape pairs; sparse: 3 shape pairs |
| `benchmark/` supports category-level execution | **Done** | All four categories: `test_unary_perf.py`, `test_binary_perf.py`, `test_contraction_perf.py`, `test_sparse_perf.py` |
| Pre-commit configuration exists | **Done** | `.pre-commit-config.yaml` |
| pyproject.toml has tool configs | **Done** | black, isort, flake8, pytest markers |

## Testing Framework

| Item | Status | Notes |
| --- | --- | --- |
| Pytest-based correctness tests | **Done** | `ctests/` and `tests/` |
| Shared tolerance/assertion helpers | **Done** | centralized in `src/flagtensor/testing/` package with `assertions.py`, `shapes.py`, `dtypes.py` modules |
| `tests/accuracy_utils.py` compatibility layer | **Done** | Re-exports from `flagtensor.testing` |
| Dtype-aware tolerance policy | **Done** | float16, float32, bfloat16 |
| Reference selection documented | **Done** | `docs/testing_strategy.md` |
| Shape coverage policy documented | **Done** | `docs/testing_strategy.md` |

## Performance Testing

| Item | Status | Notes |
| --- | --- | --- |
| Benchmark against cuTensor baselines | **Done** | Existing benchmark suite |
| Kernel/operator/wrapper modes defined | **Done** | `docs/benchmark_strategy.md` |
| Smoke vs full run distinction | **Done** | `run_flagtensor_ci.py --smoke` |
| Warmup/repetition/timing standards | **Done** | Consolidated in `benchmark_core.py` and `config.py` |
| Mode-aware CSV selection | **Done** | `benchmark_csv_path()` in CI runner |
| HTML/XLSX report generation | **Done** | HTML report tooling exists; XLSX output via `write_benchmark_xlsx()` in `visualization.py` |

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
| Release/publish workflow | **Done** | `release.yaml` with build, release notes, PyPI publish |
| Multi-backend compatibility CI | **Done** | `compatibility.yaml` with CUDA 12.1/12.4/12.6 matrix + H20 regression |

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

All previously documented issues have been resolved:

- `exp` and `log`: float64 fallback removed — float64 is no longer a supported dtype.
- `tensor_contraction_trinary`: float64 path removed; operator supports float16/float32 only.
- `block_sparse_tensor_contraction` float16: Fixed via dense fallback routing; float16 tests now active.
