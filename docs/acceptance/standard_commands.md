# FlagTensor Standard Acceptance Commands

This document provides the standard commands for running FlagTensor acceptance checks.

## Prerequisites

```bash
# Install the package
python -m pip install .

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

## Static Quality Checks

```bash
# Run pre-commit checks locally
pre-commit run --all-files --show-diff-on-failure

# Build package for verification
python -m build

# Check package metadata
twine check dist/*
```

## Correctness Testing

### Smoke Correctness (CI-level)

```bash
# Run smoke correctness for all operators
python tools/run_flagtensor_ci.py --smoke --run-correctness --results-dir ci_results_correctness --dump-json-summary

# Run smoke correctness for specific operator
python tools/run_flagtensor_ci.py --op acos --smoke --run-correctness --results-dir ci_results_correctness --dump-json-summary

# Run smoke correctness for specific category
python tools/run_flagtensor_ci.py --category unary --smoke --run-correctness --results-dir ci_results_correctness --dump-json-summary
```

### Acceptance Correctness (Full Coverage)

```bash
# Run acceptance-level correctness for all operators
python tools/run_flagtensor_ci.py --run-correctness --results-dir acceptance_results_correctness --dump-json-summary

# Run acceptance-level correctness for specific category
python tools/run_flagtensor_ci.py --category unary --run-correctness --results-dir acceptance_results_correctness --dump-json-summary

# Run acceptance-level correctness with specific benchmark mode
python tools/run_flagtensor_ci.py --run-correctness --mode operator --results-dir acceptance_results_correctness --dump-json-summary
```

### Correctness via Pytest — Category Files (Primary)

Category-level correctness files are the formal acceptance interface:

```bash
# Run all correctness tests via tests/ entry
python -m pytest -vs tests

# Run specific operator correctness via marker
python -m pytest -vs tests -m acos

# Run all operators in a category
python -m pytest -vs tests/unary/
```

### Correctness via Pytest — Legacy/Debug

Legacy per-operator files in `ctests/` are retained for debugging but are not part of the acceptance interface:

```bash
python -m pytest -vs ctests/test_CUTENSOR_OP_ACOS.py
```

## Performance Testing

### Smoke Performance (CI-level)

```bash
# Run smoke performance for all operators
python tools/run_flagtensor_ci.py --smoke --run-perf --results-dir ci_results_perf --dump-json-summary

# Run smoke performance for specific operator
python tools/run_flagtensor_ci.py --op acos --smoke --run-perf --results-dir ci_results_perf --dump-json-summary

# Run smoke performance for specific category
python tools/run_flagtensor_ci.py --category unary --smoke --run-perf --results-dir ci_results_perf --dump-json-summary
```

### Acceptance Performance (Full Coverage)

```bash
# Run acceptance-level performance for all operators
python tools/run_flagtensor_ci.py --run-perf --results-dir acceptance_results_perf --dump-json-summary

# Run acceptance-level performance with specific benchmark mode
python tools/run_flagtensor_ci.py --run-perf --mode operator --results-dir acceptance_results_perf --dump-json-summary
```

### Performance via Category Benchmarks (Primary)

Category-level benchmark files are the formal acceptance interface. Individual operators are selected via `pytest -m <op>` markers:

```bash
# Run unary category benchmark
python -m pytest -vs benchmark/test_unary_perf.py -m identity

# Run binary category benchmark
python -m pytest -vs benchmark/test_binary_perf.py -m add

# Run contraction category benchmark
python -m pytest -vs benchmark/test_contraction_perf.py -m gett

# Run sparse category benchmark
python -m pytest -vs benchmark/test_sparse_perf.py -m block_sparse_tensor_contraction
```

### Performance via Single Operator — Legacy/Debug

Legacy per-operator benchmark files are retained for debugging but are not part of the acceptance interface:

```bash
python -m pytest -vs benchmark/test_CUTENSOR_OP_ACOS_perf.py
```

## Weekly Regression

```bash
# Run weekly regression with registry-driven operator selection
python tools/run_flagtensor_weekly.py --project-root . --results-dir weekly_results --gpus 0 --mode kernel

# Run weekly with specific operator list (optional; generated from registry if omitted)
python tools/run_flagtensor_weekly.py --project-root . --op-list my_ops.txt --results-dir weekly_results --gpus 0 --mode kernel

# Run weekly with category filter
python tools/run_flagtensor_weekly.py --project-root . --category unary --results-dir weekly_results --gpus 0 --mode kernel
```

## Registry Operations

```bash
# Load and inspect registry
python - <<'PY'
import sys
sys.path.insert(0, 'src')
from flagtensor_registry import load_operator_registry

for spec in load_operator_registry():
    print(f"{spec.name}: category={spec.category}, status={spec.status}")
PY

# Check registry consistency
python - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, 'src')
from flagtensor_registry import load_operator_registry

registry = load_operator_registry()
errors = []

for spec in registry:
    impl_path = Path(spec.impl_file)
    if not impl_path.exists():
        errors.append(f"Missing impl: {spec.name} -> {spec.impl_file}")

    test_path = Path(spec.correctness_test)
    if not test_path.exists():
        errors.append(f"Missing test: {spec.name} -> {spec.correctness_test}")

    bench_path = Path(spec.benchmark_test)
    if not bench_path.exists():
        errors.append(f"Missing benchmark: {spec.name} -> {spec.benchmark_test}")

if errors:
    print("Registry consistency errors:")
    for e in errors:
        print(f"  - {e}")
else:
    print(f"Registry consistency OK: {len(registry)} operators")
PY
```

## Report Generation

```bash
# Generate HTML report from CI results
python tools/generate_flagtensor_html_report.py --results-dir ci_results_correctness --output ci_results_correctness/report.html

# Generate HTML report from acceptance results
python tools/generate_flagtensor_html_report.py --results-dir acceptance_results_correctness --output acceptance_results_correctness/report.html
```

## Environment Export

```bash
# Export environment for reproducibility
python tools/export_env.py --project-root . --output env.json
```

## GPU Cluster Validation (Slurm)

```bash
# Run correctness on GPU cluster node
srun -N 1 --job-name flagtensor-correctness --nodelist <node_name> --gres=gpu:1 --cpus-per-task=24 --mem=242144 \
  docker exec -w /workspace/FlagGems/flagtensor triton_cuda12 \
  bash -lc "python tools/run_flagtensor_ci.py --smoke --run-correctness --results-dir /tmp/flagtensor_ci_results"

# Run weekly on GPU cluster node
srun -N 1 --job-name flagtensor-weekly --nodelist <node_name> --gres=gpu:1 --cpus-per-task=24 --mem=242144 \
  docker exec -w /workspace/FlagGems/flagtensor triton_cuda12 \
  bash -lc "python tools/run_flagtensor_weekly.py --project-root /workspace/FlagGems/flagtensor --results-dir /tmp/flagtensor_weekly_results --gpus 0"
```

## Quick Acceptance Checklist

To verify acceptance readiness, run the following commands in order:

1. **Static Quality**
   ```bash
   pre-commit run --all-files --show-diff-on-failure
   ```

2. **Registry Consistency**
   ```bash
   python -c "import sys; sys.path.insert(0, 'src'); from flagtensor_registry import load_operator_registry; print(f'Registry OK: {len(list(load_operator_registry()))} operators')"
   ```

3. **Smoke Correctness**
   ```bash
   python tools/run_flagtensor_ci.py --smoke --run-correctness --results-dir /tmp/flagtensor_smoke_correctness --dump-json-summary && cat /tmp/flagtensor_smoke_correctness/summary.json
   ```

4. **Smoke Performance**
   ```bash
   python tools/run_flagtensor_ci.py --smoke --run-perf --results-dir /tmp/flagtensor_smoke_perf --dump-json-summary && cat /tmp/flagtensor_smoke_perf/summary.json
   ```

5. **Category Benchmark Validation**
   ```bash
   python -m pytest -vs benchmark/test_unary_perf.py -m identity
   python -m pytest -vs benchmark/test_binary_perf.py -m add
   python -m pytest -vs benchmark/test_contraction_perf.py -m gett
   python -m pytest -vs benchmark/test_sparse_perf.py -m block_sparse_tensor_contraction
   ```
