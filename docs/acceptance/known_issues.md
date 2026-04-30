# FlagTensor Known Issues

This document tracks known issues and limitations in the current FlagTensor implementation.

## Blocked Operators

### tensor_contraction_trinary

- **Status**: Blocked in registry
- **Issue**: float64 correctness path not stable in CI environment
- **Impact**: Operator excluded from default CI/weekly runs
- **Registry Skip Reason**: float64 correctness path not stable in CI
- **Workaround**: Can be manually included with `--include-blocked` flag for debugging

## Experimental Operators

### block_sparse_tensor_contraction

- **Status**: Experimental
- **Issue**: Sparse tensor contraction support is still under active development
- **Impact**: May have limited shape/dtype coverage compared to dense operators
- **Recommendation**: Use for evaluation only; not for production workloads

## Known Limitations

### Operator-Specific Numerical Issues

### exp (unary)

- **Issue**: float64 correctness fails in current tolerance configuration
- **Symptom**: `torch.allclose` fails when comparing FlagTensor exp output against torch.exp reference
- **Affected Shapes**: All shapes in DEFAULT_EXP_TEST_SHAPES for float64 dtype
- **Impact**: exp float64 tests fail both in legacy ctests/ and in migrated tests/unary/
- **Status**: Under investigation - may require tolerance adjustment or kernel alignment
- **Workaround**: Keep exp correctness tests proxied via ctests/ while investigating

### block_sparse_tensor_contraction (sparse)

- **Issue**: float16 correctness has precision issues
- **Symptom**: `torch.allclose` fails for float16 dtype due to numerical precision
- **Affected Shapes**: All test shapes for float16 dtype
- **Impact**: float16 tests skipped in migrated tests/sparse/, float32/float64 tests pass
- **Status**: Migrated with dtype restriction (float32/float64 only)
- **Workaround**: Use float32 or float64 for sparse tensor contraction tests

### CI Environment

- **GPU Access**: CI workflows run on ubuntu-latest (CPU) without GPU access
  - Actual GPU validation must be done via Slurm on cluster nodes
  - CI correctness/perf jobs currently validate structure and integration, not actual GPU correctness
- **Memory**: CI runners have limited memory; large shape tests are reduced in smoke mode

### Benchmark Mode Coverage

- **kernel mode**: Fully supported for most operators
- **operator mode**: Supported for subset of operators
- **wrapper mode**: Limited support; mainly for operators where wrapper-level optimization is beneficial

### Dtype Coverage

- **float16**: Fully supported across operators
- **float32**: Fully supported across operators
- **float64**: Supported in correctness tests; some operators blocked in CI due to numerical stability
- **bfloat16**: Limited support; primarily for evaluation
- **complex64/complex128**: Supported for complex-specific operators (e.g., conj)

### Shape Coverage

- **Small shapes**: (1024,), (4096,) - covered in correctness and smoke benchmark
- **Medium shapes**: (128, 128), (32, 64, 16) - covered in correctness tests
- **Large shapes**: Up to 2^24 elements - covered in full benchmark runs
- **Contraction shapes**: Specialized shapes for layout/chain validation

## Performance Notes

- **Triton Autotuner**: Current Triton version uses deprecated warmup/rep parameters
  - Deprecation warnings appear in benchmark output
  - Does not affect functionality; will be addressed in future Triton upgrade
- **cuTensor Baseline**: Performance comparisons against cuTensor C API
  - Some operators may show speedup < 1x for certain shapes/dtypes
  - This is expected behavior and not necessarily an issue

## Migration Notes

### Directory Structure Transition

- **ctests/**: Legacy correctness test directory; being migrated to tests/
- **benchmark/**: Single-operator perf files; category-level entry points added
- **tests/**: New unified correctness entry with proxy layer for legacy tests
- **src/flagtensor/testing.py**: Centralized tolerance/assertion helpers (not yet a package directory)

### Registry Transition

- **weekly_op_test.txt**: Legacy operator list; being replaced by conf/operators.yaml
- **discover_ops()**: Legacy discovery function; being replaced by registry-based filtering
- **Manual exclusion**: `--exclude-op` flags still supported but registry is preferred

## Future Work

- [x] Migrate all correctness tests from ctests/ to tests/ with category organization
  - [x] Category directories created (unary/, binary/, contraction/, sparse/)
  - [x] Loader supports skipping migrated operators
  - [x] Unary operators: 26 migrated (all except exp which has float64 precision issues)
  - [x] Binary operators: 4 migrated (add, mul, max, min - all complete)
  - [x] Contraction operators: 3 migrated (gett, tgett, ttgt)
  - [x] Sparse operators: 1 migrated (block_sparse_tensor_contraction, float32/float64 only)
- [ ] Add more unary/binary benchmark category files
  - [x] test_unary_perf.py
  - [x] test_binary_perf.py
  - [ ] test_sparse_perf.py
- [ ] Upgrade Triton to remove deprecation warnings
- [ ] Add GPU runner to CI for actual correctness validation
- [ ] Expand bfloat16 dtype coverage
- [ ] Improve wrapper mode coverage
- [ ] Add acceptance-level performance regression detection
- [ ] Resolve exp float64 numerical stability issue
