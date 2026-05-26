# FlagTensor Known Issues

This document tracks known issues and limitations in the current FlagTensor implementation.

## Experimental Operators

### block_sparse_tensor_contraction

- **Status**: Experimental
- **Issue**: Sparse tensor contraction support is still under active development
- **Impact**: May have limited shape/dtype coverage compared to dense operators
- **Recommendation**: Use for evaluation only; not for production workloads

## Known Limitations

### Operator-Specific Numerical Issues

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
- **bfloat16**: Supported across unary and contraction operators; verified in correctness tests
- **complex64/complex128**: Supported only for `conj` operator. Triton's type system does not natively support complex dtypes; other operators reject complex inputs.

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

- **ctests/**: Legacy correctness test directory; migrated to tests/
- **benchmark/**: Single-operator perf files retained as implementation details; category-level entry points are the formal acceptance interface
- **tests/**: Unified correctness entry with proxy layer for legacy tests
- **src/flagtensor/testing/**: Centralized tolerance/assertion helpers

### Registry Transition

- **weekly_op_test.txt**: Removed; operator list is generated from registry
- **discover_ops()**: Legacy discovery function; being replaced by registry-based filtering
- **Manual exclusion**: `--exclude-op` flags still supported but registry is preferred

## Future Work

- [x] Migrate all correctness tests from ctests/ to tests/ with category organization
  - [x] Category directories created (unary/, binary/, contraction/, sparse/)
  - [x] Loader supports skipping migrated operators
  - [x] Unary operators: 27 migrated
  - [x] Binary operators: 4 migrated (add, mul, max, min - all complete)
  - [x] Contraction operators: 4 migrated (gett, tgett, ttgt, tensor_contraction_trinary)
  - [x] Sparse operators: 1 migrated (block_sparse_tensor_contraction, float16 now active)
- [x] Add category-level benchmark entry points (formal acceptance interface)
  - [x] test_unary_perf.py
  - [x] test_binary_perf.py
  - [x] test_contraction_perf.py
  - [x] test_sparse_perf.py
- [ ] Upgrade Triton to remove deprecation warnings
- [ ] Add GPU runner to CI for actual correctness validation
- [x] Expand bfloat16 dtype coverage
- [ ] Improve wrapper mode coverage
- [ ] Add acceptance-level performance regression detection
