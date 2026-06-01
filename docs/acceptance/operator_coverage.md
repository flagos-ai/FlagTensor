# FlagTensor Operator Coverage Matrix

Generated from registry: `conf/operators.yaml`

## By Category

### Unary Operators (28)

| Operator | Impl | Correctness | Benchmark | Modes | Status |
| --- | --- | --- | --- | --- | --- |
| abs | Done | Done | Done | operator | stable |
| acos | Done | Done | Done | kernel, operator, wrapper | stable |
| acosh | Done | Done | Done | operator | stable |
| asin | Done | Done | Done | operator | stable |
| asinh | Done | Done | Done | operator | stable |
| atan | Done | Done | Done | operator | stable |
| atanh | Done | Done | Done | operator | stable |
| ceil | Done | Done | Done | operator | stable |
| conj | Done | Done | Done | operator | stable |
| cos | Done | Done | Done | operator | stable |
| cosh | Done | Done | Done | operator | stable |
| exp | Done | Done | Done | operator | stable |
| floor | Done | Done | Done | operator | stable |
| identity | Done | Done | Done | operator | stable |
| log | Done | Done | Done | operator | stable |
| mish | Done | Done | Done | operator | stable |
| neg | Done | Done | Done | operator | stable |
| rcp | Done | Done | Done | operator | stable |
| relu | Done | Done | Done | operator | stable |
| sigmoid | Done | Done | Done | operator | stable |
| sin | Done | Done | Done | operator | stable |
| sinh | Done | Done | Done | operator | stable |
| soft_plus | Done | Done | Done | operator | stable |
| soft_sign | Done | Done | Done | operator | stable |
| sqrt | Done | Done | Done | operator | stable |
| swish | Done | Done | Done | operator | stable |
| tan | Done | Done | Done | operator | stable |
| tanh | Done | Done | Done | operator | stable |

### Binary Operators (4)

| Operator | Impl | Correctness | Benchmark | Modes | Status |
| --- | --- | --- | --- | --- | --- |
| add | Done | Done | Done | operator | stable |
| mul | Done | Done | Done | operator | stable |
| max | Done | Done | Done | operator | stable |
| min | Done | Done | Done | operator | stable |

### Contraction Operators (5)

| Operator | Impl | Correctness | Benchmark | Modes | Status |
| --- | --- | --- | --- | --- | --- |
| gett | Done | Done | Done | kernel, operator | stable |
| tgett | Done | Done | Done | kernel, operator | stable |
| ttgt | Done | Done | Done | kernel, operator | stable |
| tensor_contraction_trinary | Done | Done | Done | kernel, operator | stable |
| trinary_generic | Done | Done | Done | operator | stable |

### Sparse Operators (1)

| Operator | Impl | Correctness | Benchmark | Modes | Status |
| --- | --- | --- | --- | --- | --- |
| block_sparse_tensor_contraction | Done | Done | Done | operator | experimental |

## Summary

- **Total Operators**: 38
- **Stable**: 37
- **Experimental**: 1
- **Blocked**: 0
- **Categories**: unary (28), binary (4), contraction (5), sparse (1)
