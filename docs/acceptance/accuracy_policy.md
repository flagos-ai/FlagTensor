# FlagTensor Accuracy Policy

## Scope

This document defines the acceptance-facing accuracy policy for FlagTensor correctness validation.

## Reference Policy

- Unary and binary operators use mathematically equivalent PyTorch expressions as the primary reference.
- cuTensor-backed contraction operators may compare against either CPU references or cuTensor references depending on operator semantics and numerical stability.
- Any operator with special numerical behavior must document its reference choice in the test implementation.

## Assertion Policy

- Bit-exact operators should use equality-style assertions.
- Floating-point operators should use close-style assertions.
- All default close-style assertions should use centralized helper APIs from `src/flagtensor/testing.py`.
- Acceptance-style tests may re-export these helpers through `tests/accuracy_utils.py`.

## Default Tolerances

| Dtype | atol | rtol |
| --- | --- | --- |
| `float16` | `1e-3` | `1e-3` |
| `bfloat16` | `2e-2` | `2e-2` |
| `float32` | `1e-5` | `1e-5` |
| `float64` | `1e-7` | `1e-7` |
| `complex64` | `1e-5` | `1e-5` |
| `complex128` | `1e-7` | `1e-7` |

## Shape Policy

- Pointwise operators must cover representative small, medium, and multi-dimensional shapes.
- Contraction operators must cover layout-sensitive and chain-sensitive shapes.
- Sparse operators must cover both dense-value correctness and structure preservation.
- Shared default shape helpers should come from centralized testing utilities rather than per-test duplication.

## Dtype Coverage Policy

- Default correctness coverage includes `float16`, `float32`, and `float64` where operator support is stable.
- `bfloat16` is opt-in and should be enabled only where kernels and references are reliable.
- Complex dtype coverage is limited to operators that explicitly support complex semantics.

## Skip / Block Policy

- Operators with known instability may be marked `blocked` in `conf/operators.yaml`.
- Blocked operators must include a `skip_reason`.
- CI and weekly workflows exclude blocked operators by default unless `--include-blocked` is explicitly requested.

## Source of Truth

- Registry metadata: `conf/operators.yaml`
- Shared helpers: `src/flagtensor/testing.py`
- Compatibility exports: `tests/accuracy_utils.py`
- Strategy overview: `docs/testing_strategy.md`
