# FlagTensor Testing Strategy

FlagTensor follows pytest-based correctness testing and keeps its testing entry aligned with the operator registry in `conf/operators.yaml`.

## Correctness principles

- Reference implementations should be chosen by operator type.
- Pure pointwise and simple tensor transforms may compare against CPU PyTorch references.
- cuTensor-backed contraction operators may compare against either CPU references or cuTensor reference implementations, depending on numerical stability and operator semantics.
- Any operator with special numerical behavior must document its reference choice explicitly in the test file.

## Assertions and tolerances

- Bit-exact operators should use equality-style assertions.
- Floating-point operators should use close-style assertions.
- Tolerance should be centralized and dtype-aware.
- Current project status:
  - correctness exists per operator in `ctests/`
  - shared helpers are centralized in `src/flagtensor/testing/`
  - compatibility exports for the acceptance-style test tree are provided in `tests/accuracy_utils.py`
- Acceptance target:
  - complete migration from per-file legacy imports toward the shared testing helper surface
  - keep project-wide default tolerances for `float16`, `float32`, and `bfloat16` under a single authority

## Shape coverage

- Every operator should cover a representative small / medium / large shape set.
- Contraction operators should additionally cover layout-sensitive and chain-sensitive shapes.
- Sparse operators should include shape/block-shape combinations that validate structure preservation.
- Shapes should gradually migrate toward centralized maintenance in config or YAML files.

## Marks and registry alignment

- Every correctness test must be selectable through `pytest -m <op_name>`.
- The authoritative operator name is the `name` field in `conf/operators.yaml`.
- `correctness_mark` in the registry must match the pytest mark used by the test.

## Skip and blocked policy

- Skips must include explicit reasons.
- Operators that are known unstable for CI should be marked as `blocked` in the registry with `skip_reason`.
- Blocked operators remain visible to tooling and reports, but are excluded from default registry-driven execution unless explicitly requested.
