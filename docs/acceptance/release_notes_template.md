# FlagTensor Release Notes Template

## Version X.Y.Z (YYYY-MM-DD)

### Summary
Brief description of this release (2-3 sentences).

### Highlights
- Major feature 1
- Major feature 2
- Major feature 3

### New Operators
| Operator | Category | Benchmark Modes | Status |
| --- | --- | --- | --- |
| operator_name | unary/binary/contraction/sparse | kernel/operator/wrapper | stable/experimental |

### Improvements
- **Testing**: Added category-level benchmark entry points for unary/binary operators
- **CI/CD**: Introduced acceptance-level workflow with full operator coverage
- **Documentation**: Added standard acceptance commands and CI matrix documentation
- **Quality**: Enhanced static quality gate with registry consistency checks

### Bug Fixes
- Fixed issue in [operator_name] correctness for [dtype]
- Resolved benchmark CSV selection bug for [mode]
- Fixed marker registration warnings in category benchmarks

### Breaking Changes
- None (or list breaking changes with migration guide)

### Known Issues
- `tensor_contraction_trinary` remains blocked due to float64 stability in CI
- See `docs/acceptance/known_issues.md` for full list

### Performance
- Average speedup across all operators: X.XXx vs cuTensor
- Notable improvements:
  - [operator_name]: +XX% speedup in kernel mode
  - [operator_name]: +XX% speedup in operator mode

### Testing Coverage
- **Total Operators**: XX
- **Stable**: XX
- **Experimental**: X
- **Blocked**: X
- **Correctness Pass Rate**: XX.X%
- **Performance Pass Rate**: XX.X%

### Compatibility
- **Python**: 3.10+
- **Triton**: X.X.X
- **cuTensor**: X.X.X
- **CUDA**: X.X

### Installation
```bash
pip install flagtensor==X.Y.Z
```

### Upgrade Guide
If upgrading from version A.B.C:
1. Run pre-commit hooks: `pre-commit run --all-files`
2. Verify registry consistency: `python -c "from flagtensor_registry import load_operator_registry; print(len(list(load_operator_registry())))"`
3. Run smoke correctness: `python tools/run_flagtensor_ci.py --smoke --run-correctness`

### Acknowledgments
- Contributor 1
- Contributor 2

### Full Changelog
- [PR #XXX] Description
- [PR #YYY] Description
- [PR #ZZZ] Description
