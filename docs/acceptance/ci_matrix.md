# FlagTensor CI Matrix

This document describes the CI/CD workflows and their purposes in the FlagTensor acceptance process.

## Workflows Overview

| Workflow | Trigger | Purpose | GPU Required | Output |
| --- | --- | --- | --- | --- |
| `quality-gate.yaml` | PR, push, manual | Static quality checks (pre-commit, build, registry) | No | Build artifacts, consistency reports |
| `ci.yaml` | PR, push, manual | Smoke-level correctness and performance | No | Smoke results, summary |
| `weekly.yaml` | Manual | Weekly regression on cluster | Yes (via Slurm) | Weekly results, artifacts |
| `acceptance.yaml` | Manual, weekly schedule | Acceptance-level full coverage | No (structure) / Yes (cluster) | Acceptance results, summary |

## Quality Gate Workflow (`quality-gate.yaml`)

### Jobs

#### pre-commit
- **Purpose**: Run static analysis and formatting checks
- **Checks**: YAML syntax, trailing whitespace, flake8, isort, black, clang-format
- **Runtime**: ~2-3 minutes
- **Failure Impact**: Blocks PR merge

#### build-check
- **Purpose**: Verify package can be built and distributed
- **Steps**: Build wheel/sdist, twine check
- **Runtime**: ~1-2 minutes
- **Failure Impact**: Blocks PR merge

#### registry-consistency
- **Purpose**: Ensure operator registry is consistent with codebase
- **Checks**:
  - All impl files exist
  - All correctness test files exist
  - All benchmark test files exist
  - Coverage statistics
- **Runtime**: ~30 seconds
- **Failure Impact**: Blocks PR merge

## CI Workflow (`ci.yaml`)

### Jobs

#### correctness-smoke
- **Purpose**: Validate correctness structure and basic functionality
- **Scope**: All active operators (non-blocked)
- **Mode**: Smoke (reduced shapes/dtypes)
- **Benchmark Mode**: Default kernel, configurable
- **Runtime**: ~5-10 minutes (CPU-only structure check)
- **Output**: summary.json, summary.md, per-operator logs
- **Failure Impact**: Warning only (GPU validation done on cluster)

#### perf-smoke
- **Purpose**: Validate benchmark structure and CSV generation
- **Scope**: All active operators (non-blocked)
- **Mode**: Smoke (reduced shapes)
- **Benchmark Mode**: Default kernel, configurable
- **Runtime**: ~5-10 minutes (CPU-only structure check)
- **Output**: summary.json, summary.md, benchmark CSVs
- **Failure Impact**: Warning only (GPU validation done on cluster)

## Weekly Workflow (`weekly.yaml`)

### Jobs

#### weekly-entry
- **Purpose**: Full weekly regression on GPU cluster
- **Scope**: All active operators (non-blocked) from registry
- **Mode**: Full (all shapes/dtypes)
- **GPU Allocation**: Configurable via `--gpus` parameter
- **Runtime**: 1-2 hours (depends on GPU count)
- **Output**: Weekly results, operator list, artifacts
- **Failure Impact**: Requires investigation

### Weekly Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `op_list` | (generated from registry) | Optional path to operator list file |
| `gpus` | `0` | GPU IDs to use (comma-separated) |
| `mode` | `kernel` | Benchmark mode (kernel/operator/wrapper) |

## Acceptance Workflow (`acceptance.yaml`)

### Jobs

#### correctness-acceptance
- **Purpose**: Acceptance-level correctness validation
- **Scope**: All active operators or specific category
- **Mode**: Full (all shapes/dtypes)
- **Category Filter**: Optional (unary/binary/contraction/sparse)
- **Benchmark Mode**: Configurable
- **Runtime**: 30-60 minutes (CPU structure) / 1-2 hours (GPU cluster)
- **Output**: ACCEPTANCE_SUMMARY.md, summary.json, per-operator logs
- **Failure Impact**: Blocks acceptance

#### perf-acceptance
- **Purpose**: Acceptance-level performance validation
- **Scope**: All active operators or specific category
- **Mode**: Full (all shapes)
- **Category Filter**: Optional (unary/binary/contraction/sparse)
- **Benchmark Mode**: Configurable
- **Runtime**: 30-60 minutes (CPU structure) / 2-4 hours (GPU cluster)
- **Output**: ACCEPTANCE_SUMMARY.md, summary.json, benchmark CSVs, speedup stats
- **Failure Impact**: Blocks acceptance

### Acceptance Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `mode` | `kernel` | Benchmark mode (kernel/operator/wrapper) |
| `category` | `""` (all) | Operator category filter |

### Acceptance Summary Output

The acceptance workflow generates detailed summaries including:

- Total operators tested
- Pass/fail counts and pass rate
- Failed operators list
- Performance speedup statistics (avg, median, min, max)
- Per-operator status table

## Cluster GPU Validation

Since GitHub Actions runners do not have GPU access, actual GPU validation is performed on the cluster using Slurm.

### Standard Slurm Template

```bash
srun -N 1 --job-name <job_name> \
  --nodelist <node_name> \
  --gres=gpu:<gpu_count> \
  --cpus-per-task=$((24*gpu_count)) \
  --mem=$((242144*gpu_count)) \
  docker exec -w /workspace/FlagGems/flagtensor triton_cuda12 \
  bash -lc "<command>"
```

### Cluster Node

- **Primary Node**: `bjdb-h20-node-038`
- **Container**: `triton_cuda12`
- **Container Path**: `/workspace/FlagGems/flagtensor`

## Artifact Storage

All workflows upload artifacts for audit and debugging:

| Artifact | Workflow | Retention | Contents |
| --- | --- | --- | --- |
| `flagtensor-ci-correctness-smoke` | ci.yaml | 30 days | Smoke correctness results |
| `flagtensor-ci-perf-smoke` | ci.yaml | 30 days | Smoke performance results |
| `flagtensor-weekly-results` | weekly.yaml | 90 days | Weekly regression results |
| `flagtensor-acceptance-correctness` | acceptance.yaml | 90 days | Acceptance correctness results |
| `flagtensor-acceptance-perf` | acceptance.yaml | 90 days | Acceptance performance results |
| `flagtensor-build-dist` | quality-gate.yaml | 30 days | Wheel/sdist packages |

## CI Status Indicators

### GitHub Step Summary

Workflows report results to GitHub Step Summary:

- **Quality Gate**: Pre-commit status, build status, registry consistency
- **CI Smoke**: Pass/fail table for correctness and performance
- **Weekly**: Operator count and overall status
- **Acceptance**: Detailed pass/fail statistics, speedup analysis

### Exit Codes

- **0**: All checks passed
- **1**: One or more checks failed
- **Non-zero**: Workflow error (e.g., missing dependencies)

## CI Best Practices

1. **Always run quality gate before PR merge**
2. **Use smoke mode for rapid iteration**
3. **Run acceptance on GPU cluster before release**
4. **Review weekly regression results weekly**
5. **Keep registry in sync with codebase**
6. **Update category benchmark entries when adding operators**

## CI vs Local Testing

| Aspect | CI | Local |
| --- | --- | --- |
| **Speed** | Fast (CPU structure) | Variable |
| **GPU Access** | No | Yes (via Slurm) |
| **Coverage** | Smoke | Full |
| **Purpose** | Structure validation | Functional validation |
| **Recommendation** | Use for PR checks | Use for acceptance validation |
