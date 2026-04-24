# FlagTensor

FlagTensor is a Triton operator playground aligned with FlagGems-style benchmarking and testing, using cuTensor C APIs as baselines.

## CI workflows

This repository provides two GitHub Actions workflows under `.github/workflows`:

- `flagtensor-ci`: split into `correctness` and `perf` jobs for smoke-style automated validation.
- `flagtensor-weekly`: runs the weekly correctness and benchmark pipeline from an operator list.

### Benchmark modes

Both workflows support the benchmark `mode` input:

- `kernel`
- `operator`

The default mode is `kernel`.

### How to use

- Trigger `flagtensor-ci` from `workflow_dispatch` when you want a quick automated check of the currently covered operators.
- Trigger `flagtensor-weekly` from `workflow_dispatch` when you want to run the weekly-style multi-operator pipeline.
- For `flagtensor-weekly`, you can optionally provide a custom operator list file; otherwise the workflow generates one from the discovered tests.

Run CI correctness locally:

```bash
python tools/run_flagtensor_ci.py --smoke --run-correctness --exclude-op tensor_contraction_trinary --mode kernel --results-dir ci_results_correctness
```

Run CI perf locally in kernel mode:

```bash
python tools/run_flagtensor_ci.py --smoke --run-perf --exclude-op tensor_contraction_trinary --mode kernel --results-dir ci_results_perf
```

Run CI perf locally in operator mode:

```bash
python tools/run_flagtensor_ci.py --smoke --run-perf --exclude-op tensor_contraction_trinary --mode operator --results-dir ci_results_perf_operator
```

Run weekly locally in kernel mode:

```bash
python tools/run_flagtensor_weekly.py --project-root . --op-list weekly_op_test.txt --gpus 0 --mode kernel --results-dir weekly_results_ci
```

Run weekly locally in operator mode:

```bash
python tools/run_flagtensor_weekly.py --project-root . --op-list weekly_op_test.txt --gpus 0 --mode operator --results-dir weekly_results_ci_operator
```

### Current limitation

`tensor_contraction_trinary` is temporarily excluded from automated workflow runs because its `float64` correctness path is not yet stable in CI.
