# FlagTensor

FlagTensor is a Triton operator playground aligned with FlagGems-style benchmarking and testing, using cuTensor C APIs as baselines.

可以通过 `python tools/run_flagtensor_ci.py --op-list tools/ci_op_list.txt` 执行 CI 基准测试。

生成测试结果之后，通过

```bash
python tools/generate_flagtensor_html_report.py \
  --benchmark-results benchmark/results \
  --env-json ci_results/env.json \
  --output ci_results/FlagTensor_CI_report.html \
  --title "FlagTensor CI 测试报告"
```

若不需要环境信息，可省略 `--env-json`。生成对应 HTML 报告。
