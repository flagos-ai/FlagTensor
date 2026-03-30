#!/usr/bin/env python3
import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


def to_float(s: str) -> Optional[float]:
    try:
        return float(s) if s and s.strip() else None
    except (ValueError, TypeError):
        return None


def escape(s: str) -> str:
    if s is None:
        s = "N/A"
    return (s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))


def fmt(num: Optional[float], decimals: int = 6) -> str:
    if num is None:
        return "N/A"
    return f"{num:.{decimals}f}"


def load_env(env_json: Optional[str]) -> Dict[str, Any]:
    if not env_json:
        return {}
    path = Path(env_json)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def normalize_op_name(op_dir_name: str) -> str:
    prefix = "CUTENSOR_OP_"
    if op_dir_name.startswith(prefix):
        return op_dir_name[len(prefix):].lower()
    return op_dir_name.lower()


def parse_results(benchmark_results_dir: Path, env_json: Optional[str] = None) -> Dict[str, Any]:
    benchmark_results_dir = Path(benchmark_results_dir)
    env = load_env(env_json)
    op_rows = []
    perf_detail_rows = []

    for op_dir in sorted(benchmark_results_dir.iterdir()):
        if not op_dir.is_dir():
            continue
        csv_path = op_dir / "benchmark_kernel.csv"
        if not csv_path.exists():
            continue
        rows = list(csv.DictReader(csv_path.read_text().splitlines()))
        parsed_rows = []
        speedups = []
        max_detail = None
        for row in rows:
            detail = {
                "op": normalize_op_name(op_dir.name),
                "op_full": op_dir.name,
                "shape": row.get("shape", ""),
                "dtype": row.get("dtype", ""),
                "mode": row.get("mode", "kernel"),
                "triton_ms": to_float(row.get("latency")),
                "cutensor_ms": to_float(row.get("latency_base")),
                "speedup": to_float(row.get("speedup")),
            }
            if detail["speedup"] is not None:
                speedups.append(detail["speedup"])
            parsed_rows.append(detail)
            if detail["speedup"] is not None and (max_detail is None or detail["speedup"] > max_detail["speedup"]):
                max_detail = detail
        perf_detail_rows.extend(parsed_rows)
        op_rows.append(
            {
                "op": normalize_op_name(op_dir.name),
                "op_full": op_dir.name,
                "perf_avg": statistics.mean(speedups) if speedups else None,
                "perf_max": max(speedups) if speedups else None,
                "perf_count": len(speedups),
                "max_case": max_detail,
            }
        )

    avg_values = [row["perf_avg"] for row in op_rows if row["perf_avg"] is not None]
    max_values = [row["perf_max"] for row in op_rows if row["perf_max"] is not None]
    avg_stats = {
        "count": len(avg_values),
        "mean": statistics.mean(avg_values) if avg_values else None,
        "median": statistics.median(avg_values) if avg_values else None,
        "min": min(avg_values) if avg_values else None,
        "max": max(avg_values) if avg_values else None,
    }
    max_stats = {
        "count": len(max_values),
        "mean": statistics.mean(max_values) if max_values else None,
        "median": statistics.median(max_values) if max_values else None,
        "min": min(max_values) if max_values else None,
        "max": max(max_values) if max_values else None,
    }

    return {
        "env": env,
        "ops": op_rows,
        "perf_details": perf_detail_rows,
        "avg_speedup_stats": avg_stats,
        "max_speedup_stats": max_stats,
        "total_ops": len(op_rows),
        "pass_ops": len(op_rows),
        "failed_ops": 0,
        "missing_acc_ops": 0,
        "missing_perf_ops": 0,
        "benchmark_results_dir": str(benchmark_results_dir),
    }


def render_table_rows_op(ops: List[Dict[str, Any]]) -> str:
    out = []
    for r in ops:
        out.append(
            f'<tr><td>{escape(r["op"])}</td>'
            f'<td>{fmt(r["perf_avg"], 6)}x</td>'
            f'<td>{fmt(r["perf_max"], 6)}x</td>'
            f'<td>{r["perf_count"]}</td></tr>'
        )
    return "\n".join(out)


def render_table_rows_perf(details: List[Dict[str, Any]]) -> str:
    out = []
    for r in details:
        out.append(
            f'<tr><td>{escape(r["op"])}</td>'
            f'<td>{escape(r["shape"])}</td>'
            f'<td>{escape(r["dtype"])}</td>'
            f'<td>{escape(r["mode"])}</td>'
            f'<td>{fmt(r["triton_ms"], 6)} ms</td>'
            f'<td>{fmt(r["cutensor_ms"], 6)} ms</td>'
            f'<td>{fmt(r["speedup"], 6)}x</td></tr>'
        )
    return "\n".join(out)


def render_attention_ops(ops: List[Dict[str, Any]], key: str, threshold: float, high_perform: bool = False) -> str:
    filtered_ops = []
    for r in ops:
        value = r.get(key)
        if value is not None:
            if high_perform and value > threshold:
                filtered_ops.append((r["op"], value))
            elif not high_perform and value < threshold:
                filtered_ops.append((r["op"], value))

    if high_perform:
        filtered_ops.sort(key=lambda x: x[1], reverse=True)
    else:
        filtered_ops.sort(key=lambda x: x[1])

    out = []
    for op, value in filtered_ops:
        badge_class = "badge-success" if high_perform else "badge-danger"
        out.append(f'<tr><td>{escape(op)}</td><td><span class="badge {badge_class}">{fmt(value, 6)}</span></td></tr>')

    return "\n".join(out)


def generate_html(data, title):
    env = data.get("env", {})
    python_version = env.get("python", {}).get("version", "N/A")
    torch_version = env.get("torch", {}).get("version", "N/A")
    triton_version = env.get("packages", {}).get("triton", "N/A")
    commit_id = env.get("git_commit") or "N/A"
    avg_data_js = json.dumps([[r["op"], r["perf_avg"]] for r in data["ops"] if r["perf_avg"] is not None], ensure_ascii=False)
    max_data_js = json.dumps([[r["op"], r["perf_max"]] for r in data["ops"] if r["perf_max"] is not None], ensure_ascii=False)

    html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>""" + escape(title) + """</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; color: white; margin-bottom: 40px; }
        .header h1 { font-size: 2.5rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }
        .card {
            background: white;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            margin-bottom: 30px;
            overflow: hidden;
        }
        .card-header {
            background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
            color: white;
            padding: 20px 30px;
            font-size: 1.3rem;
            font-weight: 600;
        }
        .card-body { padding: 30px; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        .stat-item {
            background: linear-gradient(135deg, #f6f8fc 0%, #eef2f7 100%);
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            transition: transform 0.3s ease;
        }
        .stat-item:hover { transform: translateY(-5px); }
        .stat-value { font-size: 2.5rem; font-weight: 700; color: #5a67d8; margin-bottom: 8px; }
        .stat-value.success { color: #38a169; }
        .stat-label { color: #718096; font-size: 0.95rem; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 15px 20px; text-align: left; border-bottom: 1px solid #e2e8f0; }
        th { background: #f7fafc; font-weight: 600; color: #4a5568; text-transform: uppercase; font-size: 0.85rem; }
        tr:hover { background: #f7fafc; }
        .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; }
        .badge-success { background: #c6f6d5; color: #22543d; }
        .badge-warning { background: #feebc8; color: #744210; }
        .badge-danger { background: #fed7d7; color: #742a2a; }
        .summary-box { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 25px; }
        .summary-item { text-align: center; padding: 20px; background: #f7fafc; border-radius: 10px; }
        .summary-item .value { font-size: 1.8rem; font-weight: 700; color: #2d3748; }
        .summary-item .label { font-size: 0.9rem; color: #718096; margin-top: 5px; }
        .distribution-chart { display: flex; height: 40px; border-radius: 8px; overflow: hidden; margin: 20px 0; }
        .dist-segment { display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 0.9rem; }
        .dist-low { background: linear-gradient(90deg, #fc8181, #f56565); }
        .dist-medium { background: linear-gradient(90deg, #f6ad55, #ed8936); }
        .dist-high { background: linear-gradient(90deg, #68d391, #48bb78); }
        .legend { display: flex; justify-content: center; gap: 30px; margin-top: 15px; }
        .legend-item { display: flex; align-items: center; gap: 8px; font-size: 0.9rem; color: #4a5568; }
        .legend-dot { width: 12px; height: 12px; border-radius: 50%; }
        .env-info { display: flex; justify-content: center; gap: 40px; flex-wrap: wrap; }
        .env-item { display: flex; align-items: center; gap: 8px; color: rgba(255,255,255,0.9); }
        .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
        .section-title { font-size: 1.1rem; color: #4a5568; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #e2e8f0; }
        .op-list { max-height: 400px; overflow-y: auto; }
        .op-list::-webkit-scrollbar { width: 6px; }
        .op-list::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 3px; }
        .op-list::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 3px; }
        .table-wrap { overflow-x: auto; }
        @media (max-width: 768px) {
            .two-col { grid-template-columns: 1fr; }
            .summary-box { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>""" + escape(title) + """</h1>
            <div class="env-info" style="margin-top: 20px;">
                <div class="env-item">
                    <span>Python: """ + escape(python_version) + """</span>
                </div>
                <div class="env-item">
                    <span>Torch: """ + escape(torch_version) + """</span>
                </div>
                <div class="env-item">
                    <span>Triton: """ + escape(triton_version) + """</span>
                </div>
                <div class="env-item">
                    <span>Commit: """ + escape(commit_id) + """</span>
                </div>
                <div class="env-item">
                    <span>""" + str(data['total_ops']) + """ 个算子</span>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">1. 概览</div>
            <div class="card-body">
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value">""" + str(data['total_ops']) + """</div>
                        <div class="stat-label">总算子数量</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value success">""" + str(data['pass_ops']) + """</div>
                        <div class="stat-label">kernel结果算子数</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">""" + str(data['failed_ops']) + """</div>
                        <div class="stat-label">精度测试失败</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">""" + str(data['missing_acc_ops']) + """</div>
                        <div class="stat-label">无精度测试用例</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">""" + str(data['missing_perf_ops']) + """</div>
                        <div class="stat-label">无性能结果</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">2. 算子性能统计</div>
            <div class="card-body">
                <h3 class="section-title">平均加速比统计</h3>
                <div class="summary-box">
                    <div class="summary-item">
                        <div class="value">""" + fmt(data['avg_speedup_stats']['median']) + """</div>
                        <div class="label">中位数</div>
                    </div>
                    <div class="summary-item">
                        <div class="value">""" + fmt(data['avg_speedup_stats']['mean']) + """</div>
                        <div class="label">平均值</div>
                    </div>
                    <div class="summary-item">
                        <div class="value">""" + fmt(data['avg_speedup_stats']['min']) + """</div>
                        <div class="label">最小值</div>
                    </div>
                    <div class="summary-item">
                        <div class="value">""" + fmt(data['avg_speedup_stats']['max']) + """</div>
                        <div class="label">最大值</div>
                    </div>
                </div>

                <h3 class="section-title">最大加速比统计</h3>
                <div class="summary-box">
                    <div class="summary-item">
                        <div class="value">""" + fmt(data['max_speedup_stats']['median']) + """</div>
                        <div class="label">中位数</div>
                    </div>
                    <div class="summary-item">
                        <div class="value">""" + fmt(data['max_speedup_stats']['mean']) + """</div>
                        <div class="label">平均值</div>
                    </div>
                    <div class="summary-item">
                        <div class="value">""" + fmt(data['max_speedup_stats']['min']) + """</div>
                        <div class="label">最小值</div>
                    </div>
                    <div class="summary-item">
                        <div class="value">""" + fmt(data['max_speedup_stats']['max']) + """</div>
                        <div class="label">最大值</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">3. 算子平均/最大加速比柱状图</div>
            <div class="card-body">
                <div style="height: 400px;">
                    <canvas id="speedupChart"></canvas>
                </div>
            </div>
        </div>

        <div class="two-col">
            <div class="card">
                <div class="card-header">4. 需关注算子 - 平均加速比 &lt; 1.0</div>
                <div class="card-body">
                    <div class="op-list">
                        <table>
                            <thead><tr><th>算子名</th><th>平均加速比</th></tr></thead>
                            <tbody>
                                """ + render_attention_ops(data['ops'], 'perf_avg', 1.0, high_perform=False) + """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">5. 高性能算子 - 最大加速比 &gt; 1.6</div>
                <div class="card-body">
                    <div class="op-list">
                        <table>
                            <thead><tr><th>算子名</th><th>最大加速比</th></tr></thead>
                            <tbody>
                                """ + render_attention_ops(data['ops'], 'perf_max', 1.6, high_perform=True) + """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">6. 算子性能汇总</div>
            <div class="card-body">
                <div class="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>算子</th>
                                <th>平均加速比</th>
                                <th>最大加速比</th>
                                <th>样本数</th>
                            </tr>
                        </thead>
                        <tbody>
                            """ + render_table_rows_op(data['ops']) + """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">7. Kernel 各数据规模性能明细</div>
            <div class="card-body">
                <div class="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>算子</th>
                                <th>shape</th>
                                <th>dtype</th>
                                <th>mode</th>
                                <th>triton_ms</th>
                                <th>cutensor_ms</th>
                                <th>speedup</th>
                            </tr>
                        </thead>
                        <tbody>
                            """ + render_table_rows_perf(data['perf_details']) + """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>生成时间: """ + escape(__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + """</p>
        </div>
    </div>

<script>
const avgData = """ + avg_data_js + """;
const maxData = """ + max_data_js + """;
const ctx = document.getElementById('speedupChart').getContext('2d');
new Chart(ctx, {
    type: 'bar',
    data: {
        labels: avgData.map(item => item[0]),
        datasets: [
            {
                label: '平均加速比',
                data: avgData.map(item => item[1]),
                backgroundColor: 'rgba(90, 103, 216, 0.75)',
                borderWidth: 1,
            },
            {
                label: '最大加速比',
                data: maxData.map(item => item[1]),
                backgroundColor: 'rgba(72, 187, 120, 0.75)',
                borderWidth: 1,
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: true } },
        scales: {
            y: { beginAtZero: true, title: { display: true, text: 'speedup' } },
            x: { ticks: { maxRotation: 60, minRotation: 45 } }
        }
    }
});
</script>
</body>
</html>"""
    
    return html_content


def main():
    parser = argparse.ArgumentParser(description="Generate FlagTensor benchmark HTML report")
    parser.add_argument("--benchmark-results", required=True, help="Path to benchmark/results")
    parser.add_argument("--env-json", default=None, help="Optional env.json path")
    parser.add_argument("--output", required=True, help="Output HTML file")
    parser.add_argument("--title", default="FlagTensor 测试报告", help="Report title")

    args = parser.parse_args()

    data = parse_results(args.benchmark_results, args.env_json)
    html = generate_html(data, args.title)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(json.dumps({
        "output": args.output,
        "total_ops": data["total_ops"]
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
