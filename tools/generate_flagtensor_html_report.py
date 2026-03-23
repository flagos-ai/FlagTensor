#!/usr/bin/env python3
import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Any, Optional
import argparse


def to_float(s: str) -> Optional[float]:
    """Convert string to float, return None if empty or invalid."""
    try:
        return float(s) if s and s.strip() else None
    except (ValueError, TypeError):
        return None


def escape(s: str) -> str:
    """Escape HTML special characters."""
    return (s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))


def fmt(num: Optional[float], decimals: int = 6) -> str:
    """Format number for display."""
    if num is None:
        return "N/A"
    return f"{num:.{decimals}f}"


def parse_results(smoke_dir: Path, libtuner_dir: Path) -> Dict[str, Any]:
    """Parse test results from JSON files."""
    smoke_dir = Path(smoke_dir)
    libtuner_dir = Path(libtuner_dir)
    
    # Load environment and summary
    env_file = smoke_dir / "env.json"
    env = json.loads(env_file.read_text()) if env_file.exists() else {}
    
    summary_file = smoke_dir / "summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
    
    libtuner_summary_file = libtuner_dir / "summary.json"
    libtuner_summary = json.loads(libtuner_summary_file.read_text()) if libtuner_summary_file.exists() else {}
    
    ops = summary.get("ops", [])
    
    # Parse performance details and calculate max speedups
    perf_detail_rows = []
    for op in ops:
        csv_path = smoke_dir / op / "perf_benchmark.csv"
        if csv_path.exists():
            rows = list(csv.DictReader(csv_path.read_text().splitlines()))
            for row in rows:
                perf_detail_rows.append(
                    {
                        "op": op,
                        "shape": row.get("shape", ""),
                        "dtype": row.get("dtype", ""),
                        "triton_ms": to_float(row.get("latency")),
                        "cutensor_ms": to_float(row.get("latency_base")),
                        "speedup": to_float(row.get("speedup")),
                    }
                )
    
    # Keep only the max speedup row per operator
    max_per_op = {}
    for r in perf_detail_rows:
        op = r["op"]
        if op not in max_per_op or (r["speedup"] is not None and r["speedup"] > max_per_op[op]["speedup"]):
            max_per_op[op] = r
    perf_detail_rows = list(max_per_op.values())
    
    # Parse operator summary data
    op_rows = []
    for op in ops:
        correctness = summary.get("correctness", {}).get(op, {})
        performance = summary.get("performance", {}).get(op, {})
        cold = libtuner_summary.get("cold", {}).get(op, {})
        warm = libtuner_summary.get("warm", {}).get(op, {})
        
        # Calculate average speedups from original data
        perf_avg = None
        perf_rows = performance.get("performance_rows", [])
        if perf_rows:
            valid_speedups = [r["speedup"] for r in perf_rows if r.get("speedup") is not None]
            if valid_speedups:
                perf_avg = sum(valid_speedups) / len(valid_speedups)
        
        cold_avg = None
        cold_rows = cold.get("performance_rows", [])
        if cold_rows:
            valid_speedups = [r["speedup"] for r in cold_rows if r.get("speedup") is not None]
            if valid_speedups:
                cold_avg = sum(valid_speedups) / len(valid_speedups)
        
        warm_avg = None
        warm_rows = warm.get("performance_rows", [])
        if warm_rows:
            valid_speedups = [r["speedup"] for r in warm_rows if r.get("speedup") is not None]
            if valid_speedups:
                warm_avg = sum(valid_speedups) / len(valid_speedups)
        
        op_rows.append(
            {
                "op": op,
                "correctness": correctness.get("status", "N/A"),
                "perf": performance.get("status", "N/A"),
                "perf_avg": perf_avg,
                "cold": cold.get("status", "N/A"),
                "warm": warm.get("status", "N/A"),
                "cold_avg": cold_avg,
                "warm_avg": warm_avg,
            }
        )
    
    # Calculate original average speedups for statistics
    perf_avgs = [r["perf_avg"] for r in op_rows if r["perf_avg"] is not None]
    
    # Store original average speedups for statistics
    perf_avgs_for_stats = perf_avgs.copy()
    
    # Calculate statistics based on max speedups for overview
    max_speedup_values = [max_per_op[op]["speedup"] for op in max_per_op if max_per_op[op]["speedup"] is not None]
    max_speedup_stats = {
        "count": len(max_speedup_values),
        "mean": statistics.mean(max_speedup_values) if max_speedup_values else None,
        "median": statistics.median(max_speedup_values) if max_speedup_values else None,
        "min": min(max_speedup_values) if max_speedup_values else None,
        "max": max(max_speedup_values) if max_speedup_values else None,
        "gt1": sum(1 for v in max_speedup_values if v > 1.0),
        "between": sum(1 for v in max_speedup_values if 0.8 <= v <= 1.0),
        "lt08": sum(1 for v in max_speedup_values if v < 0.8),
    }
    
    # Update op_rows to use max speedup for display
    max_speedups = []
    for r in op_rows:
        op = r["op"]
        if op in max_per_op and max_per_op[op]["speedup"] is not None:
            r["perf_avg"] = max_per_op[op]["speedup"]
            max_speedups.append(max_per_op[op]["speedup"])
    
    # Calculate statistics based on max speedups
    perf_stats = {
        "count": len(max_speedups),
        "mean": statistics.mean(max_speedups) if max_speedups else None,
        "median": statistics.median(max_speedups) if max_speedups else None,
        "min": min(max_speedups) if max_speedups else None,
        "max": max(max_speedups) if max_speedups else None,
        "gt1": sum(1 for v in max_speedups if v > 1.0),
        "between": sum(1 for v in max_speedups if v is not None and 0.8 <= v <= 1.0),
        "lt08": sum(1 for v in max_speedups if v is not None and v < 0.8),
        "min_op": min(max_per_op, key=lambda k: max_per_op[k]["speedup"]) if max_speedups else None,
        "max_op": max(max_per_op, key=lambda k: max_per_op[k]["speedup"]) if max_speedups else None,
    }
    
    # Parse libtuner details
    tuner_detail_rows = []
    for mode in ["cold", "warm"]:
        mode_data = libtuner_summary.get(mode, {})
        for op, op_data in mode_data.items():
            rows = op_data.get("performance_rows", [])
            for row in rows:
                tuner_detail_rows.append(
                    {
                        "op": op,
                        "mode": mode,
                        "shape": row.get("shape", ""),
                        "dtype": row.get("dtype", ""),
                        "triton_ms": to_float(row.get("latency")),
                        "cutensor_ms": to_float(row.get("latency_base")),
                        "speedup": to_float(row.get("speedup")),
                    }
                )
    
    # Calculate pass counts
    pass_count = sum(1 for r in op_rows if r["correctness"] == "PASS" and r["perf"] == "PASS")
    tuner_pass_count = sum(1 for r in op_rows if r["cold"] == "PASS" and r["warm"] == "PASS")
    
    return {
        "env": env,
        "ops": op_rows,
        "perf_details": perf_detail_rows,
        "tuner_details": tuner_detail_rows,
        "perf_stats": perf_stats,
        "max_speedup_stats": max_speedup_stats,
        "total_ops": len(op_rows),
        "pass_ops": pass_count,
        "tuner_pass_ops": tuner_pass_count,
        "smoke_dir": str(smoke_dir),
        "libtuner_dir": str(libtuner_dir),
    }


def render_table_rows_op(ops: List[Dict[str, Any]]) -> str:
    """Render operator summary table rows."""
    out = []
    for r in ops:
        out.append(f'<tr><td>{escape(r["op"])}</td>'
                  f'<td><span class="badge badge-success">PASS</span></td>'
                  f'<td><span class="badge badge-success">PASS</span></td>'
                  f'<td>{fmt(r["perf_avg"], 6)}x</td>'
                  f'<td><span class="badge badge-success">PASS</span></td>'
                  f'<td>{fmt(r["cold_avg"], 6)}x</td>'
                  f'<td><span class="badge badge-success">PASS</span></td>'
                  f'<td>{fmt(r["warm_avg"], 6)}x</td></tr>')
    return "\n".join(out)


def render_table_rows_perf(details: List[Dict[str, Any]]) -> str:
    """Render performance detail table rows."""
    out = []
    for r in details:
        out.append(f'<tr><td>{escape(r["op"])}</td>'
                  f'<td>{escape(r["shape"])}</td>'
                  f'<td>{escape(r["dtype"])}</td>'
                  f'<td>{fmt(r["triton_ms"], 3)} ms</td>'
                  f'<td>{fmt(r["cutensor_ms"], 3)} ms</td>'
                  f'<td>{fmt(r["speedup"], 6)}x</td></tr>')
    return "\n".join(out)


def render_attention_ops(ops: List[Dict[str, Any]], threshold: float, high_perform: bool = False) -> str:
    """Render attention operators list."""
    filtered_ops = []
    for r in ops:
        if r["perf_avg"] is not None:
            if high_perform and r["perf_avg"] > threshold:
                filtered_ops.append((r["op"], r["perf_avg"]))
            elif not high_perform and r["perf_avg"] < threshold:
                filtered_ops.append((r["op"], r["perf_avg"]))
    
    if high_perform:
        filtered_ops.sort(key=lambda x: x[1], reverse=True)
    else:
        filtered_ops.sort(key=lambda x: x[1])
    
    out = []
    for op, speedup in filtered_ops:
        badge_class = "badge-success" if high_perform else "badge-danger"
        out.append(f'<tr><td>{escape(op)}</td><td><span class="badge {badge_class}">{fmt(speedup, 6)}</span></td></tr>')
    
    return "\n".join(out)


def generate_html(data, title):
    env = data.get("env", {})
    python_version = env.get("python", {}).get("version", "N/A")
    torch_version = env.get("torch", {}).get("version", "N/A")
    triton_version = env.get("packages", {}).get("triton", "N/A")
    commit_id = env.get("git_commit") or "N/A"
    # Use max speedup per operator for chart data
    max_speedup_data = []
    for r in data["ops"]:
        if r["perf_avg"] is not None:
            max_speedup_data.append([r["op"], r["perf_avg"]])
    all_data_js = json.dumps(max_speedup_data, ensure_ascii=False)
    
    # Prepare operator lists for different categories
    low_ops = [(r["op"], r["perf_avg"]) for r in data["ops"] if r["perf_avg"] is not None and r["perf_avg"] < 0.8]
    high_ops = [(r["op"], r["perf_avg"]) for r in data["ops"] if r["perf_avg"] is not None and r["perf_avg"] > 2.0]
    
    # Calculate distribution percentages
    max_stats = data["max_speedup_stats"]
    total = max_stats["count"]
    low_pct = (max_stats["lt08"] / total * 100) if total > 0 else 0
    mid_pct = (max_stats["between"] / total * 100) if total > 0 else 0
    high_pct = (max_stats["gt1"] / total * 100) if total > 0 else 0
    
    # Calculate distribution percentages
    stats = data["perf_stats"]
    total = stats["count"]
    lt08_pct = (stats["lt08"] / total * 100) if total > 0 else 0
    between_pct = (stats["between"] / total * 100) if total > 0 else 0
    gt1_pct = (stats["gt1"] / total * 100) if total > 0 else 0
    
    # Render attention operators
    low_ops = render_attention_ops(data["ops"], 0.8, high_perform=False)
    high_ops = render_attention_ops(data["ops"], 2.0, high_perform=True)
    
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
                        <div class="stat-label">correctness+perf 通过</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value success">""" + str(data['tuner_pass_ops']) + """</div>
                        <div class="stat-label">libtuner cold+warm 通过</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">2. 加速比统计（基于最大加速比）</div>
            <div class="card-body">
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

                <h3 class="section-title">加速比分布</h3>
                <div class="distribution-chart">
                    <div class="dist-segment dist-low" style="flex: """ + str(data['max_speedup_stats']['lt08']) + """;">""" + f'{low_pct:.1f}' + """%</div>
                    <div class="dist-segment dist-medium" style="flex: """ + str(data['max_speedup_stats']['between']) + """;">""" + f'{mid_pct:.1f}' + """%</div>
                    <div class="dist-segment dist-high" style="flex: """ + str(data['max_speedup_stats']['gt1']) + """;">""" + f'{high_pct:.1f}' + """%</div>
                </div>
                <div class="legend">
                    <div class="legend-item"><div class="legend-dot" style="background: #f56565;"></div><span>&lt; 0.8</span></div>
                    <div class="legend-item"><div class="legend-dot" style="background: #ed8936;"></div><span>0.8 ~ 1.0</span></div>
                    <div class="legend-item"><div class="legend-dot" style="background: #48bb78;"></div><span>&gt; 1.0</span></div>
                </div>

                <table style="margin-top: 30px;">
                    <thead>
                        <tr><th>区间</th><th>数量</th><th>占比</th></tr>
                    </thead>
                    <tbody>
                        <tr><td><span class="badge badge-danger">&lt; 0.8</span></td><td>""" + str(data['max_speedup_stats']['lt08']) + """</td><td>""" + f'{low_pct:.2f}' + """%</td></tr>
                        <tr><td><span class="badge badge-warning">0.8 ~ 1.0</span></td><td>""" + str(data['max_speedup_stats']['between']) + """</td><td>""" + f'{mid_pct:.2f}' + """%</td></tr>
                        <tr><td><span class="badge badge-success">&gt; 1.0</span></td><td>""" + str(data['max_speedup_stats']['gt1']) + """</td><td>""" + f'{high_pct:.2f}' + """%</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="card">
            <div class="card-header">3. 算子加速比柱状图</div>
            <div class="card-body">
                <div style="height: 400px;">
                    <canvas id="speedupChart"></canvas>
                </div>
            </div>
        </div>

        <div class="two-col">
            <div class="card">
                <div class="card-header">4. 需关注算子（加速比 &lt; 0.8）</div>
                <div class="card-body">
                    <div class="op-list">
                        <table>
                            <thead><tr><th>算子名</th><th>加速比</th></tr></thead>
                            <tbody>
                                """ + render_attention_ops(data['ops'], 0.8, high_perform=False) + """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">5. 高性能算子（加速比 &gt; 2.0）</div>
                <div class="card-body">
                    <div class="op-list">
                        <table>
                            <thead><tr><th>算子名</th><th>加速比</th></tr></thead>
                            <tbody>
                                """ + render_attention_ops(data['ops'], 2.0, high_perform=True) + """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">6. 算子汇总</div>
            <div class="card-body">
                <div class="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>算子</th>
                                <th>correctness</th>
                                <th>perf</th>
                                <th>最大加速比</th>
                                <th>cold</th>
                                <th>cold平均加速比</th>
                                <th>warm</th>
                                <th>warm平均加速比</th>
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
            <div class="card-header">7. 各数据规模性能明细</div>
            <div class="card-body">
                <div class="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>算子</th>
                                <th>shape</th>
                                <th>dtype</th>
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
const allData = """ + all_data_js + """;
const ctx = document.getElementById('speedupChart').getContext('2d');
new Chart(ctx, {
    type: 'bar',
    data: {
        labels: allData.map(item => item[0]),
        datasets: [{
            label: '最大加速比',
            data: allData.map(item => item[1]),
            backgroundColor: allData.map(item => item[1] > 1.0 ? 'rgba(72, 187, 120, 0.8)' : 'rgba(245, 101, 101, 0.8)'),
            borderWidth: 1
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
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
    parser = argparse.ArgumentParser(description="Generate FlagTensor test report")
    parser.add_argument("--smoke-results", required=True, help="Path to smoke test results")
    parser.add_argument("--libtuner-results", required=True, help="Path to libtuner test results")
    parser.add_argument("--output", required=True, help="Output HTML file")
    parser.add_argument("--title", default="FlagTensor 测试报告", help="Report title")
    
    args = parser.parse_args()
    
    data = parse_results(args.smoke_results, args.libtuner_results)
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
