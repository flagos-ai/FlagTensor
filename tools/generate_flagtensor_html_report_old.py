#!/usr/bin/env python3
import argparse
import csv
import html
import json
import math
import statistics
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_rows(path: Path):
    rows = []
    if not path or not path.exists():
        return rows
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def to_float(value):
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def avg_speedup(rows):
    vals = [to_float(r.get("speedup")) for r in rows]
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def fmt(value, digits=6, suffix=""):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def badge(status):
    status = status or "N/A"
    cls = "badge-neutral"
    if status == "PASS":
        cls = "badge-success"
    elif status in {"FAIL", "ERROR"}:
        cls = "badge-danger"
    elif status in {"MISSING", "SKIP", "N/A"}:
        cls = "badge-warning"
    return f'<span class="badge {cls}">{html.escape(status)}</span>'


def escape(v):
    return html.escape(str(v))


def parse_results(smoke_dir: Path, libtuner_dir: Path):
    smoke_summary = load_json(smoke_dir / "summary.json")
    lib_summary = load_json(libtuner_dir / "summary.json")
    ops = smoke_summary.get("ops", [])

    op_rows = []
    perf_detail_rows = []
    tuner_detail_rows = []
    perf_avgs = []

    for op in ops:
        correctness = smoke_summary.get("correctness", {}).get(op, {})
        performance = smoke_summary.get("performance", {}).get(op, {})
        perf_csv = Path(performance["benchmark_csv"]) if performance.get("benchmark_csv") else None
        perf_rows = load_csv_rows(perf_csv) if perf_csv else []
        perf_avg = avg_speedup(perf_rows)
        if perf_avg is not None:
            perf_avgs.append(perf_avg)

        tuner = lib_summary.get("libtuner_compare", {}).get(op, {})
        cold = tuner.get("cold", {})
        warm = tuner.get("warm", {})
        cold_csv = Path(cold["benchmark_csv"]) if cold.get("benchmark_csv") else None
        warm_csv = Path(warm["benchmark_csv"]) if warm.get("benchmark_csv") else None
        cold_rows = load_csv_rows(cold_csv) if cold_csv else []
        warm_rows = load_csv_rows(warm_csv) if warm_csv else []
        cold_avg = avg_speedup(cold_rows)
        warm_avg = avg_speedup(warm_rows)

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

    # Store original average speedups for statistics
    original_perf_avgs = perf_avgs.copy()

    # Update op_rows to use max speedup for display
    max_speedups = []
    for r in op_rows:
        op = r["op"]
        if op in max_per_op and max_per_op[op]["speedup"] is not None:
            r["perf_avg"] = max_per_op[op]["speedup"]
            max_speedups.append(max_per_op[op]["speedup"])

    # Use original averages for statistics, but max speedups for display
    perf_avgs_for_stats = original_perf_avgs

    pass_count = sum(1 for r in op_rows if r["correctness"] == "PASS" and r["perf"] == "PASS")
    tuner_pass_count = sum(1 for r in op_rows if r["cold"] == "PASS" and r["warm"] == "PASS")
    valid_perf_rows = [r for r in op_rows if r["perf_avg"] is not None]
    min_perf_row = min(valid_perf_rows, key=lambda r: r["perf_avg"]) if valid_perf_rows else None
    max_perf_row = max(valid_perf_rows, key=lambda r: r["perf_avg"]) if valid_perf_rows else None
    perf_stats = {
        "count": len(perf_avgs_for_stats),
        "mean": statistics.mean(perf_avgs_for_stats) if perf_avgs_for_stats else None,
        "median": statistics.median(perf_avgs_for_stats) if perf_avgs_for_stats else None,
        "min": min(perf_avgs_for_stats) if perf_avgs_for_stats else None,
        "max": max(perf_avgs_for_stats) if perf_avgs_for_stats else None,
        "gt1": sum(1 for v in perf_avgs_for_stats if v > 1.0),
        "between": sum(1 for v in perf_avgs_for_stats if v is not None and 0.8 <= v <= 1.0),
        "lt08": sum(1 for v in perf_avgs_for_stats if v is not None and v < 0.8),
        "min_op": min_perf_row["op"] if min_perf_row else None,
        "max_op": max_perf_row["op"] if max_perf_row else None,
    }
    return {
        "ops": op_rows,
        "perf_details": perf_detail_rows,
        "tuner_details": tuner_detail_rows,
        "perf_stats": perf_stats,
        "total_ops": len(op_rows),
        "pass_ops": pass_count,
        "tuner_pass_ops": tuner_pass_count,
        "env": load_json(smoke_dir / "env.json") if (smoke_dir / "env.json").exists() else {},
        "smoke_dir": str(smoke_dir),
        "libtuner_dir": str(libtuner_dir),
    }


def render_table_rows_op(rows):
    out = []
    for row in rows:
        out.append(
            "<tr>"
            f"<td>{escape(row['op'])}</td>"
            f"<td>{badge(row['correctness'])}</td>"
            f"<td>{badge(row['perf'])}</td>"
            f"<td>{fmt(row['perf_avg'], suffix='x')}</td>"
            f"<td>{badge(row['cold'])}</td>"
            f"<td>{fmt(row['cold_avg'], suffix='x')}</td>"
            f"<td>{badge(row['warm'])}</td>"
            f"<td>{fmt(row['warm_avg'], suffix='x')}</td>"
            "</tr>"
        )
    return "\n".join(out)


def render_table_rows_perf(rows):
    out = []
    for row in rows:
        out.append(
            "<tr>"
            f"<td>{escape(row['op'])}</td>"
            f"<td>{escape(row['shape'])}</td>"
            f"<td>{escape(row['dtype'])}</td>"
            f"<td>{fmt(row['triton_ms'], suffix=' ms')}</td>"
            f"<td>{fmt(row['cutensor_ms'], suffix=' ms')}</td>"
            f"<td>{fmt(row['speedup'], suffix='x')}</td>"
            "</tr>"
        )
    return "\n".join(out)


def render_table_rows_tuner(rows):
    out = []
    for row in rows:
        out.append(
            "<tr>"
            f"<td>{escape(row['op'])}</td>"
            f"<td>{escape(row['mode'])}</td>"
            f"<td>{escape(row['shape'])}</td>"
            f"<td>{escape(row['dtype'])}</td>"
            f"<td>{fmt(row['triton_ms'], suffix=' ms')}</td>"
            f"<td>{fmt(row['cutensor_ms'], suffix=' ms')}</td>"
            f"<td>{fmt(row['speedup'], suffix='x')}</td>"
            "</tr>"
        )
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
    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape(title)}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 40px 20px; }}
        .container {{ max-width: 1440px; margin: 0 auto; }}
        .header {{ text-align: center; color: white; margin-bottom: 32px; }}
        .header h1 {{ font-size: 2.4rem; margin-bottom: 10px; }}
        .env-info {{ display: flex; justify-content: center; gap: 24px; flex-wrap: wrap; color: rgba(255,255,255,0.92); }}
        .card {{ background: white; border-radius: 16px; box-shadow: 0 10px 40px rgba(0,0,0,0.15); margin-bottom: 24px; overflow: hidden; }}
        .card-header {{ background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%); color: white; padding: 18px 24px; font-size: 1.2rem; font-weight: 600; }}
        .card-body {{ padding: 24px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; }}
        .stat-item {{ background: #f7fafc; border-radius: 12px; padding: 20px; text-align: center; }}
        .stat-value {{ font-size: 2rem; font-weight: 700; color: #5a67d8; }}
        .stat-label {{ color: #718096; margin-top: 6px; font-size: 0.92rem; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; font-size: 0.92rem; }}
        th {{ background: #f7fafc; font-weight: 600; color: #4a5568; position: sticky; top: 0; }}
        .table-wrap {{ max-height: 560px; overflow: auto; border: 1px solid #e2e8f0; border-radius: 12px; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 16px; font-size: 0.82rem; font-weight: 600; }}
        .badge-success {{ background: #c6f6d5; color: #22543d; }}
        .badge-warning {{ background: #feebc8; color: #744210; }}
        .badge-danger {{ background: #fed7d7; color: #742a2a; }}
        .badge-neutral {{ background: #e2e8f0; color: #4a5568; }}
        .sub {{ color: #718096; font-size: 0.92rem; margin-top: 8px; }}
        .controls {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }}
        .controls input, .controls select {{ padding: 8px 12px; border: 1px solid #d2d6dc; border-radius: 8px; }}
        canvas {{ max-height: 420px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{escape(title)}</h1>
            <div class="env-info">
                <span>算子数: {data['total_ops']}</span>
                <span>Python: {escape(python_version)}</span>
                <span>Torch: {escape(torch_version)}</span>
                <span>Triton: {escape(triton_version)}</span>
                <span>Commit: {escape(commit_id)}</span>
            </div>
        </div>

        <div class="card">
            <div class="card-header">1. 概览</div>
            <div class="card-body">
                <div class="stats-grid">
                    <div class="stat-item"><div class="stat-value">{data['total_ops']}</div><div class="stat-label">总算子数</div></div>
                    <div class="stat-item"><div class="stat-value">{data['pass_ops']}</div><div class="stat-label">correctness+perf 通过</div></div>
                    <div class="stat-item"><div class="stat-value">{data['tuner_pass_ops']}</div><div class="stat-label">libtuner cold+warm 通过</div></div>
                    <div class="stat-item"><div class="stat-value">{fmt(data['perf_stats']['mean'])}</div><div class="stat-label">平均加速比</div></div>
                    <div class="stat-item"><div class="stat-value">{fmt(data['perf_stats']['median'])}</div><div class="stat-label">中位数</div></div>
                    <div class="stat-item"><div class="stat-value">{fmt(data['perf_stats']['min'])}</div><div class="stat-label">最小值（{escape(data['perf_stats']['min_op'] or 'N/A')}）</div></div>
                    <div class="stat-item"><div class="stat-value">{fmt(data['perf_stats']['max'])}</div><div class="stat-label">最大值（{escape(data['perf_stats']['max_op'] or 'N/A')}）</div></div>
                </div>
                <div class="sub">结果目录：{escape(data['smoke_dir'])} ｜ libtuner目录：{escape(data['libtuner_dir'])}</div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">2. 算子加速比</div>
            <div class="card-body">
                <canvas id="speedupChart"></canvas>
            </div>
        </div>

        <div class="card">
            <div class="card-header">3. 算子汇总</div>
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
                            {render_table_rows_op(data['ops'])}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">4. 各数据规模性能明细</div>
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
                            {render_table_rows_perf(data['perf_details'])}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

<script>
const allData = {all_data_js};
const ctx = document.getElementById('speedupChart').getContext('2d');
new Chart(ctx, {{
    type: 'bar',
    data: {{
        labels: allData.map(item => item[0]),
        datasets: [{{
            label: '最大加速比',
            data: allData.map(item => item[1]),
            backgroundColor: allData.map(item => item[1] > 1.0 ? 'rgba(72, 187, 120, 0.8)' : 'rgba(245, 101, 101, 0.8)'),
            borderWidth: 1
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            y: {{ beginAtZero: true, title: {{ display: true, text: 'speedup' }} }},
            x: {{ ticks: {{ maxRotation: 60, minRotation: 45 }} }}
        }}
    }}
}});
</script>
</body>
</html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-results", required=True)
    parser.add_argument("--libtuner-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="FlagTensor 算子测试详细报告")
    args = parser.parse_args()

    smoke_dir = Path(args.smoke_results).resolve()
    libtuner_dir = Path(args.libtuner_results).resolve()
    output = Path(args.output).resolve()
    data = parse_results(smoke_dir, libtuner_dir)
    output.write_text(generate_html(data, args.title), encoding="utf-8")
    print(json.dumps({"output": str(output), "total_ops": data["total_ops"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
