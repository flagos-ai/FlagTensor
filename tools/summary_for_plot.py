#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parse FlagTensor benchmark result logs and produce aggregated speedup summaries.

Reads log files emitted by:  pytest -m '<op>' --level core --record log

Usage:
    python tools/summary_for_plot.py result-m_abs--level_core--record_log.log
    python tools/summary_for_plot.py result-*.log --output summary.csv
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_benchmark_log(log_path: str) -> list:
    """Parse a benchmark log file (JSON-lines or structured log format).

    Returns list of dicts with keys: op_name, dtype, shape, latency_base, latency, speedup
    """
    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try JSON first
            try:
                record = json.loads(line)
                if isinstance(record, dict):
                    rows.append(_flatten_record(record))
                continue
            except (json.JSONDecodeError, ValueError):
                pass

            # Try key=value or plain format
            parts = line.split()
            if len(parts) >= 4:
                try:
                    row = {
                        "op_name": parts[0],
                        "dtype": parts[1],
                        "latency_base": float(parts[2]),
                        "latency": float(parts[3]),
                        "speedup": float(parts[2]) / float(parts[3]) if float(parts[3]) > 0 else 0,
                    }
                    rows.append(row)
                except (ValueError, IndexError):
                    pass
    return rows


def _flatten_record(record: dict) -> dict:
    """Extract key metric fields from a nested benchmark record."""
    row = {
        "op_name": record.get("op_name", record.get("op", "")),
        "dtype": record.get("dtype", ""),
    }
    details = record.get("details", record.get("result", []))
    if isinstance(details, list):
        speeds = []
        for d in details:
            if isinstance(d, dict):
                lb = d.get("latency_base", 0)
                lt = d.get("latency", 0)
                speeds.append(float(lb) / float(lt) if float(lt) > 0 else 0)
        row["speedup"] = sum(speeds) / len(speeds) if speeds else 0
    else:
        row["speedup"] = 0
    return row


def aggregate(rows: list) -> dict:
    """Aggregate benchmark rows by op_name and dtype.

    Returns: {op_name: {dtype: avg_speedup}}
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        op = row.get("op_name", "?")
        dtype = row.get("dtype", "?")
        speedup = row.get("speedup", 0)
        if speedup > 0:
            grouped[op][dtype].append(speedup)

    result = {}
    for op, dtypes in grouped.items():
        result[op] = {}
        for dtype, speeds in dtypes.items():
            result[op][dtype] = sum(speeds) / len(speeds)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Parse FlagTensor benchmark logs and output aggregated speedup summaries"
    )
    parser.add_argument("log_file", nargs="+", help="Benchmark log file(s) to parse")
    parser.add_argument("--output", "-o", default=None, help="Output file (CSV format)")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of table")
    args = parser.parse_args()

    all_rows = []
    for pattern in args.log_file:
        for log_path in sorted(Path(".").glob(pattern)):
            all_rows.extend(parse_benchmark_log(str(log_path)))

    if not all_rows:
        print("[WARN] No benchmark rows parsed from log files.")
        sys.exit(0)

    aggregated = aggregate(all_rows)

    if args.json:
        print(json.dumps(aggregated, indent=2))
        return

    # Collect all dtype names
    all_dtypes = set()
    for dtypes in aggregated.values():
        all_dtypes.update(dtypes.keys())
    all_dtypes = sorted(all_dtypes)

    # Print header
    header = ["op_name"] + all_dtypes + ["avg_speedup"]
    rows = []

    for op_name in sorted(aggregated.keys()):
        d = aggregated[op_name]
        speeds = [d.get(dt, "") for dt in all_dtypes]
        valid_speeds = [v for v in d.values() if v > 0]
        avg = sum(valid_speeds) / len(valid_speeds) if valid_speeds else 0
        rows.append([op_name] + speeds + [f"{avg:.4f}"])

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Wrote summary to {args.output}")
    else:
        # Pretty-print table
        col_widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
        print(fmt.format(*header))
        print("-" * sum(col_widths) + "--" * (len(col_widths) - 1))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))


if __name__ == "__main__":
    main()
