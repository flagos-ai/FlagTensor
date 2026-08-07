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

"""Extract all pytest marks from conf/operators.yaml.

Outputs a list of operator marks for use in CI scripts and acceptance testing.
Usage:
    python tools/get_marks.py                  # all marks
    python tools/get_marks.py --stage stable   # stable-stage only
    python tools/get_marks.py --category unary # category filter
    python tools/get_marks.py --output ops.txt # write to file
"""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = ROOT / "conf" / "operators.yaml"


def main():
    parser = argparse.ArgumentParser(description="Extract operator marks from YAML registry")
    parser.add_argument(
        "--stage", default=None,
        help="Filter by stage name (alpha, beta, stable, experimental, active)",
    )
    parser.add_argument(
        "--category", default=None,
        help="Filter by category (unary, binary, contraction, sparse)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Write marks to file instead of stdout",
    )
    parser.add_argument(
        "--field", default="name",
        choices=["name", "id", "correctness_mark", "benchmark_mark"],
        help="Which field to output (default: name)",
    )
    args = parser.parse_args()

    with open(YAML_PATH) as f:
        data = yaml.safe_load(f)

    ops = data.get("ops", [])
    marks = []

    for op in ops:
        # Filter by stage
        if args.stage:
            stages = op.get("stages", [])
            stage_names = []
            for s in stages:
                if isinstance(s, dict):
                    stage_names.extend(s.keys())
            if args.stage not in stage_names:
                continue

        # Filter by category
        if args.category:
            cat = op.get("category", "")
            if cat != args.category:
                continue

        value = op.get(args.field, "")
        if value:
            marks.append(str(value))

    output_text = "\n".join(marks) + "\n"

    if args.output:
        Path(args.output).write_text(output_text)
        print(f"Wrote {len(marks)} marks to {args.output}")
    else:
        sys.stdout.write(output_text)


if __name__ == "__main__":
    main()
