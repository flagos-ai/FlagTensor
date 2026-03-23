#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from env_utils import build_env_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve() if args.project_root else None
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_env_payload(project_root=project_root)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
