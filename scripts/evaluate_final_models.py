#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="*", default=["full_ft", "selective_ft", "lora_ft"])
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--full-metrics", action="store_true")
    args = parser.parse_args()

    for strategy in args.strategies:
        cmd = [
            sys.executable,
            "scripts/evaluate_model.py",
            "--strategy", strategy,
            "--run-name", "final_1epoch",
            "--split", args.split,
        ]
        if args.max_samples > 0:
            cmd.extend(["--max-samples", str(args.max_samples)])
        if args.full_metrics:
            cmd.append("--full-metrics")
        print("\n=== EVALUATE:", " ".join(cmd), "===\n")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
