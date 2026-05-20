#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/hyperparameters.yaml")
    parser.add_argument("--strategies", nargs="*", default=["full_ft", "selective_ft", "lora_ft"])
    args = parser.parse_args()

    for strategy in args.strategies:
        cmd = [
            sys.executable,
            "scripts/tune_strategy.py",
            "--config", args.config,
            "--strategy", strategy,
        ]
        print("\n=== TUNING:", " ".join(cmd), "===\n")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
