#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common.config_utils import save_json


DEFAULT_TRIALS = {
    "full_ft": 3,
    "selective_ft": 4,
    "lora_ft": 6,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/hyperparameters.yaml")
    parser.add_argument("--strategies", nargs="*", default=["full_ft", "selective_ft", "lora_ft"])
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--train-subset", type=int, default=5000)
    parser.add_argument("--valid-subset", type=int, default=400)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_best = {}
    for strategy in args.strategies:
        trials = DEFAULT_TRIALS[strategy]
        if strategy == "full_ft":
            max_steps = min(args.max_steps, 600)
            train_subset = min(args.train_subset, 3000)
            valid_subset = min(args.valid_subset, 300)
        else:
            max_steps = args.max_steps
            train_subset = args.train_subset
            valid_subset = args.valid_subset

        cmd = [
            sys.executable,
            "scripts/quick_hpo.py",
            "--config", args.config,
            "--strategy", strategy,
            "--trials", str(trials),
            "--max-steps", str(max_steps),
            "--train-subset", str(train_subset),
            "--valid-subset", str(valid_subset),
            "--eval-steps", str(args.eval_steps),
            "--seed", str(args.seed),
        ]
        print("\n=== QUICK HPO:", " ".join(cmd), "===\n")
        subprocess.run(cmd, check=True)

        best_path = Path("results") / "optuna" / f"quick_best_{strategy}.json"
        all_best[strategy] = json.loads(best_path.read_text(encoding="utf-8"))

    out_path = Path("results") / "optuna" / "quick_best_all.json"
    save_json(all_best, out_path)
    print(json.dumps(all_best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
