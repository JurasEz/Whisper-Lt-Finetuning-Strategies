#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.config_utils import load_yaml, save_json
from common.train_pipeline import run_training



def mean(values):
    return sum(values) / len(values) if values else 0.0



def std(values):
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / (len(values) - 1)) ** 0.5



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/hyperparameters.yaml")
    parser.add_argument("--strategies", nargs="*", default=["full_ft", "selective_ft", "lora_ft"])
    args = parser.parse_args()

    cfg_full = load_yaml(args.config)
    seeds = list(cfg_full.get("repeat_policy", {}).get("seed_final_repeats", [42, 52, 62]))
    all_summary: dict[str, dict] = {}

    for strategy in args.strategies:
        best_path = Path("results") / "optuna" / f"best_{strategy}.json"
        if not best_path.exists():
            raise FileNotFoundError(f"Pirma paleisk tune_strategy.py: {best_path}")
        best = json.loads(best_path.read_text(encoding="utf-8"))
        best_params = best["best_params"]

        strategy_results = []
        for seed in seeds:
            run_name = f"best_seed_{seed}"
            summary_path = Path("results") / strategy / run_name / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                summary = run_training(
                    args.config,
                    strategy,
                    run_name,
                    overrides=best_params,
                    seed=seed,
                    tune_mode=False,
                )
            strategy_results.append(summary)

        wers = [x["test_metrics"]["wer"] for x in strategy_results]
        cers = [x["test_metrics"]["cer"] for x in strategy_results]
        sems = [x["test_metrics"]["sem"] for x in strategy_results]
        krs = [x["test_metrics"]["kr"] for x in strategy_results]
        peds = [x["test_metrics"]["ped"] for x in strategy_results]

        all_summary[strategy] = {
            "best_params": best_params,
            "seeds": seeds,
            "wer_mean": mean(wers),
            "wer_std": std(wers),
            "cer_mean": mean(cers),
            "cer_std": std(cers),
            "sem_mean": mean(sems),
            "sem_std": std(sems),
            "kr_mean": mean(krs),
            "kr_std": std(krs),
            "ped_mean": mean(peds),
            "ped_std": std(peds),
        }

    save_json(all_summary, Path("results") / "optuna" / "final_repeats_summary.json")
    print(json.dumps(all_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
