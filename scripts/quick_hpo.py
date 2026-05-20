#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import optuna

from common.config_utils import load_yaml, save_json
from common.train_pipeline import run_training


SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "full_ft": {
        "learning_rate": [5e-6, 1e-5, 2e-5],
        "weight_decay": [0.0, 0.01],
    },
    "selective_ft": {
        "learning_rate": [1e-5, 2e-5, 5e-5],
        "weight_decay": [0.0, 0.01],
        "freeze_encoder_bottom_n_layers": [16, 20, 24],
    },
    "lora_ft": {
        "learning_rate": [5e-5, 1e-4, 2e-4],
        "weight_decay": [0.0, 0.01],
        "lora_r": [8, 16, 32],
        "lora_alpha": [16, 32, 64],
        "lora_dropout": [0.0, 0.05, 0.1],
    },
}


def suggest_params(trial: optuna.Trial, strategy: str) -> dict[str, Any]:
    return {
        name: trial.suggest_categorical(name, values)
        for name, values in SEARCH_SPACES[strategy].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/hyperparameters.yaml")
    parser.add_argument("--strategy", required=True, choices=["full_ft", "selective_ft", "lora_ft"])
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--train-subset", type=int, default=5000)
    parser.add_argument("--valid-subset", type=int, default=400)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    study_name = f"{cfg.get('optimization', {}).get('study_name_prefix', 'bakis')}_quick_{args.strategy}"
    storage = cfg.get("optimization", {}).get("storage", "sqlite:///results/optuna/studies.db")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, args.strategy)
        params.update(
            {
                "max_epochs": 1,
                "max_steps": args.max_steps,
                "eval_steps": args.eval_steps,
                "save_steps": args.eval_steps,
                "tune_train_subset_size": args.train_subset,
                "tune_valid_subset_size": args.valid_subset,
                "tune_compute_test_metrics": False,
                "auto_resume": False,
            }
        )
        run_name = f"quick_hpo_trial_{trial.number:03d}"
        summary_path = Path("results") / args.strategy / run_name / "summary.json"
        if summary_path.exists():
            result = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            result = run_training(
                args.config,
                args.strategy,
                run_name,
                overrides=params,
                seed=args.seed,
                tune_mode=True,
            )
        trial.set_user_attr("run_name", run_name)
        trial.set_user_attr("params", params)
        return float(result["best_eval_loss"])

    study.optimize(objective, n_trials=args.trials)

    best = {
        "strategy": args.strategy,
        "study_name": study.study_name,
        "objective_metric": "best_eval_loss",
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
        "best_params": study.best_trial.user_attrs["params"],
        "run_name": study.best_trial.user_attrs["run_name"],
    }
    out_path = Path("results") / "optuna" / f"quick_best_{args.strategy}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(best, out_path)
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
