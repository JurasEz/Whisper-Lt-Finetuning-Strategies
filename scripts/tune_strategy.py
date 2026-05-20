#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
from typing import Any

import optuna
from optuna.trial import TrialState
import torch

from common.config_utils import load_yaml, save_json
from common.train_pipeline import run_training


def suggest_from_space(trial: optuna.Trial, name: str, values: list[Any]) -> Any:
    return trial.suggest_categorical(name, values)


def build_trial_overrides(trial: optuna.Trial, cfg_full: dict[str, Any], strategy: str) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    for name, values in cfg_full.get("common_search_space", {}).items():
        overrides[name] = suggest_from_space(trial, name, values)

    for name, values in cfg_full["strategies"][strategy].get("search_space", {}).items():
        overrides[name] = suggest_from_space(trial, name, values)

    return overrides


def make_study(cfg_full: dict[str, Any], strategy: str) -> optuna.Study:
    opt_cfg = cfg_full.get("optimization", {})
    study_name = f"{opt_cfg.get('study_name_prefix', 'bakis')}_{strategy}"
    storage = opt_cfg.get("storage", "sqlite:///results/optuna/studies.db")
    if isinstance(storage, str) and storage.startswith("sqlite:///"):
        sqlite_path = Path(storage.removeprefix("sqlite:///"))
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    direction = opt_cfg.get("direction", "minimize")
    sampler_name = opt_cfg.get("sampler", "tpe").lower()
    pruner_name = opt_cfg.get("pruner", "hyperband").lower()

    sampler = optuna.samplers.TPESampler() if sampler_name == "tpe" else optuna.samplers.RandomSampler()

    if pruner_name == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    elif pruner_name == "median":
        pruner = optuna.pruners.MedianPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def finish_stale_running_trials(study: optuna.Study, config_path: str, cfg_full: dict[str, Any], strategy: str) -> None:
    tuning_seed = int(cfg_full.get("repeat_policy", {}).get("tuning_seed", 42))
    for old_trial in list(study.trials):
        if old_trial.state != TrialState.RUNNING:
            continue
        if not old_trial.params:
            print(f"Skipping stale running trial {old_trial.number}: no saved params.")
            continue

        run_name = old_trial.user_attrs.get("run_name", f"optuna_trial_{old_trial.number:03d}")
        summary_path = Path("results") / strategy / run_name / "summary.json"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                result = json.load(f)
        else:
            print(f"Resuming stale running trial {old_trial.number}: {run_name}")
            result = run_training(
                config_path,
                strategy,
                run_name,
                overrides=dict(old_trial.params),
                seed=tuning_seed,
                tune_mode=True,
            )

        value = float(result["best_eval_loss"])
        trial_id = old_trial._trial_id
        study._storage.set_trial_user_attr(trial_id, "run_name", run_name)
        study._storage.set_trial_user_attr(trial_id, "overrides", dict(old_trial.params))
        study._storage.set_trial_user_attr(trial_id, "best_eval_loss", value)
        study._storage.set_trial_state_values(trial_id, TrialState.COMPLETE, [value])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/hyperparameters.yaml")
    parser.add_argument("--strategy", required=True, choices=["full_ft", "selective_ft", "lora_ft"])
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA nepasiekiama. Tuning paleisk tik GPU mazge.")

    cfg_full = load_yaml(args.config)
    strategy = args.strategy
    tuning_seed = int(cfg_full.get("repeat_policy", {}).get("tuning_seed", 42))
    opt_cfg = cfg_full.get("optimization", {})
    n_trials = args.trials or int(opt_cfg.get("trials_per_strategy", {}).get(strategy, 6))

    study = make_study(cfg_full, strategy)
    finish_stale_running_trials(study, args.config, cfg_full, strategy)

    def objective(trial: optuna.Trial) -> float:
        overrides = build_trial_overrides(trial, cfg_full, strategy)
        run_name = f"optuna_trial_{trial.number:03d}"
        summary_path = Path("results") / strategy / run_name / "summary.json"

        if summary_path.exists():
            print(f"Using existing completed summary for {strategy}/{run_name}: {summary_path}")
            with summary_path.open("r", encoding="utf-8") as f:
                result = json.load(f)
        else:
            result = run_training(
                args.config,
                strategy,
                run_name,
                overrides=overrides,
                seed=tuning_seed,
                tune_mode=True,
            )

        trial.set_user_attr("run_name", run_name)
        trial.set_user_attr("overrides", overrides)
        trial.set_user_attr("best_eval_loss", result["best_eval_loss"])
        return float(result["best_eval_loss"])

    complete_before = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    remaining_trials = max(0, n_trials - complete_before)
    print(
        f"Study {study.study_name}: complete={complete_before}, "
        f"target={n_trials}, remaining={remaining_trials}"
    )

    if remaining_trials:
        study.optimize(objective, n_trials=remaining_trials, catch=(Exception,))

    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not complete_trials:
        raise SystemExit("Nė vienas trial nepavyko sėkmingai.")

    best = {
        "strategy": strategy,
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
    }

    out_path = Path("results") / "optuna" / f"best_{strategy}.json"
    save_json(best, out_path)
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
