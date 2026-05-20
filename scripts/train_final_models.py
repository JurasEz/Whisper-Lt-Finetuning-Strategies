#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.config_utils import load_yaml, save_json
from common.train_pipeline import run_training


def load_best_params(strategy: str) -> dict:
    best_path = Path("results") / "optuna" / f"quick_best_{strategy}.json"
    if not best_path.exists():
        raise FileNotFoundError(f"Nerastas HPO rezultatu failas: {best_path}")
    best = json.loads(best_path.read_text(encoding="utf-8"))
    params = dict(best["best_params"])
    for key in [
        "max_steps",
        "eval_steps",
        "save_steps",
        "tune_train_subset_size",
        "tune_valid_subset_size",
        "tune_compute_test_metrics",
    ]:
        params.pop(key, None)
    return params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/hyperparameters.yaml")
    parser.add_argument("--strategies", nargs="*", default=["full_ft", "selective_ft", "lora_ft"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA nepasiekiama. Finalini mokyma paleisk GPU mazge arba pridek --allow-cpu testui.")

    cfg = load_yaml(args.config)
    manifest = {
        "config": args.config,
        "model_name": cfg.get("experiment", {}).get("model_name"),
        "seed": args.seed,
        "final_epochs": 1,
        "models": {},
    }

    for strategy in args.strategies:
        params = load_best_params(strategy)
        params.update(
            {
                "max_epochs": 1,
                "max_steps": -1,
                "tune_train_subset_size": 0,
                "tune_valid_subset_size": 0,
                "tune_compute_test_metrics": False,
                "auto_resume": False,
            }
        )
        run_name = "final_1epoch"
        print(f"\n=== FINAL TRAIN {strategy} ===\n")
        summary = run_training(
            args.config,
            strategy,
            run_name,
            overrides=params,
            seed=args.seed,
            tune_mode=False,
        )
        manifest["models"][strategy] = {
            "run_name": run_name,
            "model_dir": summary["model_dir"],
            "best_eval_loss": summary["best_eval_loss"],
            "hyperparameters": params,
        }

    out_path = Path("results") / "final_models.json"
    save_json(manifest, out_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
