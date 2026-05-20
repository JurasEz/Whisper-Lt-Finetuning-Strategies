#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState

from common.config_utils import load_yaml, save_json
from scripts.tune_strategy import make_study


def flatten(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(f"{name}.", value))
        else:
            out[name] = value
    return out


def read_summary(strategy: str, run_name: str) -> dict[str, Any]:
    path = Path("results") / strategy / run_name / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def export_strategy(cfg_full: dict[str, Any], strategy: str) -> list[dict[str, Any]]:
    study = make_study(cfg_full, strategy)
    rows: list[dict[str, Any]] = []

    for trial in study.trials:
        run_name = trial.user_attrs.get("run_name", f"optuna_trial_{trial.number:03d}")
        summary = read_summary(strategy, run_name)
        row: dict[str, Any] = {
            "strategy": strategy,
            "trial_number": trial.number,
            "run_name": run_name,
            "state": trial.state.name,
            "value": trial.value,
        }
        row.update({f"param.{k}": v for k, v in trial.params.items()})
        row.update({f"user_attr.{k}": v for k, v in trial.user_attrs.items() if not isinstance(v, (dict, list))})

        if summary:
            row.update(flatten("summary.", summary))

        rows.append(row)

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row.keys()})
    preferred = ["strategy", "trial_number", "run_name", "state", "value"]
    columns = preferred + [col for col in columns if col not in preferred]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/hyperparameters.yaml")
    parser.add_argument("--strategies", nargs="*", default=["full_ft", "selective_ft", "lora_ft"])
    parser.add_argument("--out", default="results/optuna/trials.csv")
    args = parser.parse_args()

    cfg_full = load_yaml(args.config)
    rows: list[dict[str, Any]] = []
    for strategy in args.strategies:
        rows.extend(export_strategy(cfg_full, strategy))

    write_csv(Path(args.out), rows)

    completed = [row for row in rows if row["state"] == TrialState.COMPLETE.name]
    best_by_strategy: dict[str, dict[str, Any]] = {}
    for row in completed:
        strategy = row["strategy"]
        current = best_by_strategy.get(strategy)
        if current is None or float(row["value"]) < float(current["value"]):
            best_by_strategy[strategy] = row

    save_json(
        {
            "trial_count": len(rows),
            "completed_count": len(completed),
            "best_by_strategy": best_by_strategy,
            "csv": args.out,
        },
        Path("results") / "optuna" / "trials_export_summary.json",
    )
    print(f"Wrote {len(rows)} trials to {args.out}")


if __name__ == "__main__":
    main()
