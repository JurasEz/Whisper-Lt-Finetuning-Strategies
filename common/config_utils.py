from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def save_json(data: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return cfg
    out = deepcopy(cfg)
    for key, value in overrides.items():
        out[key] = deepcopy(value)
    return out



def build_run_config(
    cfg: dict[str, Any],
    strategy_name: str,
    run_name: str | None = None,
    overrides: dict[str, Any] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    final_cfg: dict[str, Any] = {}
    final_cfg.update(deepcopy(cfg["experiment"]))
    final_cfg.update(deepcopy(cfg.get("splits", {})))
    final_cfg.update(deepcopy(cfg.get("segment_processing", {})))
    final_cfg.update(deepcopy(cfg.get("common_hyperparameters", {})))
    final_cfg["metrics"] = deepcopy(cfg.get("metrics", {}))
    final_cfg["repeat_policy"] = deepcopy(cfg.get("repeat_policy", {}))
    final_cfg["common_search_space"] = deepcopy(cfg.get("common_search_space", {}))
    final_cfg["optimization"] = deepcopy(cfg.get("optimization", {}))

    strategy_cfg = cfg["strategies"][strategy_name]
    final_cfg["strategy_name"] = strategy_name
    final_cfg["strategy_description"] = strategy_cfg.get("description", "")
    final_cfg.update(deepcopy(strategy_cfg.get("fixed", {})))
    final_cfg.update(deepcopy(strategy_cfg.get("defaults", {})))

    if run_name is not None:
        for run_cfg in strategy_cfg.get("pilot_3_runs", []):
            if run_cfg["run_name"] == run_name:
                final_cfg.update(deepcopy(run_cfg))
                break

    final_cfg["search_space"] = deepcopy(strategy_cfg.get("search_space", {}))
    final_cfg.setdefault("run_name", run_name or f"{strategy_name}_default")

    if seed is not None:
        final_cfg["seed"] = int(seed)
    else:
        final_cfg["seed"] = int(cfg.get("repeat_policy", {}).get("tuning_seed", 42))

    final_cfg = apply_overrides(final_cfg, overrides)
    return final_cfg
