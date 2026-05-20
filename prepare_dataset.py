from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common.config_utils import load_yaml, save_json
from common.data_utils import build_dataset_splits, discover_recordings, get_prepared_data_paths


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prepare_config(cfg_full: dict[str, Any]) -> dict[str, Any]:
    exp = cfg_full["experiment"]
    splits = cfg_full["splits"]
    segment = cfg_full["segment_processing"]
    return {
        "data_root": exp["data_root"],
        "recording_prefixes": exp.get("recording_prefixes", ["L_S", "R_S"]),
        "prepared_data_dir": exp.get("prepared_data_dir", "prepared_data/default"),
        "train_ratio": float(splits.get("train_ratio", 0.8)),
        "valid_ratio": float(splits.get("valid_ratio", 0.1)),
        "split_seed": int(splits.get("split_seed", 42)),
        **segment,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/hyperparameters.yaml")
    parser.add_argument("--force", action="store_true", help="Perrašyti jau paruoštus train/valid/test failus.")
    args = parser.parse_args()

    cfg_full = load_yaml(args.config)
    cfg = build_prepare_config(cfg_full)
    paths = get_prepared_data_paths(cfg)
    required = [paths["train"], paths["valid"], paths["test"]]
    if all(path.exists() for path in required) and not args.force:
        summary = {
            "status": "exists",
            "message": "Paruošti duomenų failai jau egzistuoja. Naudok --force, jei nori perrašyti.",
            "prepared_data_dir": str(paths["root"]),
            "files": {name: str(paths[name]) for name in ["train", "valid", "test", "summary"]},
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    splits = build_dataset_splits(cfg)
    for split_name in ["train", "valid", "test"]:
        write_jsonl(paths[split_name], splits[split_name])

    recordings = discover_recordings(cfg["data_root"], prefixes=cfg["recording_prefixes"])
    summary = {
        "data_root": cfg["data_root"],
        "prepared_data_dir": str(paths["root"]),
        "recordings": len(recordings),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "segment_processing": cfg_full.get("segment_processing", {}),
    }
    save_json(summary, paths["summary"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
