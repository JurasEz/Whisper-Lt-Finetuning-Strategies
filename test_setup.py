from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from common.config_utils import load_yaml
from common.data_utils import compute_dataset_stats, discover_recordings, get_prepared_data_paths, load_prepared_dataset_splits


REQUIRED_MODULES = [
    "torch",
    "transformers",
    "peft",
    "librosa",
    "soundfile",
    "jiwer",
    "sentence_transformers",
    "phonemizer",
    "pympi",
    "yaml",
    "optuna",
]


def check_imports() -> dict[str, str]:
    status = {}
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
            status[name] = "ok"
        except Exception as exc:
            status[name] = f"error: {exc}"
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/hyperparameters.yaml")
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()

    cfg_full = load_yaml(args.config)
    exp = cfg_full["experiment"]
    prepared_cfg = {"prepared_data_dir": exp.get("prepared_data_dir", "prepared_data/default")}
    prepared_paths = get_prepared_data_paths(prepared_cfg)

    import torch

    report = {
        "imports": check_imports(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "prepared_files": {name: str(path) for name, path in prepared_paths.items()},
        "prepared_exists": {
            name: path.exists()
            for name, path in prepared_paths.items()
            if name in {"train", "valid", "test", "summary"}
        },
    }

    if args.deep:
        recordings = discover_recordings(
            exp["data_root"],
            prefixes=exp.get("recording_prefixes", ["L_S", "R_S"]),
        )
        report["recordings_found"] = len(recordings)
        report["recording_prefixes"] = exp.get("recording_prefixes", ["L_S", "R_S"])
        report["recording_stats"] = compute_dataset_stats(recordings)

    if all(prepared_paths[name].exists() for name in ["train", "valid", "test"]):
        splits = load_prepared_dataset_splits(prepared_cfg)
        report["prepared_split_sizes"] = {name: len(rows) for name, rows in splits.items()}

    print(json.dumps(report, ensure_ascii=False, indent=2))

    failed = {name: value for name, value in report["imports"].items() if value != "ok"}
    if failed:
        raise SystemExit(f"Trūksta arba neveikia bibliotekos: {failed}")
    if not report["cuda_available"]:
        raise SystemExit("CUDA nepasiekiama. GPU job'o treniravimo nepradedu.")
    missing = [name for name in ["train", "valid", "test"] if not prepared_paths[name].exists()]
    if missing:
        raise SystemExit(f"Trūksta paruoštų duomenų failų: {missing}. Paleisk prepare_dataset.py.")


if __name__ == "__main__":
    main()
