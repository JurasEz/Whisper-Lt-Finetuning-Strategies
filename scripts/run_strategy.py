#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse

import torch

from common.train_pipeline import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/hyperparameters.yaml")
    parser.add_argument("--strategy", required=True, choices=["full_ft", "selective_ft", "lora_ft"])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA nepasiekiama. Treniravimą paleisk GPU mazge arba pridėk --allow-cpu testui.")

    run_training(args.config, args.strategy, args.run_name)


if __name__ == "__main__":
    main()
