#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


RUNS = {
    "zero_shot": ("zero_shot", Path("results/baseline_whisper_large_v3/zero_shot")),
    "baseline": ("zero_shot", Path("results/baseline_whisper_large_v3/zero_shot")),
    "baseline_whisper_large_v3": ("zero_shot", Path("results/baseline_whisper_large_v3/zero_shot")),
    "full_ft": ("final_1epoch", Path("results/full_ft/final_1epoch")),
    "selective_ft": ("final_1epoch", Path("results/selective_ft/final_1epoch")),
    "lora_ft": ("final_1epoch", Path("results/lora_ft/final_1epoch")),
}


FIELD_ORDER = [
    "strategy",
    "run_name",
    "split",
    "samples",
    "wer",
    "cer",
    "ped",
    "exact_match",
    "sem",
    "kr",
    "kr_coverage",
]


def is_missing_ped(value: object) -> bool:
    if value is None:
        return True

    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def load_ped_from_segments(path: Path) -> float | None:
    if not path.exists():
        return None

    values = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "ped" not in (reader.fieldnames or []):
            return None

        for row in reader:
            try:
                values.append(float(row["ped"]))
            except (TypeError, ValueError):
                continue

    if not values:
        return None

    return sum(values) / len(values)


def load_run_metrics(strategy: str, split: str, default_run_name: str) -> dict[str, object] | None:
    if strategy in RUNS:
        run_name, run_dir = RUNS[strategy]
    else:
        run_name = default_run_name
        run_dir = Path("results") / strategy / run_name

    eval_dir = run_dir / f"evaluation_{split}"
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"Skipping {strategy}: nerastas {metrics_path}")
        return None

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    ped_from_segments = load_ped_from_segments(eval_dir / "ped_per_segment.csv")
    if is_missing_ped(metrics.get("ped")) and ped_from_segments is not None:
        metrics["ped"] = ped_from_segments
        print(f"PED paimtas is {eval_dir / 'ped_per_segment.csv'}")

    metrics.setdefault("strategy", strategy)
    metrics.setdefault("run_name", run_name)
    metrics.setdefault("split", split)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="*", default=["zero_shot", "full_ft", "selective_ft", "lora_ft"])
    parser.add_argument("--run-name", default="final_1epoch")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    args = parser.parse_args()

    rows = []
    for strategy in args.strategies:
        metrics = load_run_metrics(strategy, args.split, args.run_name)
        if metrics is not None:
            rows.append(metrics)

    out_dir = Path("results") / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"final_{args.split}_comparison.json"
    out_csv = out_dir / f"final_{args.split}_comparison.csv"

    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    all_fields = {key for row in rows for key in row.keys()}
    fieldnames = FIELD_ORDER + sorted(all_fields - set(FIELD_ORDER))
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
