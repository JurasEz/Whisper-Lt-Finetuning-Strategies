from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from common.phoneme_utils import phonemize_texts


RUNS = [
    ("zero_shot", "zero_shot", Path("results/baseline_whisper_large_v3/zero_shot/evaluation_test")),
    ("full_ft", "final_1epoch", Path("results/full_ft/final_1epoch/evaluation_test")),
    ("selective_ft", "final_1epoch", Path("results/selective_ft/final_1epoch/evaluation_test")),
    ("lora_ft", "final_1epoch", Path("results/lora_ft/final_1epoch/evaluation_test")),
]


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        cur = [i]

        for j, cb in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            ))

        prev = cur

    return prev[-1]


def normalized_ped(ref_ph: str, pred_ph: str) -> float:
    ref_seq = str(ref_ph).split()
    pred_seq = str(pred_ph).split()

    return edit_distance(ref_seq, pred_seq) / max(1, len(ref_seq))


def load_predictions(path: Path) -> tuple[list[str], list[str], list[str], list[str]]:
    references = []
    predictions = []
    segment_ids = []
    recording_ids = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if "reference" not in reader.fieldnames or "prediction" not in reader.fieldnames:
            raise ValueError(f"Faile {path} nėra reference/prediction stulpelių. Yra: {reader.fieldnames}")

        for i, row in enumerate(reader):
            references.append(row["reference"])
            predictions.append(row["prediction"])
            segment_ids.append(row.get("segment_id", f"seg_{i}"))
            recording_ids.append(row.get("recording_id", ""))

    return references, predictions, segment_ids, recording_ids


def main() -> None:
    for strategy, run_name, eval_dir in RUNS:
        predictions_path = eval_dir / "predictions.csv"
        metrics_path = eval_dir / "metrics.json"

        if not predictions_path.exists():
            print(f"SKIP {strategy}: nerastas {predictions_path}")
            continue

        references, predictions, segment_ids, recording_ids = load_predictions(predictions_path)

        print(f"\n=== {strategy} ===")
        print(f"Segments: {len(references)}")
        print("Phonemizing references...")
        ref_ph = phonemize_texts(references, language="lt")

        print("Phonemizing predictions...")
        pred_ph = phonemize_texts(predictions, language="lt")

        peds = [
            normalized_ped(r, p)
            for r, p in zip(ref_ph, pred_ph)
        ]

        ped = float(np.mean(peds)) if peds else float("nan")

        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            metrics = {}

        metrics["ped"] = ped
        metrics["samples"] = len(references)
        metrics["strategy"] = strategy
        metrics["run_name"] = run_name

        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        ped_path = eval_dir / "ped_per_segment.csv"
        with ped_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "segment_id",
                    "recording_id",
                    "reference_phonemes",
                    "prediction_phonemes",
                    "ped",
                ]
            )
            writer.writeheader()

            for sid, rid, rph, pph, value in zip(segment_ids, recording_ids, ref_ph, pred_ph, peds):
                writer.writerow({
                    "segment_id": sid,
                    "recording_id": rid,
                    "reference_phonemes": rph,
                    "prediction_phonemes": pph,
                    "ped": value,
                })

        print(f"PED: {ped:.6f}")
        print(f"Updated: {metrics_path}")
        print(f"Saved: {ped_path}")


if __name__ == "__main__":
    main()
