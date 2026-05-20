#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Subset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from common.config_utils import save_json
from common.data_utils import SegmentTorchDataset, WhisperDataCollator, load_prepared_dataset_splits
from common.metrics import (
    SemanticScorer,
    compute_keyword_recall,
    compute_ped,
    compute_wer_cer,
    exact_match_rate,
)
from common.model_utils import load_model, load_processor
from common.train_pipeline import save_predictions_csv


def load_base_config() -> dict:
    candidates = [
        Path("results/full_ft/final_1epoch/resolved_config.json"),
        Path("results/selective_ft/final_1epoch/resolved_config.json"),
        Path("results/lora_ft/final_1epoch/resolved_config.json"),
    ]

    for path in candidates:
        if path.exists():
            cfg = json.loads(path.read_text(encoding="utf-8"))
            cfg["strategy_name"] = "baseline_whisper_large_v3"
            cfg["run_name"] = "zero_shot"
            cfg["model_weight_quantization"] = None
            return cfg

    raise FileNotFoundError("Neradau jokio final_1epoch/resolved_config.json failo.")


def compute_metrics_safe(references: list[str], predictions: list[str], full_metrics: bool) -> dict[str, float]:
    metrics = compute_wer_cer(references, predictions)

    if not full_metrics:
        return metrics

    metrics["exact_match"] = exact_match_rate(references, predictions)

    sem_scorer = SemanticScorer()
    metrics["sem"] = sem_scorer.score(references, predictions)

    kr, kr_coverage = compute_keyword_recall(references, predictions)
    metrics["kr"] = kr
    metrics["kr_coverage"] = kr_coverage

    try:
        metrics["ped"] = compute_ped(references, predictions, language="lt")
    except Exception as e:
        print(f"WARNING: PED nepavyko apskaiciuoti: {e}")
        metrics["ped"] = float("nan")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="zero_shot")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--full-metrics", action="store_true")
    args = parser.parse_args()

    cfg = load_base_config()
    cfg["run_name"] = args.run_name

    results_dir = Path("results") / "baseline_whisper_large_v3" / args.run_name / f"evaluation_{args.split}"
    results_dir.mkdir(parents=True, exist_ok=True)

    processor = load_processor(cfg)
    model = load_model(cfg)

    if torch.cuda.is_available():
        model = model.to("cuda")

    model.eval()

    splits = load_prepared_dataset_splits(cfg)
    dataset = SegmentTorchDataset(splits[args.split], processor, sampling_rate_hz=cfg["sampling_rate_hz"])

    if args.max_samples > 0:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))

    data_collator = WhisperDataCollator(processor)

    eval_args = Seq2SeqTrainingArguments(
        output_dir=str(results_dir / "tmp_trainer"),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 2)),
        predict_with_generate=True,
        generation_max_length=int(cfg["max_output_length"]),
        fp16=False,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    raw = trainer.predict(dataset)

    pred_ids = raw.predictions
    label_ids = raw.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    predictions = [t.strip() for t in processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
    references = [t.strip() for t in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    base_segments = dataset.dataset.segments if isinstance(dataset, Subset) else dataset.segments
    indices = list(dataset.indices) if isinstance(dataset, Subset) else list(range(len(base_segments)))

    segment_ids = [base_segments[i].get("segment_id", f"seg_{i}") for i in indices]
    recording_ids = [base_segments[i].get("recording_id", f"rec_{i}") for i in indices]

    save_predictions_csv(results_dir / "predictions.csv", references, predictions, segment_ids, recording_ids)

    metrics = compute_metrics_safe(references, predictions, full_metrics=args.full_metrics)
    metrics["samples"] = len(references)
    metrics["split"] = args.split
    metrics["strategy"] = "baseline_whisper_large_v3"
    metrics["run_name"] = args.run_name

    save_json(metrics, results_dir / "metrics.json")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
