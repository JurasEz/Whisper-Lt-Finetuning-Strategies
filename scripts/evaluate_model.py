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
from common.metrics import compute_all_metrics, compute_wer_cer
from common.model_utils import load_processor_from_dir, load_trained_model
from common.train_pipeline import save_predictions_csv


def load_resolved_config(strategy: str, run_name: str) -> dict:
    path = Path("results") / strategy / run_name / "resolved_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Nerastas resolved_config.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, choices=["full_ft", "selective_ft", "lora_ft"])
    parser.add_argument("--run-name", default="final_1epoch")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--full-metrics", action="store_true")
    args = parser.parse_args()

    cfg = load_resolved_config(args.strategy, args.run_name)
    model_dir = Path(args.model_dir) if args.model_dir else Path(cfg["output_root"]) / args.strategy / args.run_name / "best_model"
    results_dir = Path(cfg["results_root"]) / args.strategy / args.run_name / f"evaluation_{args.split}"
    results_dir.mkdir(parents=True, exist_ok=True)

    processor = load_processor_from_dir(model_dir, cfg)
    model = load_trained_model(model_dir, cfg)
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
        fp16=bool(torch.cuda.is_available()),
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

    metrics = compute_all_metrics(references, predictions, language="lt") if args.full_metrics else compute_wer_cer(references, predictions)
    metrics["samples"] = len(references)
    metrics["split"] = args.split
    metrics["strategy"] = args.strategy
    metrics["run_name"] = args.run_name
    save_json(metrics, results_dir / "metrics.json")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
