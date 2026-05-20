from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset
from transformers import EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from .config_utils import load_yaml, save_json
from .data_utils import SegmentTorchDataset, WhisperDataCollator, load_prepared_dataset_splits
from .model_utils import configure_model_for_strategy, count_trainable_parameters, load_model, load_processor


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_prepared_paths(cfg_full: dict[str, Any]) -> dict[str, Path]:
    exp_cfg = cfg_full.get("experiment", {})
    prepared_root = Path(exp_cfg.get("prepared_data_dir", "prepared_data/ls_rs_spontaneous_norm_80_10_10"))
    return {
        "prepared_root": prepared_root,
        "train": prepared_root / "train.jsonl",
        "valid": prepared_root / "valid.jsonl",
        "test": prepared_root / "test.jsonl",
    }

def _build_run_config(cfg_full: dict[str, Any], strategy_name: str, run_name: str, overrides: dict[str, Any] | None = None, seed: int | None = None) -> dict[str, Any]:
    common = cfg_full.get("common_hyperparameters", {})
    strategy_cfg = cfg_full["strategies"][strategy_name]
    fixed = strategy_cfg.get("fixed", {})
    defaults = strategy_cfg.get("defaults", {})
    exp_cfg = cfg_full.get("experiment", {})

    cfg: dict[str, Any] = {}
    cfg["strategy_name"] = strategy_name
    cfg["run_name"] = run_name
    cfg["model_name"] = exp_cfg.get("model_name", "openai/whisper-large-v3")
    cfg["language"] = exp_cfg.get("language", "lithuanian")
    cfg["task"] = exp_cfg.get("task", "transcribe")
    cfg["sampling_rate_hz"] = int(exp_cfg.get("sampling_rate_hz", 16000))

    cfg["output_root"] = exp_cfg.get("output_root", "outputs")
    cfg["results_root"] = exp_cfg.get("results_root", "results")

    prepared = _resolve_prepared_paths(cfg_full)
    cfg["prepared_root"] = str(prepared["prepared_root"])
    cfg["prepared_data_dir"] = str(prepared["prepared_root"])
    cfg["train_jsonl"] = str(prepared["train"])
    cfg["valid_jsonl"] = str(prepared["valid"])
    cfg["test_jsonl"] = str(prepared["test"])

    cfg["optimizer"] = common.get("optimizer", "paged_adamw_8bit")
    cfg["scheduler"] = common.get("scheduler", "linear")
    cfg["early_stopping_patience_evals"] = int(common.get("early_stopping_patience_evals", 3))
    cfg["per_device_train_batch_size"] = int(common.get("per_device_train_batch_size", 2))
    cfg["per_device_eval_batch_size"] = int(common.get("per_device_eval_batch_size", 2))
    cfg["gradient_accumulation_steps"] = int(common.get("gradient_accumulation_steps", 8))
    cfg["gradient_checkpointing"] = bool(common.get("gradient_checkpointing", True))
    cfg["mixed_precision"] = common.get("mixed_precision", "fp16_if_cuda")
    cfg["max_output_length"] = int(common.get("max_output_length", 225))
    cfg["eval_steps"] = int(common.get("eval_steps", 200))
    cfg["save_steps"] = int(common.get("save_steps", 200))
    cfg["logging_steps"] = int(common.get("logging_steps", 25))
    cfg["save_total_limit"] = int(common.get("save_total_limit", 2))
    cfg["best_model_metric"] = common.get("best_model_metric", "loss")
    cfg["greater_is_better"] = bool(common.get("greater_is_better", False))
    cfg["max_steps"] = int(common.get("max_steps", -1))
    cfg["tune_train_subset_size"] = int(common.get("tune_train_subset_size", 0))
    cfg["tune_valid_subset_size"] = int(common.get("tune_valid_subset_size", 0))
    cfg["tune_compute_test_metrics"] = bool(common.get("tune_compute_test_metrics", False))
    cfg["auto_resume"] = bool(common.get("auto_resume", False))

    cfg["max_epochs"] = defaults.get("max_epochs", common.get("max_epochs", common.get("max_epochs_default", 5)))
    cfg["warmup_ratio"] = defaults.get("warmup_ratio", common.get("warmup_ratio", common.get("warmup_ratio_default", 0.05)))
    cfg["weight_decay"] = defaults.get("weight_decay", common.get("weight_decay", common.get("weight_decay_default", 0.0)))
    cfg["learning_rate"] = defaults.get("learning_rate", 1e-4)

    for k, v in fixed.items():
        cfg[k] = v

    for k, v in defaults.items():
        cfg[k] = v

    if overrides:
        for k, v in overrides.items():
            cfg[k] = v

    if seed is not None:
        cfg["seed"] = int(seed)
    else:
        cfg["seed"] = int(cfg_full.get("repeat_policy", {}).get("tuning_seed", 42))

    return cfg


def make_output_dirs(cfg: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = Path(cfg["output_root"]) / cfg["strategy_name"] / cfg["run_name"]
    results_dir = Path(cfg["results_root"]) / cfg["strategy_name"] / cfg["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, results_dir


def save_predictions_csv(path: Path, references: list[str], predictions: list[str], segment_ids: list[str], recording_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["segment_id", "recording_id", "reference", "prediction"])
        writer.writeheader()
        for sid, rid, ref, pred in zip(segment_ids, recording_ids, references, predictions):
            writer.writerow({
                "segment_id": sid,
                "recording_id": rid,
                "reference": ref,
                "prediction": pred,
            })

def run_training(config_path: str, strategy_name: str, run_name: str, overrides: dict[str, Any] | None = None, seed: int | None = None, tune_mode: bool = False) -> dict[str, Any]:
    cfg_full = load_yaml(config_path)
    cfg = _build_run_config(cfg_full, strategy_name=strategy_name, run_name=run_name, overrides=overrides, seed=seed)

    output_dir, results_dir = make_output_dirs(cfg)
    save_json(cfg, results_dir / "resolved_config.json")

    processor = load_processor(cfg)
    model = load_model(cfg)
    model = configure_model_for_strategy(model, cfg)

    if hasattr(model, "config"):
        try:
            model.config.forced_decoder_ids = None
            model.config.suppress_tokens = []
            model.config.use_cache = False if cfg.get("gradient_checkpointing", True) else model.config.use_cache
        except Exception:
            pass

    param_stats = count_trainable_parameters(model)
    save_json(param_stats, results_dir / "parameter_counts.json")

    train_segments = _load_jsonl(Path(cfg["train_jsonl"]))
    valid_segments = _load_jsonl(Path(cfg["valid_jsonl"]))
    test_segments = _load_jsonl(Path(cfg["test_jsonl"]))

    splits = load_prepared_dataset_splits(cfg)
    train_dataset = SegmentTorchDataset(splits["train"], processor, sampling_rate_hz=cfg["sampling_rate_hz"])
    valid_dataset = SegmentTorchDataset(splits["valid"], processor, sampling_rate_hz=cfg["sampling_rate_hz"])
    test_dataset = SegmentTorchDataset(splits["test"], processor, sampling_rate_hz=cfg["sampling_rate_hz"])

    if tune_mode:
        train_subset_size = int(cfg.get("tune_train_subset_size", 0))
        valid_subset_size = int(cfg.get("tune_valid_subset_size", 0))
        if train_subset_size > 0:
            train_dataset = Subset(train_dataset, range(min(train_subset_size, len(train_dataset))))
        if valid_subset_size > 0:
            valid_dataset = Subset(valid_dataset, range(min(valid_subset_size, len(valid_dataset))))

    save_json(
        {
            "train_segments": len(train_segments),
            "valid_segments": len(valid_segments),
            "test_segments": len(test_segments),
        },
        results_dir / "split_sizes.json",
    )

    data_collator = WhisperDataCollator(processor)

    preview = data_collator([train_dataset[0], train_dataset[1]])
    bad_keys = {"input_ids", "decoder_input_ids", "decoder_attention_mask"} & set(preview.keys())
    if bad_keys:
        raise ValueError(f"Neteisingi batch raktai Whisper modeliui: {sorted(bad_keys)}")
    print("BATCH KEYS:", list(preview.keys()))

    use_fp16 = bool(torch.cuda.is_available() and cfg.get("mixed_precision") == "fp16_if_cuda")
    optim_name = cfg.get("optimizer", "adamw_torch")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=float(cfg["max_epochs"]),
        max_steps=int(cfg.get("max_steps", -1)),
        warmup_ratio=float(cfg["warmup_ratio"]),
        lr_scheduler_type=str(cfg.get("scheduler", "linear")),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        eval_steps=int(cfg["eval_steps"]),
        save_steps=int(cfg["save_steps"]),
        logging_steps=int(cfg["logging_steps"]),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        predict_with_generate=False,
        generation_max_length=int(cfg["max_output_length"]),
        load_best_model_at_end=True,
        metric_for_best_model=cfg["best_model_metric"],
        greater_is_better=bool(cfg["greater_is_better"]),
        fp16=use_fp16,
        gradient_checkpointing=bool(cfg["gradient_checkpointing"]),
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        dataloader_num_workers=0,
        optim=optim_name,
        seed=int(cfg["seed"]),
    )

    print("BATCH KEYS:", preview.keys())

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
        compute_metrics=None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(cfg["early_stopping_patience_evals"]))],
    )

    resume_checkpoint = None
    if cfg.get("auto_resume", True):
        resume_checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
        if resume_checkpoint:
            print(f"Resuming training from checkpoint: {resume_checkpoint}")

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    train_seconds = time.time() - t0
    save_json({"training_seconds": train_seconds}, results_dir / "timing.json")

    model_dir = output_dir / "best_model"
    trainer.save_model(str(model_dir))
    processor.save_pretrained(str(model_dir))

    best_eval_loss = trainer.state.best_metric
    if best_eval_loss is None:
        eval_result = trainer.evaluate(valid_dataset)
        best_eval_loss = float(eval_result.get("eval_loss", 999.0))

    summary = {
        "best_eval_loss": float(best_eval_loss),
        "model_dir": str(model_dir),
        "run_name": run_name,
        "strategy_name": strategy_name,
        "tune_mode": bool(tune_mode),
    }
    save_json(summary, results_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--strategy", required=True, choices=["full_ft", "selective_ft", "lora_ft"])
    parser.add_argument("--run_name", required=True)
    args = parser.parse_args()
    run_training(args.config, args.strategy, args.run_name)


if __name__ == "__main__":
    main()
