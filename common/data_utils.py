from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np
import soundfile as sf
import torch
from pympi.Elan import Eaf
from torch.utils.data import Dataset

from .text_utils import normalize_text

RECORDING_RE = re.compile(r"^(?P<lossy>[LR])_(?P<speech>[RS])(?P<src>[ADPRST])_(?P<gender>[FM])(?P<age>[1-5])_(?P<speaker>[A-Z]{2}\d{3})_(?P<session>\d{2})$")


@dataclass
class Recording:
    recording_id: str
    dir_path: Path
    wav_path: Path
    eaf_path: Path
    meta: dict[str, Any]


class SegmentTorchDataset(Dataset):
    def __init__(self, segments, processor, sampling_rate_hz=16000):
        self.segments = segments
        self.processor = processor
        self.sampling_rate_hz = sampling_rate_hz

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        item = self.segments[idx]

        audio_path = item["audio_path"]
        text = item["text"]

        start_sec = item.get("start_sec")
        end_sec = item.get("end_sec")
        if start_sec is not None and end_sec is not None:
            info = sf.info(audio_path)
            start_frame = max(0, int(float(start_sec) * info.samplerate))
            end_frame = min(info.frames, int(float(end_sec) * info.samplerate))
            audio, sr = sf.read(audio_path, start=start_frame, stop=end_frame)
        else:
            audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)

        if sr != self.sampling_rate_hz:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate_hz)

        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate_hz,
            return_tensors="pt",
        ).input_features[0]

        labels = self.processor.tokenizer(
            text,
            add_special_tokens=True,
        ).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
            "segment_id": item.get("segment_id"),
            "recording_id": item.get("recording_id"),
        }


@dataclass
class WhisperDataCollator:
    processor: any

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )

        bos_id = self.processor.tokenizer.bos_token_id
        if bos_id is not None and labels.shape[1] > 0:
            if (labels[:, 0] == bos_id).all():
                labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def parse_recording_meta(recording_id: str) -> dict[str, Any]:
    m = RECORDING_RE.match(recording_id)
    return m.groupdict() if m else {}



def discover_recordings(data_root: str | Path, prefixes: Iterable[str] = ("L_S", "R_S")) -> list[Recording]:
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Nerastas data_root: {data_root}")

    prefixes = tuple(prefixes)
    recordings: list[Recording] = []
    for item in sorted(data_root.iterdir()):
        if not item.is_dir():
            continue
        if not item.name.startswith(prefixes):
            continue
        wav_path = item / f"{item.name}.wav"
        eaf_path = item / f"{item.name}.eaf"
        if not wav_path.exists() or not eaf_path.exists():
            wavs = sorted(item.glob("*.wav"))
            eafs = sorted(item.glob("*.eaf"))
            if not wavs or not eafs:
                continue
            wav_path = wavs[0]
            eaf_path = eafs[0]
        recordings.append(
            Recording(
                recording_id=item.name,
                dir_path=item,
                wav_path=wav_path,
                eaf_path=eaf_path,
                meta=parse_recording_meta(item.name),
            )
        )
    return recordings



def choose_speech_tiers(eaf: Eaf) -> list[str]:
    tier_names = list(eaf.get_tier_names())
    selected = []
    for tier in tier_names:
        low = tier.lower()
        if "noise" in low or "triuk" in low:
            continue
        ann = eaf.get_annotation_data_for_tier(tier)
        if ann:
            selected.append(tier)
    return selected or tier_names



def extract_segments_from_recording(
    recording: Recording,
    min_duration_sec: float,
    max_duration_sec: float,
    noise_labels: Iterable[str],
) -> list[dict[str, Any]]:
    eaf = Eaf(str(recording.eaf_path))
    segments = []
    seg_idx = 0
    for tier in choose_speech_tiers(eaf):
        for beg_ms, end_ms, text in eaf.get_annotation_data_for_tier(tier):
            duration_sec = (end_ms - beg_ms) / 1000.0
            if duration_sec < min_duration_sec or duration_sec > max_duration_sec:
                continue
            norm = normalize_text(text, noise_labels=noise_labels)
            if not norm:
                continue
            segments.append(
                {
                    "segment_id": f"{recording.recording_id}_{seg_idx:05d}",
                    "recording_id": recording.recording_id,
                    "audio_path": str(recording.wav_path),
                    "start_sec": beg_ms / 1000.0,
                    "end_sec": end_ms / 1000.0,
                    "duration_sec": duration_sec,
                    "text": norm,
                    "tier": tier,
                }
            )
            seg_idx += 1
    return segments



def merge_adjacent_segments(
    segments: list[dict[str, Any]],
    target_merged_duration_sec: float,
    max_gap_for_merge_sec: float,
    min_duration_sec: float,
    max_duration_sec: float,
) -> list[dict[str, Any]]:
    if not segments:
        return []

    segments = sorted(segments, key=lambda x: (x["recording_id"], x["start_sec"]))
    merged = []
    current = segments[0].copy()

    def finalize(item: dict[str, Any]) -> None:
        item["duration_sec"] = item["end_sec"] - item["start_sec"]
        if min_duration_sec <= item["duration_sec"] <= max_duration_sec:
            merged.append(item.copy())

    for nxt in segments[1:]:
        same_recording = nxt["recording_id"] == current["recording_id"]
        gap = nxt["start_sec"] - current["end_sec"]
        candidate_dur = nxt["end_sec"] - current["start_sec"]
        if same_recording and 0 <= gap <= max_gap_for_merge_sec and candidate_dur <= target_merged_duration_sec:
            current["text"] = f"{current['text']} {nxt['text']}".strip()
            current["end_sec"] = nxt["end_sec"]
            current["segment_id"] = current["segment_id"]
        else:
            finalize(current)
            current = nxt.copy()
    finalize(current)

    for i, item in enumerate(merged):
        item["segment_id"] = f"{item['recording_id']}_m{i:05d}"
    return merged



def split_recordings(recordings: list[Recording], train_ratio: float, valid_ratio: float, seed: int) -> dict[str, list[Recording]]:
    if not recordings:
        return {"train": [], "valid": [], "test": []}

    records = recordings[:]
    rng = random.Random(seed)
    rng.shuffle(records)

    n = len(records)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    train = records[:n_train]
    valid = records[n_train:n_train + n_valid]
    test = records[n_train + n_valid:]
    return {"train": train, "valid": valid, "test": test}



def build_segments_for_split(recordings: list[Recording], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    all_segments = []
    for rec in recordings:
        all_segments.extend(
            extract_segments_from_recording(
                rec,
                min_duration_sec=cfg["min_duration_sec"],
                max_duration_sec=cfg["max_duration_sec"],
                noise_labels=cfg.get("remove_noise_labels", []),
            )
        )
    if cfg.get("merge_segments", False):
        all_segments = merge_adjacent_segments(
            all_segments,
            target_merged_duration_sec=cfg["target_merged_duration_sec"],
            max_gap_for_merge_sec=cfg["max_gap_for_merge_sec"],
            min_duration_sec=cfg["min_duration_sec"],
            max_duration_sec=cfg["max_duration_sec"],
        )
    return all_segments



def build_dataset_splits(cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    recordings = discover_recordings(cfg["data_root"], prefixes=cfg.get("recording_prefixes", ["L_S", "R_S"]))
    split_recs = split_recordings(recordings, cfg["train_ratio"], cfg["valid_ratio"], cfg["split_seed"])
    return {key: build_segments_for_split(val, cfg) for key, val in split_recs.items()}



def compute_dataset_stats(recordings: list[Recording]) -> dict[str, Any]:
    total_seconds = 0.0
    by_lossy = {"L": 0, "R": 0}
    by_source = {}
    missing_files = 0
    for rec in recordings:
        if not rec.wav_path.exists() or not rec.eaf_path.exists():
            missing_files += 1
            continue
        try:
            info = sf.info(str(rec.wav_path))
            total_seconds += info.frames / info.samplerate
        except Exception:
            pass
        lossy = rec.meta.get("lossy")
        if lossy in by_lossy:
            by_lossy[lossy] += 1
        src = rec.meta.get("src", "?")
        by_source[src] = by_source.get(src, 0) + 1
    return {
        "recordings": len(recordings),
        "hours": total_seconds / 3600.0,
        "by_lossy": by_lossy,
        "by_source": by_source,
        "missing_files": missing_files,
    }



def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items



def get_prepared_data_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    prepared_root = Path(cfg.get("prepared_data_dir", "prepared_data/ls_rs_spontaneous_norm_80_10_10"))
    return {
        "root": prepared_root,
        "train": prepared_root / "train.jsonl",
        "valid": prepared_root / "valid.jsonl",
        "test": prepared_root / "test.jsonl",
        "summary": prepared_root / "summary.json",
    }



def load_prepared_dataset_splits(cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    paths = get_prepared_data_paths(cfg)
    for split_name in ["train", "valid", "test"]:
        if not paths[split_name].exists():
            raise FileNotFoundError(
                f"Nerastas paruoštas split failas: {paths[split_name]}. Pirmiausia paleisk prepare_dataset.py"
            )
    return {
        "train": load_jsonl(paths["train"]),
        "valid": load_jsonl(paths["valid"]),
        "test": load_jsonl(paths["test"]),
    }
