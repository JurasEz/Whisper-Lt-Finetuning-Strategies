from __future__ import annotations

import math
from collections import Counter

import jiwer
import numpy as np

from .phoneme_utils import normalized_ped
from .text_utils import extract_keywords


class SemanticScorer:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def score(self, references: list[str], predictions: list[str]) -> float:
        emb_ref = self.model.encode(references, normalize_embeddings=True, convert_to_numpy=True)
        emb_pred = self.model.encode(predictions, normalize_embeddings=True, convert_to_numpy=True)
        sims = np.sum(emb_ref * emb_pred, axis=1)
        return float(np.mean(sims)) if len(sims) else float("nan")



def compute_wer_cer(references: list[str], predictions: list[str]) -> dict[str, float]:
    wer = jiwer.wer(references, predictions)
    cer = jiwer.cer(references, predictions)
    return {"wer": float(wer), "cer": float(cer)}



def compute_keyword_recall(references: list[str], predictions: list[str]) -> tuple[float, float]:
    recalls = []
    covered = 0
    for ref, pred in zip(references, predictions):
        kws = extract_keywords(ref)
        if not kws:
            continue
        covered += 1
        pred_tokens = set(pred.split())
        hits = sum(1 for kw in kws if kw in pred_tokens)
        recalls.append(hits / len(kws))
    if not recalls:
        return float("nan"), 0.0
    return float(np.mean(recalls)), covered / max(1, len(references))



def compute_ped(references: list[str], predictions: list[str], language: str = "lt") -> float:
    peds = [normalized_ped(r, p, language=language) for r, p in zip(references, predictions)]
    return float(np.mean(peds)) if peds else float("nan")



def exact_match_rate(references: list[str], predictions: list[str]) -> float:
    if not references:
        return 0.0
    hits = sum(1 for r, p in zip(references, predictions) if r == p)
    return hits / len(references)



def compute_all_metrics(references: list[str], predictions: list[str], language: str = "lt") -> dict[str, float]:
    metrics = compute_wer_cer(references, predictions)
    metrics["exact_match"] = exact_match_rate(references, predictions)

    sem_scorer = SemanticScorer()
    metrics["sem"] = sem_scorer.score(references, predictions)

    kr, kr_coverage = compute_keyword_recall(references, predictions)
    metrics["kr"] = kr
    metrics["kr_coverage"] = kr_coverage

    try:
        metrics["ped"] = compute_ped(references, predictions, language=language)
    except Exception as e:
        print(f"WARNING: PED nepavyko apskaiciuoti: {e}")
        metrics["ped"] = float("nan")
    return metrics
