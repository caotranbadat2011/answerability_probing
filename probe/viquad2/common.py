"""Shared utilities for the ViQuAD2 answerability probe workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_partition(labels: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return item-disjoint-ish indices for model selection and threshold calibration.

    ViQuAD2 IDs are unique question IDs, so splitting indices is sufficient here. Each
    answerability class is shuffled separately to preserve the class ratio.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    left, right = [], []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        cut = max(1, min(len(indices) - 1, round(len(indices) * fraction)))
        left.extend(indices[:cut].tolist())
        right.extend(indices[cut:].tolist())
    return np.asarray(sorted(left)), np.asarray(sorted(right))


def binary_metrics(prob_answerable: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    pred_answerable = prob_answerable >= threshold
    answerable = labels == 1
    unanswerable = ~answerable
    tp_a = int((pred_answerable & answerable).sum())
    fn_a = int((~pred_answerable & answerable).sum())
    tp_u = int((~pred_answerable & unanswerable).sum())
    fn_u = int((pred_answerable & unanswerable).sum())
    answerable_accuracy = tp_a / max(int(answerable.sum()), 1)
    refusal_recall = tp_u / max(int(unanswerable.sum()), 1)
    predicted_refusal = int((~pred_answerable).sum())
    refusal_precision = tp_u / max(predicted_refusal, 1)
    refusal_f1 = (
        2 * refusal_precision * refusal_recall / (refusal_precision + refusal_recall)
        if refusal_precision + refusal_recall else 0.0
    )
    return {
        "threshold": float(threshold),
        "answerable_total": int(answerable.sum()),
        "unanswerable_total": int(unanswerable.sum()),
        "answerable_correct": tp_a,
        "answerable_missed": fn_a,
        "refusal_correct": tp_u,
        "refusal_missed": fn_u,
        "answerable_accuracy": answerable_accuracy,
        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,
        "refusal_f1": refusal_f1,
        "balanced_accuracy": (answerable_accuracy + refusal_recall) / 2,
        "accuracy": (tp_a + tp_u) / max(len(labels), 1),
    }


def choose_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    target_refusal_recall: float | None = None,
    min_answerable_accuracy: float | None = None,
) -> tuple[float, dict]:
    if target_refusal_recall is not None and min_answerable_accuracy is not None:
        raise ValueError(
            "Choose either target_refusal_recall or min_answerable_accuracy, not both"
        )
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0 + 1e-7])))
    scored = [binary_metrics(probabilities, labels, float(t)) for t in candidates]
    feasible = scored
    if target_refusal_recall is not None:
        feasible = [m for m in scored if m["refusal_recall"] >= target_refusal_recall]
    elif min_answerable_accuracy is not None:
        feasible = [
            m for m in scored
            if m["answerable_accuracy"] >= min_answerable_accuracy
        ]
    if not feasible:
        constraint = (
            f"refusal_recall >= {target_refusal_recall}"
            if target_refusal_recall is not None
            else f"answerable_accuracy >= {min_answerable_accuracy}"
        )
        raise ValueError(f"No threshold satisfies {constraint}")
    if min_answerable_accuracy is not None:
        best = max(
            feasible,
            key=lambda m: (
                m["refusal_recall"],
                m["balanced_accuracy"],
                m["refusal_precision"],
                m["threshold"],
            ),
        )
    else:
        best = max(
            feasible,
            key=lambda m: (
                m["balanced_accuracy"],
                m["refusal_f1"],
                m["answerable_accuracy"],
                -m["threshold"],
            ),
        )
    return best["threshold"], best


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
