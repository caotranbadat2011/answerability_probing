#!/usr/bin/env python3
"""Fit and export a TyDiQA+UDHR probe using the paper's item-fold protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

PROBING = Path(__file__).resolve().parents[1] / "probing"
sys.path.insert(0, str(PROBING))
from _device import pick_device  # noqa: E402


def load_cache(path: Path):
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    meta = [
        json.loads(line)
        for line in (path / "meta.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    activations = np.memmap(path / "acts.npy", dtype=np.float16, mode="r").reshape(
        -1, config["n_layers"], config["hidden_size"]
    )
    if len(meta) != len(activations):
        raise ValueError(
            f"Inconsistent cache {path}: {len(meta)} metadata vs "
            f"{len(activations)} activations"
        )
    return config, meta, activations


def layer_column(config: dict, layer: int) -> int:
    saved = config.get("saved_layers", list(range(config["n_layers"])))
    if layer not in saved:
        raise ValueError(f"Layer {layer} is not present in cache; available: {saved}")
    return saved.index(layer)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    positives = int(labels.sum())
    negatives = int((1 - labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def load_features(path: Path, layer: int, clean_only: bool):
    config, meta, acts = load_cache(path)
    column = layer_column(config, layer)
    features = np.asarray(acts[:, column, :], dtype=np.float32)
    labels = np.asarray([row["answerable"] for row in meta], dtype=np.int64)
    if clean_only:
        keep = np.asarray([not row.get("in_organizer_sample", False) for row in meta])
        meta = [row for row, selected in zip(meta, keep) if selected]
        features = features[keep]
        labels = labels[keep]
    return config, meta, features, labels


def check_compatible(left: dict, right: dict) -> None:
    for key in ("model", "prompt_style", "pooling", "max_len", "hidden_size"):
        if left.get(key) != right.get(key):
            raise ValueError(
                f"TyDiQA/UDHR cache mismatch for {key}: "
                f"{left.get(key)!r} != {right.get(key)!r}"
            )


def item_folds(meta: list[dict], folds: int, seed: int) -> np.ndarray:
    items = [str(row.get("item_id") or "") for row in meta]
    if any(not item for item in items):
        raise ValueError("Every UDHR row must have item_id for item-disjoint folds")
    unique = sorted(set(items))
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(unique))
    by_item = {
        unique[index]: int(position % folds)
        for position, index in enumerate(permutation)
    }
    return np.asarray([by_item[item] for item in items], dtype=np.int64)


def fit_model(features, labels, device: str, l2: float, max_iter: int) -> dict:
    x = torch.tensor(np.asarray(features, dtype=np.float32), device=device)
    y = torch.tensor(np.asarray(labels, dtype=np.float32), device=device)
    mean = x.mean(0, keepdim=True)
    scale = x.std(0, keepdim=True).clamp(min=1e-6)
    x = (x - mean) / scale
    weight = torch.zeros(x.shape[1], device=device, requires_grad=True)
    bias = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias], max_iter=max_iter, line_search_fn="strong_wolfe"
    )
    loss_function = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_function(x @ weight + bias, y) + l2 * (weight**2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        "weight": weight.detach().cpu().numpy().astype(np.float32),
        "bias": bias.detach().cpu().numpy().astype(np.float32),
        "mean": mean.detach().cpu().numpy().reshape(-1).astype(np.float32),
        "scale": scale.detach().cpu().numpy().reshape(-1).astype(np.float32),
    }


def logits(model: dict, features) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    return (
        ((x - model["mean"]) / model["scale"]) @ model["weight"]
        + model["bias"][0]
    ).astype(np.float64)


def fit_bias_f1(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return bias b for the paper rule: abstain when score + b < 0."""
    order = np.argsort(scores)
    sorted_scores = np.asarray(scores)[order]
    sorted_labels = np.asarray(labels)[order]
    negative_count = int((sorted_labels == 0).sum())
    if negative_count == 0 or negative_count == len(sorted_labels):
        raise ValueError("F1 calibration requires both answerability classes")
    hits = np.concatenate([[0], np.cumsum(sorted_labels == 0)])
    f1 = 2.0 * hits / (np.arange(len(hits)) + negative_count)
    best = np.flatnonzero(f1 >= f1.max() - 1e-12)
    cut = int(best[len(best) // 2])
    low = sorted_scores[cut - 1] if cut > 0 else sorted_scores[0] - 1.0
    high = sorted_scores[cut] if cut < len(sorted_scores) else sorted_scores[-1] + 1.0
    return float(-(low + high) / 2)


def metrics(scores: np.ndarray, labels: np.ndarray, bias: float) -> dict:
    pred_answerable = scores + bias >= 0
    answerable = labels == 1
    unanswerable = ~answerable
    answerable_correct = int((pred_answerable & answerable).sum())
    refusal_correct = int((~pred_answerable & unanswerable).sum())
    false_refusal = int((~pred_answerable & answerable).sum())
    missed_refusal = int((pred_answerable & unanswerable).sum())
    refusal_precision = refusal_correct / max(refusal_correct + false_refusal, 1)
    refusal_recall = refusal_correct / max(refusal_correct + missed_refusal, 1)
    refusal_f1 = (
        2 * refusal_precision * refusal_recall / (refusal_precision + refusal_recall)
        if refusal_precision + refusal_recall else 0.0
    )
    return {
        "rows": len(labels),
        "answerable_total": int(answerable.sum()),
        "unanswerable_total": int(unanswerable.sum()),
        "answerable_correct": answerable_correct,
        "false_refusal": false_refusal,
        "refusal_correct": refusal_correct,
        "missed_refusal": missed_refusal,
        "answerable_accuracy": answerable_correct / max(int(answerable.sum()), 1),
        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,
        "refusal_f1": refusal_f1,
        "accuracy": (answerable_correct + refusal_correct) / max(len(labels), 1),
        "auroc": auroc(scores, labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tydi-cache", type=Path, required=True)
    parser.add_argument("--udhr-cache", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--calib-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="answerability-probing-paper-data")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    if args.folds < 3:
        raise ValueError("--folds must be at least 3")
    if not 0 <= args.calib_fold < args.folds:
        raise ValueError("--calib-fold must be within the fold range")
    device = pick_device(args.device)
    tydi_config, _, tydi_x, tydi_y = load_features(
        args.tydi_cache, args.layer, clean_only=True
    )
    udhr_config, udhr_meta, udhr_x, udhr_y = load_features(
        args.udhr_cache, args.layer, clean_only=False
    )
    check_compatible(tydi_config, udhr_config)
    folds = item_folds(udhr_meta, args.folds, args.seed)

    print(
        f"TyDiQA={len(tydi_y):,}; UDHR={len(udhr_y):,}; layer={args.layer}; "
        f"l2={args.l2:g}; device={device}"
    )
    oof_scores = np.empty(len(udhr_y), dtype=np.float64)
    biases = []
    exported_model = None
    item_array = np.asarray([row["item_id"] for row in udhr_meta])
    for fold in range(args.folds):
        calibration = folds == fold
        train_x = np.concatenate([tydi_x, udhr_x[~calibration]])
        train_y = np.concatenate([tydi_y, udhr_y[~calibration]])
        model = fit_model(train_x, train_y, device, args.l2, args.max_iter)
        fold_scores = logits(model, udhr_x[calibration])
        oof_scores[calibration] = fold_scores
        fold_bias = fit_bias_f1(fold_scores, udhr_y[calibration])
        biases.append(fold_bias)
        if fold == args.calib_fold:
            exported_model = model
        print(
            f"fold={fold} items={len(set(item_array[calibration]))} "
            f"rows={int(calibration.sum()):,} bias={fold_bias:+.6f} "
            f"threshold={1 / (1 + np.exp(fold_bias)):.6f}"
        )

    assert exported_model is not None
    global_bias = float(np.mean(biases))
    threshold = float(1 / (1 + np.exp(global_bias)))
    report_metrics = metrics(oof_scores, udhr_y, global_bias)
    report = {
        "protocol": "paper TyDiQA+UDHR; item-disjoint 5-fold UDHR calibration",
        "model": tydi_config["model"],
        "layer": args.layer,
        "hidden_state_index": args.layer,
        "layer_semantics": "hidden_states index; 0 is the embedding output",
        "hidden_size": tydi_config["hidden_size"],
        "prompt_style": tydi_config["prompt_style"],
        "pooling": tydi_config["pooling"],
        "max_len": tydi_config["max_len"],
        "l2": args.l2,
        "folds": args.folds,
        "calib_fold": args.calib_fold,
        "bias_by_fold": biases,
        "global_bias": global_bias,
        "threshold_p_answerable": threshold,
        "threshold": threshold,
        "decision": "answerable if probability >= threshold",
        "metrics_oof_udhr": report_metrics,
        "tydi_rows": len(tydi_y),
        "udhr_rows": len(udhr_y),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "probe.npz", **exported_model)
    (args.out / "config.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.out / "udhr_oof_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row, fold, score in zip(udhr_meta, folds, oof_scores):
            probability = float(1 / (1 + np.exp(-np.clip(score, -50, 50))))
            handle.write(json.dumps({
                "id": row["id"], "item_id": row["item_id"],
                "question_lang": row["question_lang"],
                "context_lang": row["passage_lang"],
                "answerable": int(row["answerable"]), "fold": int(fold),
                "prob_answerable": probability,
                "pred_answerable": bool(score + global_bias >= 0),
            }, ensure_ascii=False) + "\n")
    print("Paper-style OOF metrics:")
    print(json.dumps(report_metrics, indent=2))
    print(f"Global threshold: {threshold:.9f}")
    print(f"Exported probe to {args.out}")

    if args.wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project, name=args.wandb_run_name,
            job_type="paper-data-probe-training",
            config={key: value for key, value in report.items() if key != "metrics_oof_udhr"},
        )
        run.summary.update({f"udhr_oof/{key}": value for key, value in report_metrics.items()})
        artifact = wandb.Artifact(
            name="qwen3-4b-tydiqa-udhr-answerability-probe",
            type="answerability-probe",
        )
        artifact.add_dir(str(args.out))
        run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()
