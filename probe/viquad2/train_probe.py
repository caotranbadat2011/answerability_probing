#!/usr/bin/env python3
"""Sweep every Qwen layer and L2 value, calibrate a threshold, and export the best probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "probing"))

from _device import pick_device  # noqa: E402
from common import auroc, binary_metrics, choose_threshold, stratified_partition  # noqa: E402


def load_cache(path: Path) -> tuple[dict, list[dict], np.memmap]:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    meta = [json.loads(line) for line in (path / "meta.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    activations = np.memmap(path / "acts.npy", dtype=np.float16, mode="r").reshape(
        -1, config["n_layers"], config["hidden_size"]
    )
    if len(meta) != len(activations):
        raise ValueError(f"Inconsistent cache {path}: {len(meta)} metadata vs {len(activations)} activations")
    return config, meta, activations


def fit_probe(
    features: np.ndarray,
    labels: np.ndarray,
    device: str,
    l2: float,
    class_weight: str,
    max_iter: int,
) -> dict:
    x = torch.tensor(np.asarray(features, dtype=np.float32), device=device)
    y = torch.tensor(np.asarray(labels, dtype=np.float32), device=device)
    mean = x.mean(0, keepdim=True)
    scale = x.std(0, keepdim=True).clamp(min=1e-6)
    x = (x - mean) / scale
    weight = torch.zeros(x.shape[1], device=device, requires_grad=True)
    bias = torch.zeros(1, device=device, requires_grad=True)
    positive_weight = None
    if class_weight == "balanced":
        positives = float(y.sum())
        negatives = float(len(y) - positives)
        positive_weight = torch.tensor([negatives / max(positives, 1.0)], device=device)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.LBFGS(
        [weight, bias], max_iter=max_iter, line_search_fn="strong_wolfe"
    )

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


def predict(model: dict, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    logits = ((x - model["mean"]) / model["scale"]) @ model["weight"] + model["bias"][0]
    logits = np.clip(logits, -50, 50)
    return 1.0 / (1.0 + np.exp(-logits))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--l2-values", default="0.0001,0.001,0.01")
    parser.add_argument("--selection-fraction", type=float, default=0.5)
    parser.add_argument("--target-refusal-recall", type=float, default=0.9)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="answerability-probing-viquad2")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    train_config, train_meta, train_acts = load_cache(args.train_cache)
    val_config, val_meta, val_acts = load_cache(args.validation_cache)
    for key in ("model", "n_layers", "hidden_size", "layer_semantics"):
        if train_config.get(key) != val_config.get(key):
            raise ValueError(f"Train/validation cache mismatch for {key}: {train_config.get(key)!r} != {val_config.get(key)!r}")
    train_labels = np.asarray([row["answerable"] for row in train_meta], dtype=np.int64)
    val_labels = np.asarray([row["answerable"] for row in val_meta], dtype=np.int64)
    selection_indices, calibration_indices = stratified_partition(
        val_labels, args.selection_fraction, args.seed
    )
    l2_values = [float(value) for value in args.l2_values.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)

    print(
        f"Train={len(train_labels):,}; validation selection={len(selection_indices):,}; "
        f"threshold calibration={len(calibration_indices):,}; device={device}"
    )
    sweep = []
    best = None
    for layer in range(train_config["n_layers"]):
        x_train = np.asarray(train_acts[:, layer, :])
        x_select = np.asarray(val_acts[selection_indices, layer, :])
        for l2 in l2_values:
            model = fit_probe(
                x_train,
                train_labels,
                device,
                l2,
                args.class_weight,
                args.max_iter,
            )
            probabilities = predict(model, x_select)
            score = auroc(probabilities, val_labels[selection_indices])
            record = {"layer": layer, "hidden_state_index": layer + 1, "l2": l2, "selection_auroc": score}
            sweep.append(record)
            if best is None or (score, -l2) > (best["selection_auroc"], -best["l2"]):
                best = record
            print(f"layer={layer:02d} hidden_state={layer + 1:02d} l2={l2:g} AUROC={score:.4f}", flush=True)

    assert best is not None
    best_layer = best["layer"]
    print(f"Best: block {best_layer}, hidden_states[{best_layer + 1}], l2={best['l2']:g}, AUROC={best['selection_auroc']:.4f}")

    # Refit after model selection with train + selection; calibration remains untouched.
    refit_features = np.concatenate(
        [np.asarray(train_acts[:, best_layer, :]), np.asarray(val_acts[selection_indices, best_layer, :])]
    )
    refit_labels = np.concatenate([train_labels, val_labels[selection_indices]])
    final_model = fit_probe(
        refit_features,
        refit_labels,
        device,
        best["l2"],
        args.class_weight,
        args.max_iter,
    )
    calibration_features = np.asarray(val_acts[calibration_indices, best_layer, :])
    calibration_probabilities = predict(final_model, calibration_features)
    threshold, threshold_metrics = choose_threshold(
        calibration_probabilities,
        val_labels[calibration_indices],
        args.target_refusal_recall,
    )
    calibration_auroc = auroc(calibration_probabilities, val_labels[calibration_indices])
    threshold_metrics["auroc"] = calibration_auroc
    print("Calibration metrics:")
    print(json.dumps(threshold_metrics, indent=2))

    np.savez_compressed(
        args.out / "probe.npz",
        weight=final_model["weight"],
        bias=final_model["bias"],
        mean=final_model["mean"],
        scale=final_model["scale"],
    )
    artifact_config = {
        "format_version": 1,
        "model": train_config["model"],
        "layer": best_layer,
        "hidden_state_index": best_layer + 1,
        "layer_semantics": "transformer_block_output",
        "hidden_size": train_config["hidden_size"],
        "prompt_style": train_config["prompt_style"],
        "max_len": train_config["max_len"],
        "enable_thinking": False,
        "label_mapping": {"unanswerable": 0, "answerable": 1},
        "probability": "answerable",
        "threshold": threshold,
        "decision": "answerable if probability >= threshold",
        "l2": best["l2"],
        "class_weight": args.class_weight,
        "seed": args.seed,
        "selection_auroc": best["selection_auroc"],
        "calibration_metrics": threshold_metrics,
    }
    (args.out / "config.json").write_text(
        json.dumps(artifact_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "layer_sweep.json").write_text(
        json.dumps(sweep, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.out / "calibration_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for local_index, row_index in enumerate(calibration_indices):
            handle.write(
                json.dumps(
                    {
                        "id": val_meta[row_index]["id"],
                        "question": val_meta[row_index].get("question"),
                        "answerable": int(val_labels[row_index]),
                        "prob_answerable": float(calibration_probabilities[local_index]),
                        "pred_answerable": bool(calibration_probabilities[local_index] >= threshold),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            job_type="answerability-probe-training",
            config={
                "model": train_config["model"],
                "train_rows": len(train_labels),
                "validation_rows": len(val_labels),
                "layers": train_config["n_layers"],
                "l2_values": l2_values,
                "target_refusal_recall": args.target_refusal_recall,
                "class_weight": args.class_weight,
                "seed": args.seed,
            },
        )
        table = wandb.Table(columns=["layer", "hidden_state_index", "l2", "selection_auroc"])
        for record in sweep:
            table.add_data(
                record["layer"],
                record["hidden_state_index"],
                record["l2"],
                record["selection_auroc"],
            )
        run.log({"layer_sweep": table})
        run.summary.update(
            {
                "best_layer": best_layer,
                "hidden_state_index": best_layer + 1,
                "selection_auroc": best["selection_auroc"],
                "calibration_auroc": calibration_auroc,
                "threshold": threshold,
                **{f"calibration/{key}": value for key, value in threshold_metrics.items()},
            }
        )
        artifact = wandb.Artifact(
            name="qwen3-4b-viquad2-answerability-probe",
            type="answerability-probe",
        )
        artifact.add_dir(str(args.out))
        run.log_artifact(artifact)
        run.finish()
    print(f"Exported probe artifact to {args.out}")


if __name__ == "__main__":
    main()
