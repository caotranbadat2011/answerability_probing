#!/usr/bin/env python3
"""Evaluate a frozen ViQuAD2 probe and threshold on the untouched test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import auroc, binary_metrics  # noqa: E402


def load_cache(path: Path) -> tuple[dict, list[dict], np.memmap]:
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


def predict(parameters: dict, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    logits = (
        ((x - parameters["mean"]) / parameters["scale"])
        @ parameters["weight"]
        + parameters["bias"][0]
    )
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="answerability-probing-viquad2")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    probe_config = json.loads(
        (args.probe / "config.json").read_text(encoding="utf-8")
    )
    with np.load(args.probe / "probe.npz") as archive:
        parameters = {key: archive[key] for key in ("weight", "bias", "mean", "scale")}

    cache_config, meta, activations = load_cache(args.test_cache)
    for key in ("model", "hidden_size"):
        if probe_config.get(key) != cache_config.get(key):
            raise ValueError(
                f"Probe/test cache mismatch for {key}: "
                f"{probe_config.get(key)!r} != {cache_config.get(key)!r}"
            )

    layer = int(probe_config["layer"])
    if not 0 <= layer < cache_config["n_layers"]:
        raise ValueError(f"Probe layer {layer} is outside the test cache")
    labels = np.asarray([row["answerable"] for row in meta], dtype=np.int64)
    probabilities = predict(parameters, np.asarray(activations[:, layer, :]))
    threshold = float(probe_config["threshold"])
    metrics = binary_metrics(probabilities, labels, threshold)
    metrics["auroc"] = auroc(probabilities, labels)
    metrics["rows"] = len(labels)

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "split": "test",
        "probe": str(args.probe),
        "test_cache": str(args.test_cache),
        "layer": layer,
        "hidden_state_index": probe_config["hidden_state_index"],
        "threshold_source": "validation calibration; not fitted on test",
        "metrics": metrics,
    }
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    prediction_rows = []
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row, probability in zip(meta, probabilities):
            prediction = {
                "id": row["id"],
                "question": row.get("question"),
                "answerable": int(row["answerable"]),
                "prob_answerable": float(probability),
                "pred_answerable": bool(probability >= threshold),
            }
            prediction_rows.append(prediction)
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    print("Test metrics (threshold frozen from validation):")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote test report to {args.out}")

    if args.wandb:
        import wandb

        run_name = (
            f"{args.wandb_run_name}-test" if args.wandb_run_name else None
        )
        run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            job_type="answerability-probe-test",
            config={
                "probe": str(args.probe),
                "test_rows": len(labels),
                "layer": layer,
                "threshold": threshold,
            },
        )
        run.summary.update({f"test/{key}": value for key, value in metrics.items()})
        table = wandb.Table(
            columns=[
                "id",
                "question",
                "answerable",
                "prob_answerable",
                "pred_answerable",
            ]
        )
        for row in prediction_rows:
            table.add_data(
                row["id"],
                row["question"],
                row["answerable"],
                row["prob_answerable"],
                row["pred_answerable"],
            )
        run.log({"test_predictions": table})
        artifact = wandb.Artifact(
            name="qwen3-4b-viquad2-answerability-test",
            type="answerability-probe-evaluation",
        )
        artifact.add_dir(str(args.out))
        run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()
