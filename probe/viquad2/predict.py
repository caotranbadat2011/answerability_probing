#!/usr/bin/env python3
"""Smoke-test an exported answerability probe on one context/question pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "probing"))

from _device import pick_device  # noqa: E402
from _loading import load_backbone, load_tokenizer, verify_hidden_states  # noqa: E402
from _prompts import build_prompt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = json.loads((args.probe / "config.json").read_text(encoding="utf-8"))
    arrays = np.load(args.probe / "probe.npz")
    device = pick_device(args.device)
    tokenizer = load_tokenizer(config["model"], "right")
    model, state_count, hidden_size = load_backbone(config["model"], device)
    verify_hidden_states(model, tokenizer, device, state_count, hidden_size)
    _, ids, truncated = build_prompt(
        tokenizer,
        args.context,
        args.question,
        config["max_len"],
        style="bare",
        enable_thinking=False,
    )
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(inputs)
    with torch.inference_mode():
        states = model(
            input_ids=inputs,
            attention_mask=mask,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states
    feature = states[config["hidden_state_index"]][0, -1].float().cpu().numpy()
    standardised = (feature - arrays["mean"]) / arrays["scale"]
    logit = float(standardised @ arrays["weight"] + arrays["bias"][0])
    probability = float(1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50))))
    result = {
        "prob_answerable": probability,
        "threshold": config["threshold"],
        "answerable": probability >= config["threshold"],
        "layer": config["layer"],
        "hidden_state_index": config["hidden_state_index"],
        "truncated": truncated,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
