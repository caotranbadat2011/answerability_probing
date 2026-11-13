#!/usr/bin/env python3
"""Extract every Transformer-block final-token activation for ViQuAD2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROBING = HERE.parent / "probing"
sys.path.insert(0, str(PROBING))

from _device import pick_device  # noqa: E402
from _loading import load_backbone, load_tokenizer, verify_hidden_states  # noqa: E402
from _prompts import build_prompt  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"id", "context", "question", "answerable"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise KeyError(f"row {index} in {path} is missing {sorted(missing)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0, help="0 keeps all rows")
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()

    device = pick_device(args.device, require_gpu=args.require_gpu)
    rows = read_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No input rows")
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.model, "right")
    print(f"Loading {args.model} on {device} ({args.dtype}) ...", flush=True)
    model, hidden_state_count, hidden_size = load_backbone(args.model, device, args.dtype)
    verify_hidden_states(model, tokenizer, device, hidden_state_count, hidden_size)
    transformer_layers = hidden_state_count - 1
    print(
        f"Verified {transformer_layers} Transformer layers, {hidden_size} hidden dimensions; "
        "hidden_states[0] (embedding) will not be cached.",
        flush=True,
    )

    config_path = args.out / "config.json"
    acts_path = args.out / "acts.npy"
    meta_path = args.out / "meta.jsonl"
    config = {
        "model": args.model,
        "input": str(args.input),
        "rows": len(rows),
        "n_layers": transformer_layers,
        "saved_layers": list(range(transformer_layers)),
        "hidden_size": hidden_size,
        "max_len": args.max_len,
        "pooling": "last",
        "prompt_style": "bare",
        "enable_thinking": False,
        "dtype_on_disk": "float16",
        "layer_semantics": "transformer_block_output",
        "hidden_state_index": "block_index + 1",
    }
    existing = [path.exists() for path in (config_path, acts_path, meta_path)]
    completed_ids: set[str] = set()
    row_bytes = transformer_layers * hidden_size * np.dtype(np.float16).itemsize
    if any(existing):
        if not all(existing):
            raise SystemExit(
                f"Partial cache metadata in {args.out}; config.json, acts.npy and meta.jsonl "
                "must either all exist or all be absent"
            )
        old = json.loads(config_path.read_text(encoding="utf-8"))
        for key in (
            "model",
            "n_layers",
            "hidden_size",
            "max_len",
            "prompt_style",
            "layer_semantics",
        ):
            if old.get(key) != config.get(key):
                raise SystemExit(
                    f"Cannot resume {args.out}: {key} was {old.get(key)!r}, now {config.get(key)!r}"
                )
        existing_meta = [
            json.loads(line)
            for line in meta_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        activation_rows, remainder = divmod(acts_path.stat().st_size, row_bytes)
        if remainder or activation_rows != len(existing_meta):
            raise SystemExit(
                f"Inconsistent partial cache: {activation_rows} activation rows, "
                f"{len(existing_meta)} metadata rows, remainder={remainder} bytes"
            )
        completed_ids = {row["id"] for row in existing_meta}
        # A pilot cache may later be resumed against the full JSONL. Keep the
        # manifest truthful while preserving all shape-critical settings.
        old["rows"] = len(completed_ids) + sum(row["id"] not in completed_ids for row in rows)
        config_path.write_text(json.dumps(old, indent=2), encoding="utf-8")
        print(f"Resuming {args.out}: {len(completed_ids):,} rows already complete", flush=True)
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows = [row for row in rows if row["id"] not in completed_ids]
    if not rows:
        print("Nothing to do; activation cache is complete.")
        return

    print(f"Tokenising {len(rows):,} prompts ...", flush=True)
    encoded = []
    truncated = 0
    for row in rows:
        _, ids, was_truncated = build_prompt(
            tokenizer,
            row["context"],
            row["question"],
            args.max_len,
            style="bare",
            enable_thinking=False,
        )
        encoded.append((ids, row))
        truncated += int(was_truncated)
    encoded.sort(key=lambda pair: len(pair[0]))

    batches, current = [], []
    for ids, row in encoded:
        width = max([len(item[0]) for item in current] + [len(ids)])
        if current and (
            (len(current) + 1) * width > args.token_budget or len(current) >= args.batch_size
        ):
            batches.append(current)
            current = []
        current.append((ids, row))
    if current:
        batches.append(current)
    print(f"{len(batches):,} batches; {truncated:,} prompts had passage truncation", flush=True)

    with acts_path.open("ab") as acts_handle, meta_path.open("a", encoding="utf-8") as meta_handle:
        with torch.inference_mode():
            completed = 0
            for batch_index, pairs in enumerate(batches, start=1):
                width = max(len(ids) for ids, _ in pairs)
                input_ids = torch.full(
                    (len(pairs), width), tokenizer.pad_token_id, dtype=torch.long, device=device
                )
                attention_mask = torch.zeros((len(pairs), width), dtype=torch.long, device=device)
                for index, (ids, _) in enumerate(pairs):
                    length = len(ids)
                    input_ids[index, :length] = torch.tensor(ids, dtype=torch.long, device=device)
                    attention_mask[index, :length] = 1
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                states = outputs.hidden_states
                last_positions = attention_mask.sum(dim=1) - 1
                row_indices = torch.arange(len(pairs), device=device)
                # states[0] is embedding output; cache block outputs 0..N-1 from states[1:].
                pooled = torch.stack(
                    [state[row_indices, last_positions, :] for state in states[1:]], dim=1
                )
                acts_handle.write(pooled.to(torch.float16).cpu().numpy().tobytes())
                for _, row in pairs:
                    meta_handle.write(
                        json.dumps(
                            {
                                "id": row["id"],
                                "answerable": int(row["answerable"]),
                                "split": row.get("split"),
                                "question": row["question"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                acts_handle.flush()
                meta_handle.flush()
                completed += len(pairs)
                if batch_index == 1 or batch_index % 50 == 0 or batch_index == len(batches):
                    print(f"[{batch_index}/{len(batches)}] {completed:,}/{len(rows):,}", flush=True)

    expected_rows = len(completed_ids) + len(rows)
    expected = expected_rows * row_bytes
    actual = acts_path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"Activation cache size mismatch: expected {expected}, found {actual}")
    print(f"Wrote {acts_path} ({actual / 2**30:.2f} GiB) and {meta_path}")


if __name__ == "__main__":
    main()
