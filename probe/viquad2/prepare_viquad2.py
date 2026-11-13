#!/usr/bin/env python3
"""Download ViQuAD2 and write stable JSONL splits for activation extraction."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset


DEFAULT_DATASET = "taidng/UIT-ViQuAD2.0"


def normalise_row(row: dict, split: str, index: int) -> dict:
    missing = [key for key in ("context", "question", "is_impossible") if key not in row]
    if missing:
        raise KeyError(f"ViQuAD2 row is missing required columns: {missing}")
    impossible = bool(row["is_impossible"])
    row_id = str(row.get("id") or f"{split}-{index:06d}")
    answers = row.get("answers") or {"text": [], "answer_start": []}
    return {
        "id": row_id,
        "split": split,
        "context": str(row["context"]),
        "question": str(row["question"]),
        "answerable": int(not impossible),
        "is_impossible": impossible,
        "answers": answers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("data/viquad2_probe"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--max-train", type=int, default=0, help="0 keeps all rows")
    parser.add_argument("--max-validation", type=int, default=0, help="0 keeps all rows")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    requested = {
        "train": (args.train_split, args.max_train),
        "validation": (args.validation_split, args.max_validation),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": args.dataset, "seed": args.seed, "splits": {}}

    for output_name, (source_name, limit) in requested.items():
        if source_name not in dataset:
            raise KeyError(f"Dataset has splits {list(dataset)}, not {source_name!r}")
        source = dataset[source_name]
        indices = list(range(len(source)))
        if limit:
            random.Random(args.seed).shuffle(indices)
            indices = sorted(indices[:limit])
        rows = [normalise_row(source[index], output_name, index) for index in indices]
        path = args.output_dir / f"{output_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts = Counter(row["answerable"] for row in rows)
        manifest["splits"][output_name] = {
            "source_split": source_name,
            "rows": len(rows),
            "answerable": counts[1],
            "unanswerable": counts[0],
            "path": str(path),
        }
        print(
            f"{output_name}: {len(rows):,} rows | "
            f"answerable={counts[1]:,} | unanswerable={counts[0]:,} -> {path}"
        )

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
