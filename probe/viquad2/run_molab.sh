#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
DATA_DIR="${DATA_DIR:-data/viquad2_probe}"
CACHE_ROOT="${CACHE_ROOT:-probe/caches/qwen3-4b-viquad2}"
OUTPUT_DIR="${OUTPUT_DIR:-probe/runs/qwen3-4b-viquad2/best_probe}"
PILOT="${PILOT:-0}"
WANDB_ARGS=()
PREPARE_ARGS=()
EXTRACT_ARGS=()

if [[ "$PILOT" == "1" ]]; then
  PREPARE_ARGS+=(--max-train 400 --max-validation 200 --max-test 200)
  EXTRACT_ARGS+=(--limit 0)
fi

if [[ "${WANDB_ENABLED:-0}" == "1" ]]; then
  WANDB_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT:-answerability-probing-viquad2}")
  if [[ -n "${WANDB_RUN_NAME:-}" ]]; then
    WANDB_ARGS+=(--wandb-run-name "$WANDB_RUN_NAME")
  fi
fi

python probe/viquad2/prepare_viquad2.py \
  --output-dir "$DATA_DIR" \
  "${PREPARE_ARGS[@]}"

python probe/viquad2/extract_activations.py \
  --input "$DATA_DIR/train.jsonl" \
  --out "$CACHE_ROOT/train" \
  --model "$MODEL" \
  --require-gpu \
  "${EXTRACT_ARGS[@]}"

python probe/viquad2/extract_activations.py \
  --input "$DATA_DIR/validation.jsonl" \
  --out "$CACHE_ROOT/validation" \
  --model "$MODEL" \
  --require-gpu \
  "${EXTRACT_ARGS[@]}"

python probe/viquad2/train_probe.py \
  --train-cache "$CACHE_ROOT/train" \
  --validation-cache "$CACHE_ROOT/validation" \
  --out "$OUTPUT_DIR" \
  --min-answerable-accuracy "${MIN_ANSWERABLE_ACCURACY:-0.85}" \
  "${WANDB_ARGS[@]}"

# The test split is touched only after layer, L2, and threshold are frozen.
python probe/viquad2/extract_activations.py \
  --input "$DATA_DIR/test.jsonl" \
  --out "$CACHE_ROOT/test" \
  --model "$MODEL" \
  --require-gpu \
  "${EXTRACT_ARGS[@]}"

python probe/viquad2/evaluate_probe.py \
  --probe "$OUTPUT_DIR" \
  --test-cache "$CACHE_ROOT/test" \
  --out "$OUTPUT_DIR/test_evaluation" \
  "${WANDB_ARGS[@]}"

echo "Probe artifact: $OUTPUT_DIR"
