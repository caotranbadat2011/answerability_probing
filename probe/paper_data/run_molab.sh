#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
CACHE_ROOT="${CACHE_ROOT:-probe/caches/qwen3-4b-paper-data}"
OUTPUT_DIR="${OUTPUT_DIR:-probe/runs/qwen3-4b-paper-data}"
PILOT="${PILOT:-0}"
TYDI_LIMIT="${TYDI_LIMIT:-3000}"
UDHR_LIMIT="${UDHR_LIMIT:-0}"
WANDB_ARGS=()

if [[ "$PILOT" == "1" ]]; then
  TYDI_LIMIT="${PILOT_TYDI_LIMIT:-200}"
  UDHR_LIMIT="${PILOT_UDHR_LIMIT:-2}"
fi

if [[ "${WANDB_ENABLED:-0}" == "1" ]]; then
  WANDB_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT:-answerability-probing-paper-data}")
  if [[ -n "${WANDB_RUN_NAME:-}" ]]; then
    WANDB_ARGS+=(--wandb-run-name "$WANDB_RUN_NAME")
  fi
fi

echo "=== Prepare the paper datasets: TyDiQA + UDHR ==="
python data/fetch_sources.py --only tydiqa mist-sample mist-test
python data/extract_qa_templates.py
python data/tydi/build_tydiqa.py
python data/udhr/build_rows.py

echo "=== Cache TyDiQA activations (paper cap: <=3000 rows/language) ==="
python -u probe/probing/extract_activations.py \
  --source tydi --split train --limit-per-lang "$TYDI_LIMIT" \
  --model "$MODEL" --prompt-style bare --pooling last --max-len 1024 \
  --token-budget "${TOKEN_BUDGET:-4096}" --require-gpu \
  --out "$CACHE_ROOT/train"

echo "=== Cache UDHR activations ==="
python -u probe/probing/extract_activations.py \
  --source udhr --limit-per-lang "$UDHR_LIMIT" \
  --model "$MODEL" --prompt-style bare --pooling last --max-len 1024 \
  --token-budget "${TOKEN_BUDGET:-4096}" --require-gpu \
  --out "$CACHE_ROOT/udhr"

echo "=== Paper-style leave-one-language-out layer sweep on TyDiQA ==="
LAYERS="$(python - "$CACHE_ROOT/train/config.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
print(",".join(str(x) for x in config.get("saved_layers", range(config["n_layers"]))))
PY
)"
mkdir -p "$OUTPUT_DIR"
python probe/probing/train_probes.py \
  --acts "$CACHE_ROOT/train" --layers "$LAYERS" \
  --l2 "${L2:-0.001}" --device "${PROBE_DEVICE:-auto}" \
  --out "$OUTPUT_DIR/layer_sweep.json"

BEST_LAYER="$(python - "$OUTPUT_DIR/layer_sweep.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["best_layer"])
PY
)"
echo "Selected layer: $BEST_LAYER"

echo "=== Fit TyDiQA+UDHR probe and paper-style item-fold threshold ==="
mkdir -p "$OUTPUT_DIR/final_probe"
cp "$OUTPUT_DIR/layer_sweep.json" "$OUTPUT_DIR/final_probe/layer_sweep.json"
python probe/paper_data/train_paper_probe.py \
  --tydi-cache "$CACHE_ROOT/train" --udhr-cache "$CACHE_ROOT/udhr" \
  --layer "$BEST_LAYER" --l2 "${L2:-0.001}" --folds 5 --calib-fold 0 \
  --device "${PROBE_DEVICE:-auto}" --out "$OUTPUT_DIR/final_probe" \
  "${WANDB_ARGS[@]}"

echo "Paper-data probe: $OUTPUT_DIR/final_probe"
