# Qwen3-4B answerability probe on ViQuAD2

This workflow adapts the paper's frozen-activation linear probe to
`Qwen/Qwen3-4B-Instruct-2507` and `taidng/UIT-ViQuAD2.0`.

It deliberately keeps answerability separate from answer generation:

1. render `context + question` through Qwen's chat template without a refusal instruction;
2. freeze Qwen and cache the final-input-token state after every Transformer block;
3. fit an L2 logistic regression for every block and regularisation value;
4. choose the block by AUROC on one half of ViQuAD2 validation;
5. refit and choose the refusal threshold on the untouched half;
6. export `probe.npz` and `config.json` for inference.

`hidden_states[0]` is the embedding output and is not treated as a Transformer layer.
Block `i` is always read from `hidden_states[i + 1]`.

## Molab setup

```bash
git clone --branch qwen3-4b-viquad2 \
  https://github.com/caotranbadat2011/answerability_probing.git
cd answerability_probing

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-probe.txt
source .venv/bin/activate
```

Activate W&B only when desired:

```bash
export WANDB_ENABLED=1
export WANDB_PROJECT=answerability-probing-viquad2
export WANDB_RUN_NAME=qwen3-4b-viquad2-full
```

Run a small end-to-end pilot first:

```bash
PILOT=1 bash probe/viquad2/run_molab.sh
```

After the pilot succeeds, use fresh output directories for the full run:

```bash
DATA_DIR=data/viquad2_probe_full \
CACHE_ROOT=probe/caches/qwen3-4b-viquad2-full \
OUTPUT_DIR=probe/runs/qwen3-4b-viquad2/full_probe \
bash probe/viquad2/run_molab.sh
```

The activation cache is large because it stores all Transformer blocks. It is
resumable: rerunning the same extraction command skips IDs already written after
checking that cache shape and model settings are consistent.

## Output

The selected artifact directory contains:

- `probe.npz`: weight, bias, feature mean, and feature scale;
- `config.json`: model, block, hidden-state index, threshold, and metrics;
- `layer_sweep.json`: AUROC for every block/L2 pair;
- `calibration_predictions.jsonl`: per-question calibration decisions.

The 420-question legal evaluation must not be used to choose a layer. If a legal
development split is available, it may be used later to recalibrate only the
threshold; retain a separate legal test split for final reporting.
