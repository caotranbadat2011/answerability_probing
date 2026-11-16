# Qwen3-4B probe on the paper datasets

This workflow replaces ViQuAD2 with the paper's probe data and protocol:

- TyDiQA answerability train, capped at 3,000 rows per language;
- UDHR-QA, 17,280 rows across 24 question and 25 context languages;
- bare `context + question` prompt and final-token hidden state;
- L2 logistic regression (`l2=0.001`);
- leave-one-language-out TyDiQA layer selection;
- five item-disjoint UDHR folds for threshold calibration;
- final probe trained on TyDiQA plus four UDHR folds.

The data protocol matches the paper. The backbone remains
`Qwen/Qwen3-4B-Instruct-2507` for TrustMargin compatibility, rather than the
paper's `Qwen/Qwen3.5-9B`, so this is an adaptation rather than a numerical
reproduction.

## Molab

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-probe.txt
uv pip install --python .venv/bin/python -r requirements-data.txt
source .venv/bin/activate
```

Pilot:

```bash
PILOT=1 bash probe/paper_data/run_molab.sh
```

Full run with W&B:

```bash
CACHE_ROOT=probe/caches/qwen3-4b-paper-data-full \
OUTPUT_DIR=probe/runs/qwen3-4b-paper-data-full \
WANDB_ENABLED=1 \
WANDB_PROJECT=answerability-probing-paper-data \
WANDB_RUN_NAME=qwen3-4b-tydiqa-udhr \
bash probe/paper_data/run_molab.sh
```

The inference artifact is written to `OUTPUT_DIR/final_probe/` as
`probe.npz`, `config.json`, and `udhr_oof_predictions.jsonl`. Use different
cache paths for pilot and full runs.
