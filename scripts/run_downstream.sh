#!/bin/bash
# End-to-end downstream pipeline: labels.npz -> H1/H3/H4 results.
# Run after Phase-1 labeling produces artifacts/labels/main/labels.npz.
# H2 (truncation) and H5 (cohorts) are launched separately (run_ablations.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
J=/root/jax-env/bin/python
A=/root/ai-env/bin/python
export HF_HOME=/root/hf_cache
# wandb ENABLED for CPT: the val-ppl-vs-tokens efficiency curves (our method vs
# baselines) are a required deliverable. featurize doesn't log; classifier only with --wandb.
# Auth comes from ~/.netrc (set once via `wandb login`); no key in the repo.
export WANDB_PROJECT=metagrad-distill
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TOKENIZERS_PARALLELISM=false

LABELS=artifacts/labels/main/labels.npz
DATA=artifacts/data/mgd_v1
FEATS=artifacts/features/mgd_v1.npz
CLF=artifacts/clf/main
TAG=b10

echo "==================== [1/4] FEATURIZE (jax) ===================="
$J -m src.classifier.featurize --data_dir $DATA --out_path $FEATS --bs 256

echo "==================== [2/4] CLASSIFIER / H1 (ai) ===================="
$A -m src.classifier.train --labels $LABELS --features $FEATS --data_dir $DATA \
   --out_dir $CLF --model lgbm
echo "--- H1 result ---"; cat $CLF/h1.json

echo "==================== [3/4] SELECTION all methods (ai) ===================="
$A -m scripts.run_select --labels $LABELS --pred $CLF/pred.npz \
   --features $FEATS --val_features ${FEATS%.npz}_val.npz \
   --data_dir $DATA --tag $TAG --budget_frac 0.10
echo "--- selection summary ---"; cat artifacts/select/$TAG/summary.json

echo "==================== [4/4] CPT all methods / H3+H4 (ai, 8 GPU) ===================="
$A -m scripts.run_cpt_all --select_dir artifacts/select/$TAG --data_dir $DATA \
   --out_dir artifacts/cpt/$TAG --epochs 3 --lr 3e-5 --bs 32 --eval_every_tokens 150000
echo "--- H3 table ---"; cat artifacts/report/$TAG/h3_table.md

echo "==================== DOWNSTREAM DONE ===================="
