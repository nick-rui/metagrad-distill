#!/bin/bash
# Parameterized pipeline: featurize -> classifier(H1) -> select -> CPT(H3) for ANY
# corpus + labels. Usage:
#   run_pipeline.sh <data_dir> <labels.npz> <tag>
# Optional 4th arg: a pred.npz from a DIFFERENT corpus' classifier (cross-corpus
# amortization #3) — skips local classifier training and selects with that pred.
set -euo pipefail
cd "$(dirname "$0")/.."
J=/root/jax-env/bin/python
A=/root/ai-env/bin/python
export HF_HOME=/root/hf_cache WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform TOKENIZERS_PARALLELISM=false

DATA=$1; LABELS=$2; TAG=$3; XPRED=${4:-}
FEATS=artifacts/features/$TAG.npz
CLF=artifacts/clf/$TAG

echo "==== [1] FEATURIZE $DATA ===="
$J -m src.classifier.featurize --data_dir $DATA --out_path $FEATS --bs 256

if [ -z "$XPRED" ]; then
  echo "==== [2] CLASSIFIER / H1 ===="
  $A -m src.classifier.train --labels $LABELS --features $FEATS --data_dir $DATA --out_dir $CLF --model lgbm
  echo "--- H1 ---"; cat $CLF/h1.json
  PRED=$CLF/pred.npz
else
  echo "==== [2] CROSS-CORPUS: using external pred $XPRED ===="
  PRED=$XPRED
fi

echo "==== [3] SELECT $TAG ===="
$A -m scripts.run_select --labels $LABELS --pred $PRED --features $FEATS \
   --val_features ${FEATS%.npz}_val.npz --data_dir $DATA --tag $TAG --budget_frac 0.10
echo "--- selection purity ---"; cat artifacts/select/$TAG/summary.json

echo "==== [4] CPT / H3 ($TAG) ===="
$A -m scripts.run_cpt_all --select_dir artifacts/select/$TAG --data_dir $DATA \
   --out_dir artifacts/cpt/$TAG --epochs 3 --lr 3e-5 --bs 32 --eval_every_tokens 150000
echo "--- H3 ---"; cat artifacts/report/$TAG/h3_table.md
echo "==== PIPELINE $TAG DONE ===="
