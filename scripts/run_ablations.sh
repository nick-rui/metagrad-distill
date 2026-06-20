#!/bin/bash
# Ablations: flash-hog A/B (memory/speed + ranking agreement), H2 truncation, H5 cohorts.
# Run after run_downstream.sh (needs features + classifier preds for H5).
set -euo pipefail
cd "$(dirname "$0")/.."
J=/root/jax-env/bin/python
A=/root/ai-env/bin/python
export HF_HOME=/root/hf_cache
export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TOKENIZERS_PARALLELISM=false

DATA=artifacts/data/mgd_v1
PRED=artifacts/clf/main/pred.npz
FEATS=artifacts/features/mgd_v1.npz

echo "==================== flash-hog A/B bench (jax, 1 GPU) ===================="
# xla OOMs at L_inner>=256 today; flash-hog should extend the feasible range.
CUDA_VISIBLE_DEVICES=0 $J -m scripts.bench_flashhog --k 64 --T 16 \
   --L_list 128,256,512,1024 --out artifacts/bench/flashhog_bench.json
cat artifacts/bench/flashhog_bench.json

echo "==================== H2 truncation ablation (jax, 1 GPU) ===================="
CUDA_VISIBLE_DEVICES=0 $J -m src.eval.truncation --data_dir $DATA \
   --out_path artifacts/ablation/truncation.json --Ts 1 2 4 8 16 --k 32 \
   --n_batches 12 --optimizer adam
cat artifacts/ablation/truncation.json

echo "==================== H5 cohort lift (ai, 8 GPU) ===================="
$A -m scripts.run_cohorts --pred $PRED --features $FEATS --data_dir $DATA \
   --cohort_dir artifacts/cohorts/sel --cpt_dir artifacts/cohorts/cpt \
   --n_per 2000 --n_cohorts 14 --epochs 2 --lr 3e-5
cat artifacts/cohorts/h5.json

echo "==================== ABLATIONS DONE ===================="
