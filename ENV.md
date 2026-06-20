# Environment notes

This node is a Prime Intellect 8×H100 container. **Host NVIDIA driver = 570.124.06 (CUDA 12.8 max), not upgradable from inside the container.** Everything GPU must be compatible with CUDA ≤ 12.8.

## `/root/jax-env` (metagradients)
```
uv venv /root/jax-env --python 3.12 --seed
uv pip install "jax[cuda12]" flax optax
```
- jax 0.10.2 — works on driver 570 despite bundling CUDA 12.9 pip libs (JAX forward-compat / its own ptxas). Verified on all 8 GPUs.
- Run: `/root/jax-env/bin/python ...`

## `/root/ai-env` (torch eval / features / final CPT / vLLM)
```
uv venv /root/ai-env --python 3.12 --seed
uv pip install --reinstall torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "vllm==0.19.1" --torch-backend=cu128          # newest vLLM whose default wheel is cu128
uv pip install --torch-backend=cu128 accelerate peft trl datasets deepspeed bitsandbytes wandb einops sentencepiece evaluate scikit-learn matplotlib
MAX_JOBS=64 CUDA_HOME=/usr/local/cuda uv pip install flash-attn --no-build-isolation
```
- torch 2.10.0+cu128, vllm 0.19.1, transformers 5.12.1. **vLLM ≥0.20 ships only cu129/cu130 wheels → will NOT run here.**
- Run: `/root/ai-env/bin/python ...`

## Gotcha
- Do **not** install `jax[cuda12]` and `torch` in the same venv — their bundled `nvidia-*-cu12` libs conflict. Keep them separate (done).
- Why the metagradient work is in JAX: differentiating through an unrolled optimizer is first-class via `jax.grad` + `optax` + `jax.checkpoint`; in PyTorch it needs `functorch`/`higher` gymnastics. Per project requirement.
