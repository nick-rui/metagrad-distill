# flash-hog ThunderKittens kernels (vendored, INACTIVE)

Vendored from **[kesavanramakrishnan/flash-hog](https://github.com/kesavanramakrishnan/flash-hog)**
(commit `751e3c0`) — a fork of `marcelroed/flash-hog` that adds a ThunderKittens (TK)
implementation of the attention **double-backward** (bwd-of-bwd), the kernel the
metagradient HVP runs through.

## Status: kept for reference / future use — we run the **Pallas** path for now
The active attention backend in this project is the stock **Pallas** double-backward
(`GPT2Config.attn_impl="flashhog"` → `flash_hog.jax.attention.dot_product_attention`,
which defaults to Pallas). These TK kernels are **not** wired into the pipeline and are
**not** built on this node. They are vendored here so the source travels with the repo.

## What's here
```
csrc/tk_bwdbwd/   ffi.cu, stage1.cuh, stage2.cuh   — the TK CUDA kernel (+ BENCHMARKS.md)
jax/_tk_build.py  jax/_tk_gpu.py                    — JAX FFI wrappers (lazy build via enable())
```

## Why it's faster (BENCHMARKS.md)
Only the **double-backward** differs from Pallas (cuDNN fwd + first bwd are identical).
TK wins at **long sequences** and is ~neutral/slower at short ones (H200, bf16):
seq 512 → 0.82×, 1024 → 1.17×, 2048 → 1.34× (grows further at longer seq). This matches
our finding that flash-hog's payoff is in the long-context regime (results.md §2.1).

## How to activate (NOT done here)
TK is opt-in and lazily compiled: `flash_hog.jax._tk_gpu.enable()` triggers an `nvcc`
build of `ffi.cu` against the ThunderKittens headers. The fork's `tk` extra pins
**`nvidia-cuda-nvcc>=13` (CUDA 13)** — this node is **driver 570 / CUDA 12.8 max**
(see `../../ENV.md`), so the TK build is not attempted here. To use TK, build on a
CUDA-13-capable box, then call `enable()` before the first attention call; the Pallas
path is numerically identical (cos 1.0000 vs fp32), so labels are unaffected.
