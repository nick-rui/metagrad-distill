# Feasibility: memory & compute budget

Done **before** running, per requirement. Empirical numbers from `scripts/bench_metagrad.py`
(GPT-2 small, 1×H100 80GB, JAX, XLA platform allocator). Updated as measured.

## The expensive operation: one metagradient round
A round = backprop a held-out target loss `Φ` through an unrolled Adam inner loop of
`T` steps over `k` sequences of length `L_inner`, differentiating w.r.t. the `k`-vector
of per-sequence weights (a 2nd-order/reverse-over-reverse computation).

**Memory drivers** (float32):
- inner activations: `O(n_layer · k · L_inner · d)` — dominant, grows with `k`.
- vocab logits: `k · L_inner · V · 4B` (V=50257) — e.g. k=64,L=128 → 1.6 GB/forward.
- unrolled-trajectory carries: `T · 3 · |params|·4B` = `T · 1.5 GB` (params+Adam m,v). scan+`checkpoint` keeps only carries, recomputing inner activations in backward.

**Design decision (measured):** full per-block rematerialization cut memory but made
2nd-order XLA compile ~10 min/shape — impractical. Chosen path: **no block-remat**, keep
`k` moderate, and buy coverage with **more rounds** (rounds are cheap, compile is ~45 s).

### Measured (no block-remat, T=16, val_bs=128, 1×H100)
| k | L_inner | peak GB | compile s | round s | status |
|---|---|---|---|---|---|
| 32 | 128 | 48.0 | 43 | 1.82 | ok |
| 64 | 128 | 52.1 | 40 | 3.24 | **ok (chosen)** |
| 96 | 128 | — | — | — | OOM |
| 64 | 256 | — | — | — | OOM |
| 96 | 256 | — | — | — | OOM |

Memory is dominated by the unrolled-trajectory **carries (∝ T)**, ~52 GB at T=16,
nearly flat in `k` (48→52 GB for k=32→64) while round time grows with `k`
(1.8→3.2 s). That's why `L_inner=256` OOMs and `k=96` doesn't fit. Transient
**compile** scratch peaks ~75 GB (fits per-GPU, one shard each).

## Coverage & total labeling cost
A sequence's label = mean of its z-scored scores over the `C` rounds it appears in.
With corpus `M`, batch `k`, coverage `C`: rounds `R = C·M/k`, sharded over 8 GPUs.

```
R = C · M / k         wall ≈ (R · round_s) / 8 + compile(~45s)
```

## Chosen knobs (justified)
| Knob | Symbol | Value | Why |
|---|---|---|---|
| Corpus size | M | 50,000 seq (12.8M tok) | full oracle computable once; clean 3-cluster ground truth |
| Seq length | T_seq | 256 | design default-ish; featurize/final-train use full 256 |
| Inner length | L_inner | 128 | halves logits+activation memory; prefix is enough for the ranking signal |
| Batch / round | k | **64** | 52 GB peak; largest benchmarked fit; bigger z-score group than k=32 |
| Inner steps | T | 16 | DPG default; ablate {1,8,16,32} (runtime∝T; T=96 carries OOM, capped) |
| Inner optimizer | — | Adam | DPG finding; SGD kept as negative control |
| Coverage | C | 4 | each seq labeled ~4× → stable z-scored mean; R=3125 rounds, ~22 min on 8 GPU |
| Final budget | n | top 10% tokens | ablate {1,5,10,25}% |
| Classifier | — | LightGBM on 768-d mean hidden | strong cheap default; Ridge/MLP ablation |

## Other phases (all cheap relative to labeling)
- **Featurize** 50k seqs (forward only): minutes on 1 GPU.
- **Classifier** train/score: seconds (CPU).
- **Final CPT** per method: GPT-2 on ≤1.28M tokens, a few passes — minutes/GPU; ~8 methods × budgets run in parallel across GPUs.
- **Ablations**: T-sweep reuses one labeling shard; feature/k ablations reuse cached scores.

**Verdict:** the full plan fits comfortably in an overnight window on 8×H100. Labeling
(the bottleneck) is bounded to tens of minutes; everything else is minutes.
