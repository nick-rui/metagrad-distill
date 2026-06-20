# Results — MetaGrad-Distill (MGD)

> Clean, curated results log. Each section states a hypothesis, the setup, the number, and the takeaway.
> Raw run artifacts live in `artifacts/`. This file is the human-readable summary of record.

**Project:** Distill an expensive metagradient data-quality oracle into a cheap classifier for data selection in continued pretraining (CPT). See `design_doc.md`.

**Status:** 🚧 In progress — started 2026-06-20. Env rebuilt + **flash-hog higher-order attention integrated** (2026-06-20); Phase-1 labeling running.

---

## 0. Setup & environment

| Component | Choice |
|---|---|
| Metagradient engine | JAX 0.10.2 + optax (unrolled differentiable Adam, `jax.grad`, `remat`) |
| Attention backend | `xla` float32 explicit SDPA (default, validated) **or** `flashhog` — [flash-hog](https://github.com/marcelroed/flash-hog) higher-order Pallas kernel (linear-memory, bf16) for long-context metagradients |
| Eval / features | PyTorch 2.10 (`/root/ai-env`) |
| Model | GPT-2 small (124M) |
| Testbed | Continued pretraining (CPT) |
| Target domain `Φ` | held-out LM loss on PubMed biomedical abstracts |
| Corpus `D` | mixture: PubMed (good) / C4 web (off-domain) / token-shuffled PubMed (corrupt) |
| Hardware | 8× H100 80GB (single node) |

Method recap: metagradient `τ_i = ∂Φ/∂w_i` at `w=1` via backprop through a short Adam inner loop; goodness `s_i = −τ_i`. Distill `s` into a cheap regressor over features, score the whole corpus, keep top-`n`.

---

## 1. Hypotheses & headline numbers

| ID | Hypothesis | Metric | Result |
|---|---|---|---|
| H1 | Cheap classifier predicts oracle metagradient scores | Spearman ρ(ŝ, s) | _pending_ |
| H2 | Short inner loops preserve ranking | ρ(trunc-T, full-T) | _pending_ |
| H3 | Top-n selection beats baselines, approaches oracle | held-out PubMed ppl | _pending_ |
| H4 | Classifier is cheap + predictive (Pareto) | power vs cost | _pending_ |
| H5 | Aggregate ŝ predicts cohort lift | held-out R²/ρ | _pending_ |

---

## 2. Detailed results

_(populated as experiments complete; newest first within each subsection)_

### 2.3 Metagradient oracle quality (relabel @ lr=3e-5, 3200 rounds, 8×H100, 2026-06-20)
Full Phase-1 labeling at the corrected lr. 49,168/50,000 seqs covered (mean 4.2 rounds each), ~23 min. The averaged z-scored oracle label separates the clusters almost perfectly:

| cluster | mean label | top-10%-by-label share (corpus base) |
|---|---|---|
| good (PubMed) | **+0.709** | **99.5%** (40%) |
| offdomain (C4) | −0.313 | 0.4% (40%) |
| corrupt (shuffled) | −0.795 | 0.1% (20%) |

Cohen's d(good−corrupt) = **+1.94**. I.e. selecting the top-10% of tokens by the metagradient oracle yields a near-pure PubMed set — the oracle is a strong, ground-truth-aligned data selector. This is the expensive signal the cheap classifier (H1) must now reproduce.

### 2.2 Inner-loop lr is decisive — metagradient only ranks data in the *stable* regime (2026-06-20)
The first full labeling run (lr=1e-3, the original default) produced **near-noise labels**: cluster means good −0.002 / offdomain −0.002 / corrupt +0.008, i.e. corrupt (token-shuffled PubMed) scored *highest* and was 2× over-represented in the top-10% (Cohen's d good−corrupt = −0.02). Root cause: at lr=1e-3 the inner Adam loop **destroys the model in 16 steps** — val loss `phi` runs to 6.6–6.7 (ppl ~750) vs base GPT-2 PubMed loss 3.45 (ppl 31). The metagradient through a diverging trajectory measures catastrophic-forgetting dynamics dominated by the high-gradient corrupt sequences, not genuine data value.

An lr sweep (`scripts/diag_lr.py`, 30 rounds each, k=64/T=16) shows a sharp transition:

| inner lr | phi (val loss, base=3.45) | Cohen's d (good−corrupt) | cluster means good / off / corrupt |
|---|---|---|---|
| **3e-5** | **3.63 (stable)** | **+1.75** ✓ | **+0.73 / −0.34 / −0.80** |
| 1e-4 | 4.00 | −4.35 | −0.78 / −0.03 / +1.62 |
| 3e-4 | 5.38 | −1.69 | −0.53 / −0.01 / +1.08 |
| 1e-3 (orig) | 6.70 (diverged) | −0.005 | +0.05 / −0.07 / +0.06 |

**Takeaway:** only when the inner loop stays near the base model (lr≈3e-5, phi≈base) does the metagradient recover the ground-truth ordering **good > offdomain > corrupt** — and there it does so cleanly (d=+1.75). This is the metagradient analogue of the DPG paper's "short proxy runs are enough" caveat: the proxy must be *stable*, not just short. **Fix applied:** default inner lr → 3e-5 in `label.py`/`run_label.py`; full labeling re-run at lr=3e-5.

### 2.1 flash-hog higher-order attention (H2-enabler)
- **flash-hog runs on this node** (2026-06-20): installed `flash-hog==0.6.0` into the cuda12 jax-env via `--no-deps` (its `pyproject` pins `jax[cuda13]`, which would *not* run on this driver-570 / CUDA-12.8 box; the Pallas kernel itself compiles fine against our existing `jax[cuda12]` 0.10.2). Verified the full higher-order path on an H100: forward → `bwd` → `bwd_fwd` → `bwd_bwd` all execute and return **finite** gradients, including a `grad(grad(...))` HVP test. vmap-batching works.
- **Constraints found:** (1) the kernel routes through cuDNN fused attention → **bf16/fp16 only** (float32 q/k/v raises `NotImplementedError`); we cast q/k/v to bf16 inside attention and back. (2) API is per-sequence `[T, n_heads, head_dim]` (no batch dim) → we `vmap` over the batch. Integrated as an opt-in backend `GPT2Config.attn_impl="flashhog"` in `src/metagrad/model_gpt2.py`; default stays the validated float32 `xla` path.
- **A/B vs XLA** (`scripts/bench_flashhog.py`): _pending_ — peak-memory / round-time at L_inner∈{128,256,512,1024} and Spearman ρ(s_xla, s_flashhog). Hypothesis: flash-hog's linear attention memory makes long-L_inner metagradients feasible where the float32 XLA path OOMs (XLA OOMs at L_inner=256 today).

### 2.0 Sanity checks
- **Fresh-env rebuild verified** (2026-06-20): both venvs rebuilt from scratch on a clean container per `ENV.md` (jax-env: jax 0.10.2 + 8 GPUs; ai-env: torch 2.10.0+cu128, CUDA on 8 devices). Corpus regenerated identically (M=50000, V=2000, 12.8M tokens). Metagrad unit test re-PASSED in the rebuilt env (good +0.93 > corrupt −0.92).
- **GPT-2 JAX forward validated** (2026-06-20): on held-out data, val(pubmed) ppl=32.2, good(pubmed)=31.2, offdomain(c4)=37.4, corrupt(shuffled)=4137. Correct ordering → the testbed has a clean ground truth (target=pubmed ⇒ good < offdomain ≪ corrupt in loss).
- **Metagradient unit test PASS** (2026-06-20): tiny 2-layer model, T=4. `tau = ∂Φ/∂w` finite & non-zero; sign sanity holds — val-matching "good" seqs score +0.92 vs random "corrupt" −0.91 (good > corrupt). Fixes that mattered: eps *inside* Adam's `sqrt(v+eps)` (else 0·∞=NaN for zero-grad params), moderate attention mask value −1e9 (not `finfo.min`) for stable 2nd-order grad.

---

## 3. Notes, surprises, decisions
- 2026-06-20: Project kickoff. Metagradients in JAX (per requirement); direct unrolled Adam since `T` is small (REPLAY reserved for scaling).
- 2026-06-20: **Feasibility measured before running** (`FEASIBILITY.md`). One metagrad round (GPT-2 small, no block-remat): k=64, L_inner=128, T=16 → 52 GB peak (75 GB live under the platform allocator while streaming rounds), 3.24 s/round. Memory is dominated by the unrolled-trajectory carries (∝T), nearly flat in k; L_inner=256 and k≥96 OOM. Per-block remat fixed memory but made 2nd-order compile ~10 min → rejected; chosen lever is more (cheap) rounds at moderate k.
- 2026-06-20: Chosen knobs — M=50k, T_seq=256, L_inner=128, k=64, T=16, coverage C=4, budget n=top-10% tokens (ablate 1/5/10/25%).
- 2026-06-20: **Phase-1 labeling running** on all 8×H100 (3200 rounds total, ~22 min, k=64/T=16/L_inner=128).
- 2026-06-20: **Infra rebuilt on fresh container + flash-hog integrated.** Two venvs rebuilt from scratch (jax-env, ai-env). flash-hog higher-order attention kernel installed and verified on driver-570/CUDA-12.8 (had to bypass its `jax[cuda13]` pin via `--no-deps`; leaf deps added: chex/jaxtyping/toolz/einops/wadler_lindig). Wired as opt-in `attn_impl="flashhog"` backend; bf16-only + per-sequence `[T,nh,hd]` API handled in `_sdpa_flashhog`. Rationale: at the current L_inner=128 attention isn't the memory bottleneck (logits `[k,L,vocab]` + trajectory carries dominate), but flash-hog's linear attention memory is what unlocks the long-L_inner / longer-T_seq ablations the XLA path OOMs on — strengthening **H2**.
