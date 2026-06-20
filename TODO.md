# TODO — MetaGrad-Distill (MGD)

Living checklist. I update this as I work: `[ ]` todo, `[~]` in progress, `[x]` done, `[!]` blocked/abandoned (with reason).
Last updated: 2026-06-20 (env rebuilt on fresh container; flash-hog integrated; labeling running).

## flash-hog higher-order attention (added 2026-06-20)
- [x] Install `flash-hog` on driver-570/CUDA-12.8 (via `--no-deps`, bypassing its `jax[cuda13]` pin). Verify fwd/bwd/bwd_bwd finite on H100.
- [x] Wire opt-in backend `GPT2Config.attn_impl="flashhog"` (`_sdpa_flashhog`: bf16 + per-seq vmap). Default `xla` path unchanged + re-verified.
- [ ] A/B bench vs XLA (`scripts/bench_flashhog.py`): peak-mem/time + ρ(s) at L_inner∈{128,256,512,1024}. Show long-L_inner feasibility (XLA OOMs at 256). → **H2 enabler**.

## Legend / conventions
- Two envs: `/root/jax-env` (metagradients, JAX) and `/root/ai-env` (torch: eval, sentence features, vLLM).
- Model: GPT-2 small (124M). Domain testbed: continued pretraining (CPT).
- Target domain `Φ`: held-out LM loss on **PubMed biomedical abstracts** (DAPT-style, Gururangan 2020).
- Corpus `D` = mixture of clusters: `good` (PubMed) / `offdomain` (C4 web) / `corrupt` (token-shuffled PubMed). Ground-truth: a working method preferentially selects `good`.

## Phase 0 — Infra & environment
- [x] Survey node, install torch stack (`/root/ai-env`), cu128 fix, flash-attn.
- [x] Install JAX+CUDA (`/root/jax-env`), verify 8 GPUs.
- [x] Research metagradient implementations (Engstrom 2025 REPLAY; Thrush 2026 DPG). Decide: direct unrolled `optax.adam` + `jax.grad` + `remat`.
- [x] Repo scaffolding, TODO.md, results.md, README, ENV.md. Commit.

## Phase 1 — Data pipeline (`src/data`) — DONE
- [x] Pick + verify datasets load (PubMed=MedRAG/pubmed, offdomain=allenai/c4).
- [x] Tokenize (GPT-2 BPE), chunk to `T_seq=256`, build clusters {good,offdomain,corrupt}.
- [x] Held-out PubMed target val set (2k seqs, disjoint).
- [x] Persist store: `artifacts/data/mgd_v1/` — tokens.npy [50000,256], val.npy [2000,256], meta.csv, stats.json (12.8M tokens).

## Phase 2 — Metagradient oracle (`src/metagrad`) — CORE VALIDATED
- [x] Differentiable inner loop `A`: hand-rolled differentiable Adam, unrolled via `lax.scan`+`jax.checkpoint`.
- [x] `metagrad_scores()` → `s=-tau`. Hand-wrote GPT-2 in JAX (transformers v5 dropped Flax); validated fwd vs ppl.
- [x] **Unit test PASS:** grads flow to `w`, finite; good(+0.92) > corrupt(-0.91). Fixed NaN (eps-in-sqrt; mask=-1e9). Re-PASSED in rebuilt env 2026-06-20.
- [x] Memory/timing check (`FEASIBILITY.md`): k=64 fits (52GB), k≥96 / L_inner=256 OOM under XLA float32 → motivates flash-hog.

## Phase 3 — Labeling loop (`src/labeling`)
- [~] R rounds: reset model, sample k, zscore within round, average across rounds (~3–5x coverage). Persist `(seq_id, label)`. **Running 2026-06-20**: 3200 rounds across 8×H100, k=64/T=16/L_inner=128, ~4.1x coverage.
- [ ] **Full-metagradient oracle** over all M (small scale) = gold standard for validation.

## Phase 4 — Classifier (`src/classifier`)
- [ ] Featurizer: base GPT-2 mean hidden state (primary); sentence-transformer embedding (ablation).
- [ ] Regressor: LightGBM / Ridge / MLP.
- [ ] **H1 Faithfulness:** Spearman ρ(ŝ, oracle s) on held-out seqs.

## Phase 5 — Selection (`src/select`)
- [ ] Score all M (forward only), pick top-n (token budget). Optional dedup/diversity penalty.

## Phase 6 — Final CPT + eval (`src/train_final`, `src/baselines`, `src/eval`)
- [ ] CPT GPT-2 on selected tokens; eval held-out PubMed ppl/loss.
- [ ] Baselines: random, perplexity-top, perplexity-correlation, DSIR-style, oracle top-n, full-corpus.
- [ ] **H3 Downstream win:** classifier top-n > baselines, approaches oracle.

## Phase 7 — Ablations & Pareto (`src/eval`)
- [ ] **H2 Truncation:** ρ(truncated-T, full-T) over T∈{1,8,16,32,96}; smallest T preserving ranking. Adam-vs-SGD control.
- [ ] Feature ablation, k ablation, budget ablation (1/5/10/25%).
- [ ] **H4 Pareto:** predictive-power-vs-cost plot for all methods.

## Phase 8 — Cohort lift (`src/eval`) [stretch]
- [ ] **H5:** build cohorts (vary one property), CPT each, lift = ppl improvement; aggregate ŝ predicts held-out cohort lift (R²/ρ + scatter).

## Hygiene (always)
- [ ] Keep `results.md` clean + current. Commit incrementally. Update this TODO.
