# TODO — MetaGrad-Distill (MGD)

Living checklist. I update this as I work: `[ ]` todo, `[~]` in progress, `[x]` done, `[!]` blocked/abandoned (with reason).
Last updated: 2026-06-20 (initial).

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

## Phase 1 — Data pipeline (`src/data`)
- [ ] Pick + verify datasets load (PubMed good, C4 offdomain). Fallbacks if unavailable.
- [ ] Tokenize (GPT-2 BPE), chunk to `T_seq` sequences, build clusters.
- [ ] Held-out PubMed target val set (disjoint from corpus).
- [ ] Persist sequence store: token ids + metadata (`seq_id, cluster, n_tokens`). Report corpus stats.

## Phase 2 — Metagradient oracle (`src/metagrad`) — RISKIEST, build + unit-test first
- [ ] Differentiable inner loop `A`: per-example weighted LM loss, unrolled `optax.adam` for `T` steps, `remat`.
- [ ] `metagrad_scores(seqs, model_init, val, T, lr)` → `s = -tau`, `tau = grad(Φ_val, w)`.
- [ ] **Unit test (toy: ~10 seqs, T=4, tiny model):** grads flow to `w`, finite; sign sanity (val-like seq scores > corrupted seq).
- [ ] Memory/timing check at GPT-2 small, k=256/512, T=16.

## Phase 3 — Labeling loop (`src/labeling`)
- [ ] R rounds: reset model, sample k, zscore within round, average across rounds (~3–5x coverage). Persist `(seq_id, label)`.
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
