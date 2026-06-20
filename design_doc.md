# MetaGrad-Distill: A Cheap Data-Quality Classifier Distilled from Metagradient Attribution

**Hackathon:** Mercor Inference-Time Compute Hackathon — Applied AI Track
**Inspiration:** *Synthetic Data for any Differentiable Target* (Thrush et al., 2026) and the metagradient machinery of Engstrom et al. (2025).

---

## 1. TL;DR

We want to pick the best `n` tokens out of a corpus of `N` tokens (`n << N`) to train a model under a fixed token budget. The *expensive but high-quality* signal for "is this sequence good training data?" is the **metagradient**: differentiate a downstream target metric (e.g. validation loss) with respect to a per-sequence loss weight. A large favorable metagradient means training on that sequence moves the target metric the most.

Computing this metagradient for the **entire** corpus is roughly as costly as just training on everything — useless under a budget. So instead we:

1. Compute true metagradient scores for a **small sampled subset** of sequences, using **short truncated inner training runs** (cheap).
2. **Distill** those scores into a cheap classifier/regressor: `sequence → predicted goodness`.
3. Run the cheap classifier over the **whole corpus** (forward passes only) and keep the **top `n`**.
4. Train the final model on the selected tokens.

The classifier amortizes a one-time, bounded "oracle" cost into a reusable scorer that is cheap to run everywhere. That is precisely the move the hackathon's cost-quality graph hints at: **a point that breaks the Pareto frontier** — cheap to evaluate, yet inheriting the predictive power of an expensive metric.

---

## 2. Problem framing

Let `D` be a corpus chunked into `M = N / T_seq` sequences of `T_seq` tokens each. We have a token budget allowing training on `n` tokens, i.e. on `n / T_seq` sequences. We want to choose the subset that maximizes a downstream target metric `Φ` (e.g. loss on a held-out target distribution, or a benchmark).

This is the hackathon's core question in disguise: **find a cheap metric, computable without training, that predicts the training value of data, and place it on the cost-quality Pareto frontier.** Our metric is *learned* — a classifier trained against an expensive oracle.

We use **continued pretraining (CPT)** as the primary testbed rather than RL. CPT is fully differentiable end-to-end, cheap, and isolates the mechanism cleanly. The method is testbed-agnostic (`Φ` and the inner algorithm `A` can be swapped), so the RL version is a stretch goal (§9). The hackathon explicitly accepts solutions that "reason from first principles," so a clean CPT demonstration of the mechanism is in scope.

---

## 3. Background: the metagradient as a data-quality oracle

Following the DPG paper, let `A(w, D)` be a learning algorithm that trains a target model on `D` with a **per-sequence weighted loss** `Σ_i w_i · ℓ(model, x_i)`, with all `w_i = 1`. Let `Φ(A(w, D))` be a differentiable downstream target (we use held-out validation LM loss). The **metagradient** is

```
τ_i = ∂Φ / ∂w_i   evaluated at w = 1
```

obtained by backpropagating through the inner training trajectory. Intuitively, `τ_i` says how much up-weighting sequence `i` would change the target. If `Φ` is a loss we minimize, then **lower `τ_i` = better sequence**, so we define the goodness score

```
s_i = − τ_i
```

Two findings from the DPG paper carry directly into our design:

- **Use Adam in the inner loop**, not SGD. Single-step SGD metagradients collapse to influence-function approximations and underperform; Adam (and several inner steps) gives the signal real teeth.
- The inner algorithm `A` is **stateless** — reset the target model at the start of every labeling round so scores are comparable across rounds.

The cost of the metagradient scales with the number of inner steps and the number of sequences in the run. The whole proposal is about keeping both small while preserving the ranking.

---

## 4. The two observations that make it cheap

**Observation 1 — short proxy runs are enough.** You can usually tell whether a training run will be good well before it finishes (~10% in). The *ranking* of sequences by usefulness stabilizes early, even if absolute losses don't. So we compute metagradients through a **truncated** inner loop (few steps `T`) instead of a full run. This is an assumption we explicitly test (ablation over `T`).

**Observation 2 — goodness is a learnable function of cheap features.** Data quality correlates with surface and semantic properties (domain, language, fluency, topical match to the target, n-gram statistics, base-model perplexity). If so, a cheap classifier can predict the oracle score from cheap features and generalize to unscored sequences. This is what lets us avoid computing metagradients for all `M` sequences.

Together: pay a small, bounded oracle cost on a sample → distill → score everything cheaply.

---

## 5. Method: MetaGrad-Distill (MGD)

### Phase 1 — Label generation (bounded oracle cost)

```
for round = 1 ... R:
    reset target model to base init        # A is stateless
    seqs ← sample k sequences from D        # k small, fits one inner run
    s    ← metagradient_scores(seqs, target_model_init, val_set, T, lr)
    s    ← zscore(s)                        # group-relative: removes per-batch offset
    accumulate s into a running per-sequence mean
labels = { seq : mean of its z-scored scores across the rounds it appeared in }
```

`metagradient_scores` (conceptual — see §7 for libraries):

```
def metagradient_scores(seqs, model_init, val_set, T, lr):
    w     = ones(len(seqs), requires_grad=True)
    model = clone(model_init)
    state = init_adam(model)
    for t in range(T):
        per_ex_loss = lm_loss_per_example(model, seqs)     # vector, length k
        loss = sum(w * per_ex_loss) / len(seqs)
        # every update must be a differentiable function of w:
        model, state = differentiable_adam_step(model, grad(loss, model), state, lr)
    Phi = lm_loss(model, val_set)            # scalar, differentiable in model → in w
    tau = grad(Phi, w)                       # metagradient, length k
    return -tau                              # higher = better (Phi is a loss)
```

**Why z-score + average across rounds.** A sequence's metagradient depends on the batch it sits in and the model state, so a raw score is noisy. Z-scoring within a batch is the group-relative normalization GRPO uses in the DPG paper; averaging over several batches (each sequence appears in multiple rounds) turns a context-dependent signal into a stable, transferable label the classifier can actually learn.

### Phase 2 — Classifier training (negligible cost)

Featurize each labeled sequence, then fit a regressor `f: features → predicted score`.

- **Features (cheap, pick one to start):** frozen small sentence-embedding; or the base model's mean hidden state; or summary stats (base-model perplexity, length, n-gram entropy, type-token ratio). Embeddings are the strong default.
- **Regressor:** gradient-boosted trees (LightGBM/XGBoost) or a small MLP. Linear is a fine baseline.

### Phase 3 — Corpus scoring + selection (cheap)

Run the featurizer + regressor over all `M` sequences (forward only, no training, no optimizer). Keep the **top `n / T_seq`** by predicted score. Optionally add a dedup/diversity penalty so the top-`n` isn't near-duplicates (ties into the hackathon's diversity/dedup metrics).

### Phase 4 — Final training + evaluation

Train the model on the selected tokens; evaluate against baselines (§8).

---

## 6. Why this should sit where it sits on the Pareto frontier

**Cost.** Marginal cost over "just train on `n`" is labeling + scoring:

- Labeling ≈ `R · k · T` sequence-steps. Choose so this is `<< N` (e.g. label ~10k sequences with `T = 16` is tiny next to a full-corpus pass).
- Classifier training ≈ free.
- Corpus scoring ≈ `M` cheap forward passes — far below a training pass, and far below the full-metagradient oracle (which costs a full-corpus training run *plus* metagradient overhead).

**Predictive power.** The labels come from the oracle (near the top of the quality axis — the hackathon graph's "lift after post-training"). If Observation 2 holds, the classifier keeps most of that power. So MGD targets the empty upper-left region of the cost-quality plot: **cheap and predictive**, because the expensive part is paid once, offline, and reused.

### Mapping to the hackathon's required metrics

- **Task-level metric, expensive:** the metagradient goodness score `s_i = −τ_i` (model-dependent oracle).
- **Task-level metric, cheap:** the classifier's predicted score `ŝ_i ≈ s_i`.
- **Dataset-level metric:** aggregate predicted scores over a cohort (mean `ŝ`, or fraction of high-scoring sequences) to predict that cohort's lift.
- **Pareto evaluation:** plot predictive power (Spearman ρ vs held-out lift) against compute cost for random / length / perplexity / classifier / truncated-metagradient / full-metagradient / full-train.

---

## 7. Implementation spec for the coding agent

### Recommended stack

- **Models:** start with **GPT-2 small (124M)** for speed (matches the paper's multi-step metagradient setting); scale to **Llama-3.2-1B** if time permits.
- **Metagradient capability needed:** differentiate through a short **Adam** inner loop w.r.t. per-sequence loss weights. Implementation options, in order of accessibility:
  1. Engstrom et al. (2025) metagradient tooling if directly usable.
  2. PyTorch `torch.func` (functional params) + a hand-written differentiable Adam unrolled for `T` steps; gradient-checkpoint the unroll if memory is tight.
  3. The `higher` library (differentiable optimizers).
  4. JAX (what the paper used) if the agent prefers it.
  Keep `T` small so the unrolled graph fits in memory.
- **Featurizer/classifier:** sentence-transformers (frozen) + LightGBM/XGBoost or a small MLP.
- **Eval:** Eleuther LM eval harness for perplexity/benchmarks; scipy for Spearman/R².
- **Compute:** 8×H100 single node is ample for GPT-2 + short inner loops.

### Dataset design (so success is measurable)

Construct `D` as a **mixture of clearly-good and clearly-noisy data** relative to a target val set (e.g. good = target-domain/clean text such as Wikipedia or clean C4; bad = off-domain or corrupted web text). Set `Φ` = LM loss on a held-out **target-domain** val split. This gives a ground-truth sanity check: a working method should preferentially select the good cluster. Keep `M` small enough (e.g. 50k–500k sequences) that the **full-metagradient oracle is computable once** for comparison.

### Suggested starting config

| Knob | Symbol | Start | Ablate over |
|---|---|---|---|
| Sequence length | `T_seq` | 512 | 256, 1024 |
| Corpus size (sequences) | `M` | 100k | — |
| Sequences per metagrad round | `k` | 512 | 256, 1024 |
| Inner training steps | `T` | 16 | 1, 8, 32, 96 |
| Labeling rounds | `R` | set so each labeled seq appears ~3–5× | — |
| Inner optimizer | — | **Adam** | SGD (expect worse) |
| Final token budget | `n` | top 10% of tokens | 1%, 5%, 25% |

### Repo layout

```
metagrad-distill/
  README.md
  configs/                 # yaml for each phase
  src/
    data/                  # load corpus, chunk to sequences, build target val set, build good/bad mixture
    metagrad/              # inner loop A, differentiable Adam, metagradient_scores()
    labeling/              # Phase 1 orchestration: sample, run metagrad, zscore, average, persist labels
    classifier/            # Phase 2: featurize, train regressor, save model
    select/                # Phase 3: score full corpus, optional dedup, pick top-n
    train_final/           # Phase 4: CPT on selected tokens, eval
    baselines/             # random, perplexity-top, perplexity-correlation, DSIR-style, oracle
    eval/                  # spearman(classifier vs oracle), pareto plot, cohort-lift prediction
  scripts/
    run_label.py  run_classifier.py  run_select.py  run_final.py  run_pareto.py
```

### Build order (hand to the agent as a checklist)

1. Data pipeline: chunking, target val set, good/bad mixture, sequence index/store.
2. `metagrad_scores()` on a tiny toy (10 sequences, `T=4`) — verify gradients flow to `w`. **This is the riskiest component; build and unit-test it first.**
3. Phase-1 labeling loop with z-scoring + cross-round averaging; persist `(seq_id, label)`.
4. **Full-metagradient oracle** over all `M` (small-scale only) — the gold standard to validate against.
5. Featurizer + regressor; report Spearman ρ between `ŝ` and oracle on held-out sequences (**H1**).
6. Phase-3 corpus scoring + top-`n` selection (+ optional dedup).
7. Phase-4 final CPT + eval; baselines; results table.
8. Ablations (`T`, features, `k`) and the Pareto plot.
9. Hackathon cohort experiment (§8) for lift prediction.

---

## 8. Experiments & hypotheses

State each as a falsifiable hypothesis with a held-out check.

- **H1 — Faithfulness.** A cheap classifier predicts oracle metagradient scores: high Spearman ρ between `ŝ` and `s` on held-out sequences.
- **H2 — Truncation transfers.** Rankings from short inner loops match full-length ones: high ρ between truncated-`T` and full-`T` metagradient scores. Find the smallest `T` that preserves the ranking.
- **H3 — Downstream win.** Training on top-`n` by classifier beats random / perplexity-top / perplexity-correlation on final target eval, and approaches the full-metagradient oracle.
- **H4 — Pareto.** On predictive-power-vs-cost axes, the classifier lands cheap-and-predictive (upper-left). Produce the plot.
- **H5 — Cohort lift (hackathon protocol).** Build several cohorts (vary one property at a time), CPT each, measure lift = `acc_after − acc_before`. The dataset-level aggregate of `ŝ` predicts held-out cohort lift better than cheap baselines (report held-out R² / RMSE / ρ, with the scatter).

**Baselines:** random; perplexity-top (low base-model perplexity); perplexity-correlation (Thrush et al., 2025); DSIR-style domain matching; full-metagradient oracle (small scale); full-corpus upper bound.

---

## 9. Risks, mitigations, stretch goals

**Risks & mitigations**

- *Metagradient label noise (batch/state dependence).* → z-score within batch + average across rounds; reset model each round.
- *Truncation bias.* → ablate `T` (H2); pick smallest `T` that preserves ranking.
- *Memory of the unrolled inner loop.* → small model, small `T`, gradient checkpointing.
- *Weak features → classifier can't distill.* → ablate featurizers; embeddings as strong default.
- *Selection collapse to near-duplicates.* → dedup/diversity penalty on top-`n`.
- *Adam-vs-SGD pitfall.* → use Adam inside `A` (paper finding); keep SGD as a negative control.

**Stretch goals**

- Swap `Φ` from generic val loss to a **benchmark/task loss** → targeted, capability-specific selection.
- **Active labeling:** use classifier uncertainty to choose which sequences to compute metagradients for next (a labeling-efficiency loop).
- **RL extension (the hackathon's literal ask):** make `A` an RL loop and `Φ` post-RL eval, so the same distillation predicts post-RL lift per cohort. Harder (metagradients through RL), flagged as future work.

---

## 10. One-paragraph pitch

The expensive, near-oracle signal for data quality — the metagradient of a downstream metric w.r.t. each sequence's training weight — is too costly to compute over a whole corpus. MetaGrad-Distill computes it on a small sample using short proxy training runs, distills it into a cheap classifier, and uses that classifier to score and select from the entire corpus. The result is a data-quality metric that is cheap to evaluate yet carries oracle-grade predictive power: a single point that breaks the cost-quality Pareto frontier the hackathon asks us to chart.
