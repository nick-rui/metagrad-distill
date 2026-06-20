# MetaGrad-Distill: what works, and how we got there

A narrative companion to `results.md` (which is the numbers-of-record) and
`design_doc.md` (the original plan). This doc explains the **mechanism that
works** — distilling an expensive metagradient oracle into a cheap classifier —
how we *built* it, and how we *evaluated* it. The honest limits are in §7 and in
`results.md` §2.11–2.13; this doc is about the parts that hold up.

---

## 1. The idea in one paragraph

To pick the best `n` tokens to train on, you want a per-sequence "is this good
training data?" score. The **gold signal** is the *metagradient*: differentiate a
downstream target metric Φ (held-out PubMed loss) with respect to each training
sequence's loss weight `wᵢ`. `sᵢ = −∂Φ/∂wᵢ` says exactly how much up-weighting
sequence `i` would lower the target. It's near-oracle quality — but computing it
for a whole corpus costs as much as training on everything. So we compute it on a
**small sample**, **distill** it into a cheap regressor over forward-pass features,
and run that regressor over the whole corpus. That's MetaGrad-Distill (MGD).

The thing that works: **the cheap classifier reproduces the expensive oracle**
(ρ up to 0.76 held-out) and selects nearly as well as it (within ~1% of the
oracle's downstream lift on cluster corpora). The thing that doesn't (§7): that
selection doesn't beat the *best* cheap baselines.

---

## 2. The oracle: computing `sᵢ = −∂Φ/∂wᵢ`

The oracle is a **second-order** quantity — a gradient through training. For a
sampled batch of `k` sequences:

```
w = ones(k)                      # per-sequence loss weights, start at 1
model = base GPT-2               # reset every round ("A is stateless")
for t in 1..T:                   # short Adam inner loop, T=16
    per_ex = lm_loss_per_example(model, batch)   # [k]
    loss   = sum(w * per_ex) / k
    model  = adam_step(model, grad(loss, model)) # every step differentiable in w
Φ = lm_loss(model, held_out_PubMed)              # scalar, differentiable in w
τ = grad(Φ, w)                                   # the metagradient, length k
s = −τ                                           # higher = better
```

Implementation choices that made this work (JAX, `/root/jax-env`):
- **Differentiate through the optimizer.** The whole inner loop is one
  differentiable function of `w`; we unroll it with `lax.scan` + `jax.checkpoint`
  (rematerialization) so the reverse-mode graph fits in memory.
- **Hand-rolled differentiable Adam**, not SGD (the DPG paper's finding: Adam +
  several steps gives the signal teeth; single-step SGD collapses to influence
  functions).
- **Two numerical fixes** that were the difference between NaN and working:
  `eps` goes *inside* `sqrt(v + eps)` (else zero-grad params give `0·∞ = NaN` in
  the second-order grad), and the attention mask uses `−1e9` (not `finfo.min`) for
  a stable 2nd-order gradient.
- **GPT-2 hand-written in JAX** because transformers v5 dropped Flax.

### Is the oracle actually correct? (We checked.)
A second-order autodiff is easy to get subtly wrong. We validated it against a
**black-box finite difference** (`scripts/validate_oracle_fd.py`): re-run the
*same* inner loop at `w = 1 ± ε·eᵢ`, central-difference Φ, and compare to the
autodiff τ. Result: **Spearman 0.982, sign-agreement 94%.** The metagradient
genuinely is `∂Φ/∂wᵢ` — every downstream result rests on real ground.

### The label that the classifier learns
A single round's `sᵢ` is noisy (it depends on the batch and the model state). So
labels are **group-relative + averaged** (`src/labeling/label.py`):
1. **Z-score** `s` within each round (removes the per-batch offset — the same
   normalization GRPO uses).
2. Each sequence appears in **~4 rounds**; its label is the **mean** of its
   z-scored scores.

Over 3200 rounds across 8×H100 (~22 min), this turns a noisy per-round signal
into a stable, transferable target. Sanity: on the good/off/corrupt corpus the
averaged label's top-10% is **99.5% on-target** (Cohen's d = +1.94).

> **The lr lesson (why this works at all).** The inner-loop learning rate is
> decisive. At the original `lr=1e-3`, 16 Adam steps *destroy* the model
> (val loss 3.4 → 6.7) and the metagradient becomes noise — corrupt data scored
> *highest*. At **`lr=3e-5`** the inner loop stays near the base model and the
> oracle cleanly recovers good > off-domain > corrupt (Cohen's d flips from −0.02
> to +1.75). "Short proxy runs are enough" — but only if they're *stable*.

---

## 3. Distillation: matching the oracle with a cheap classifier (this is the win)

This is the part that works. We never compute the oracle for the whole corpus;
instead we learn `f: features → ŝ` on the small labeled sample and run `f`
everywhere.

**Features (cheap, forward-only):** the base GPT-2 **mean-pooled final hidden
state** (768-d) per sequence — one forward pass, no training, no optimizer
(`src/classifier/featurize.py`).

**Regressor:** **LightGBM** trained to predict the averaged oracle label from
those 768 features (`src/classifier/train.py`). Trees over a frozen embedding;
training is seconds on CPU.

**How well it matches (H1, held-out 20% of labeled sequences):**

| corpus | Spearman ρ(ŝ, oracle s) | what it shows |
|---|---|---|
| good/off/corrupt | **0.72** | distillation works |
| all-PubMed clean/repetitive/noised | **0.76** | works *even when surface cues mislead* |

The second row is the striking one. On that corpus, `noised` data has a 0.995
feature-cosine to the target (looks almost identical) yet is low-value — and the
classifier, from **features alone**, assigns it the oracle's low score (−0.95).
The subtle feature signal the oracle cares about *is* learnable; the cheap model
recovers it.

**Why it can match the oracle at all:** the oracle's value signal correlates with
properties that a frozen embedding already encodes (domain, fluency, structure).
The metagradient is expensive to *compute*, but what it measures is, to a large
degree, a *learnable function of cheap features*. That's the whole bet, and for
ranking purposes it pays off (ρ ≈ 0.7–0.76).

---

## 4. How we evaluated (the protocol)

Everything is judged by a single, end-to-end **downstream** test — not by the
proxy score:

```
score every sequence  →  keep top 10% of tokens  →  CPT GPT-2 on that subset
                       →  measure held-out PubMed perplexity improvement
```

Lower final perplexity = the selection chose better data. We run the *identical*
budget and schedule (3 epochs, lr 3e-5, fixed token budget) for every method so
the only difference is *which data was selected* (`scripts/run_cpt_all.py`,
`src/train_final/cpt.py`, torch `/root/ai-env`).

**The methods we compare** (each scores a sequence; we keep the top 10%):

| method | score | role |
|---|---|---|
| random | noise | floor |
| length | # tokens | trivial control |
| ppl_top | −base_loss (lowest perplexity) | "clean text" heuristic |
| ppl_corr | −\|loss − target_band\| | perplexity-correlation (Thrush 2025) |
| domain_match | cosine(features, target centroid) | DSIR-style — the strong baseline |
| **oracle** | true metagradient s | expensive gold reference |
| **classifier** | ŝ (LightGBM) | **ours** |

**Why this comparison is the right one:** `oracle` is the signal we wish we could
afford everywhere; `classifier` is our cheap copy. The five baselines are the
"can't a one-forward-pass trick do just as well?" challengers. Two questions:
**H1** — does `classifier ≈ oracle`? **H3** — do they beat the baselines?

**Corpora as traps.** We didn't just use one dataset; each corpus is engineered to
*fool a specific baseline*, so we can tell whether the metagradient captures
something they miss:
- `mgd_v1` (good PubMed / off-domain C4 / corrupt shuffled) — basic discrimination.
- `mgd_hard` (clean / repetitive[lowest-ppl] / noised[clean-features]) — fools
  `ppl_top` and `domain_match` by construction.
- `mgd_diff` (all clean PubMed, stratified by difficulty) — `domain_match` is
  blind, `ppl_top` picks the easy stuff.

**Rigor.** Single runs hide noise, so the key comparisons are **5-seed** with
paired significance (`scripts/multiseed_cpt.py`): across-seed std ≈ 0.02, so we can
say which differences are real and which are noise.

---

## 5. What works — the evidence

**(a) Distillation is faithful (H1).** ρ = 0.72–0.76 held-out; the classifier
reproduces the oracle's per-cluster ordering almost exactly, including the
counter-intuitive `noised` case.

**(b) The classifier selects as well as the oracle (H3, cluster corpora).** On
`mgd_v1`: classifier **+8.02** ppl vs oracle **+8.09** — it captures **99%** of the
oracle's downstream lift at a forward-pass cost. On `mgd_hard`: classifier **+7.65**
≈ oracle **+7.68**.

**(c) It beats the naive baselines, sometimes dramatically.** On `mgd_hard`,
`ppl_top` is *catastrophically* fooled — it selects 100% of the lowest-perplexity
`repetitive` junk and CPT makes the model **−53.9** ppl *worse* — while MGD is
robust (+7.65). MGD reliably beats random, length, and perplexity-only selection.

**(d) The oracle is mechanically real.** Finite-difference ρ = 0.982 (§2).

**(e) The learned value-function transfers (amortization).** A classifier trained
on `mgd_v1`'s labels, applied to the *different* `mgd_hard` corpus, still selects
96.8% clean and gets **+7.58** (≈ native +7.65). You pay the oracle **once** and
score new corpora cheaply — the economic premise of the method.

**(f) Using the score as importance-sampling weights is competitive.** Instead of a
hard top-n cut, sampling each batch ∝ a relu-gated ŝ gives **+7.73**, edging the
hard cut (+7.65) — though naive loss-weighting loses (`scripts/cpt_soft.py`).

---

## 6. The stack that made it fast

- **Two isolated envs** (their CUDA libs conflict): `/root/jax-env` (JAX 0.10.2 +
  optax) for metagradients — differentiating through an optimizer is first-class
  in JAX; `/root/ai-env` (torch 2.10 cu128) for eval, features, final CPT.
- **8×H100**, single node. Labeling shards rounds across all 8 GPUs.
- **flash-hog** higher-order attention kernel integrated as an opt-in backend
  (numerically faithful, ρ=0.9999 vs XLA); TK kernels vendored for later.
- Driver-570 / CUDA-12.8 constrained the whole stack (see `ENV.md`).

---

## 7. The honest boundary (so this doc isn't propaganda)

What works is **distillation and faithfulness**: a real oracle, cheaply and
accurately reproduced, transferable across corpora. What does **not** hold is the
strong claim that this *beats the best cheap baseline*:
- When good data forms a feature-identifiable cluster, `domain_match` ties MGD —
  cheap feature-matching already finds it.
- On `mgd_diff`, where the metagradient's *unique* edge should show, it
  mis-fires: it prefers "hard" high-loss data that doesn't actually train better,
  and the classifier (and even the oracle) **significantly underperform random**
  (`results.md` §2.11–2.12).

We traced *why* (§2.13): the short-horizon metagradient is biased by **gradient
magnitude** (hard examples have big gradients), so it conflates "high-gradient"
with "valuable." Weight decay doesn't fix it; aggressive **loss-clipping** does
move the preference back toward typical data — the active line of work.

**Bottom line:** the *machinery* works and is validated end-to-end. The *value
proposition* (a cheap score that beats one-forward-pass baselines at data
selection) is not yet demonstrated, and the path to it runs through (a) de-biasing
the metagradient's magnitude sensitivity and (b) a target Φ where value isn't
already captured by surface similarity.
