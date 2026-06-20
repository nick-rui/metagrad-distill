# Results — MetaGrad-Distill (MGD)

> Clean, curated results log. Each section states a hypothesis, the setup, the number, and the takeaway.
> Raw run artifacts live in `artifacts/`. This file is the human-readable summary of record.

**Project:** Distill an expensive metagradient data-quality oracle into a cheap classifier for data selection in continued pretraining (CPT). See `design_doc.md`.

**Status:** 🚧 In progress — started 2026-06-20.

---

## 0. Setup & environment

| Component | Choice |
|---|---|
| Metagradient engine | JAX 0.10.2 + optax (unrolled differentiable Adam, `jax.grad`, `remat`) |
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

### 2.0 Sanity checks
- _pending: metagradient unit test (gradient flow + sign sanity)._

---

## 3. Notes, surprises, decisions
- 2026-06-20: Project kickoff. Metagradients in JAX (per requirement); direct unrolled Adam since `T` is small (REPLAY reserved for scaling).
