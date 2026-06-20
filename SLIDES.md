# MetaGrad-Distill — 3-Slide Outline

*A cheap data-quality metric distilled from an expensive metagradient oracle.*

---

## Slide 1 — The question: which data is worth training on?

- Under a fixed token budget, **which sequences should we train on** to best improve a target metric Φ (e.g. held-out loss on a target domain)?
- We want a **score per sequence** that is *cheap to compute* yet *predicts training value*.
- Cheap heuristics (length, perplexity, domain match) are weak proxies. The truly faithful signal is expensive — that's the tension we attack.

**Teaching point:** data selection = find a score `s_i` that ranks sequences by how much they help Φ.

---

## Slide 2 — The theory: the metagradient *is* the oracle

- Train with a **per-sequence weighted loss** `Σ_i w_i · ℓ(x_i)`, all `w_i = 1`.
- The **metagradient** measures how nudging one sequence's weight moves the target:

  `τ_i = ∂Φ / ∂w_i`  →  goodness  `s_i = −τ_i`

  computed by **backpropagating through a short inner training run** (differentiate through Adam steps). Lower τ = up-weighting it lowers Φ = better data.
- This is **near-oracle** but costs ~a full training run over the whole corpus — too expensive to run everywhere.

**Teaching point:** "is this data good?" becomes a *gradient* of the target through training — exact, but pricey.

---

## Slide 3 — The trick: pay once, predict everywhere

1. Compute true metagradient scores on a **small sample** (short proxy runs).
2. **Distill** them into a cheap classifier: `features → predicted score`.
3. Run the classifier over the **whole corpus** (forward passes only); keep the top-`n`.

- The expensive oracle is paid **once, offline**, then **amortized** into a reusable scorer.
- Result: a point on the **upper-left of the cost–quality Pareto frontier** — *cheap to evaluate, yet inherits oracle-grade predictive power.*

**Teaching point:** distillation turns a one-time expensive signal into a cheap, corpus-wide metric — breaking the cost↔quality trade-off.
