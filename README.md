# MetaGrad-Distill (MGD)

Distill an expensive **metagradient** data-quality oracle into a **cheap classifier** for training-data selection in continued pretraining (CPT).

The oracle score for a sequence is `s_i = −∂Φ/∂w_i`: how much up-weighting that sequence's training loss would *lower* a downstream target metric `Φ` (held-out LM loss on a target domain), computed by differentiating through a short Adam inner-training loop. Computing it for a whole corpus is as expensive as training on everything; instead we compute it on a small sample, distill it into a regressor over cheap features, and score the full corpus with forward passes only.

See [`design_doc.md`](design_doc.md) for the full design, [`TODO.md`](TODO.md) for the build checklist, and [`results.md`](results.md) for results.

## Repo layout
```
src/
  data/        load corpus, chunk to sequences, build good/bad mixture + target val
  metagrad/    differentiable inner loop + metagradient_scores()   (JAX)
  labeling/    Phase-1 labeling: sample, run metagrad, zscore, average, persist
  classifier/  featurize + train regressor
  select/      score full corpus, pick top-n
  train_final/ CPT on selected tokens + eval
  baselines/   random, perplexity, perplexity-correlation, DSIR, oracle
  eval/        spearman, pareto, cohort-lift
scripts/       run_*.py entry points
configs/       yaml configs per phase
artifacts/     run outputs (gitignored where large)
```

## Environments
- **`/root/jax-env`** — JAX 0.10.2 + flax + optax. Used for metagradients. `JAX` because differentiating through training is far easier/cheaper here.
- **`/root/ai-env`** — PyTorch 2.10 (cu128) + transformers + vLLM. Used for eval, featurization, final CPT.

See [`ENV.md`](ENV.md) for the why behind versions (driver-570 / cu128 constraint).

## References
- Engstrom et al. 2025, *Optimizing ML Training with Metagradient Descent* (REPLAY), arXiv:2503.13751
- Thrush et al. 2026, *Synthetic Data for any Differentiable Target* (DPG), arXiv:2604.08423
- Gururangan et al. 2020, *Don't Stop Pretraining* (DAPT)
