"""Baseline per-sequence selection scores (higher = more likely to be selected),
all cheap, computed from features the corpus already has. Compared against the
metagradient-distilled classifier and the oracle.

  random          : control.
  length          : #non-eos tokens (trivial heuristic).
  ppl_top         : prefer LOW base-model perplexity (fluent text)  -> score=-base_loss.
  domain_match    : DSIR/contrastive proxy — cosine of seq feature to the target
                    (val) feature centroid. Prefers on-target-domain text.
  ppl_corr        : perplexity-correlation proxy (Thrush 2025) — rank text by how
                    much LOWER its base ppl is than the corpus median *and* close to
                    the target ppl band; cheap stand-in, labelled as a proxy.
  oracle          : the metagradient labels themselves (gold upper bound for selection).
  classifier      : trained regressor predictions (our method).
"""
from __future__ import annotations
import numpy as np


def random_scores(M, seed=0):
    return np.random.default_rng(seed).standard_normal(M)


def length_scores(tok, eos_id=50256):
    return (tok != eos_id).sum(axis=1).astype(np.float64)


def ppl_top_scores(base_loss):
    return -np.asarray(base_loss, np.float64)


def domain_match_scores(feats, val_feats):
    F = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    c = val_feats.mean(0); c = c / (np.linalg.norm(c) + 1e-8)
    return F @ c


def ppl_corr_scores(base_loss, val_base_loss):
    """Proxy for perplexity-correlation: prefer sequences whose base-model loss
    sits in the target (val) loss band — i.e. text the base model finds as
    (un)surprising as the target domain. Score = -|loss - target_median|."""
    target = np.median(val_base_loss)
    return -np.abs(np.asarray(base_loss, np.float64) - target)


def all_baseline_scores(tok, feats, base_loss, val_feats, val_base_loss, seed=0):
    return dict(
        random=random_scores(len(tok), seed),
        length=length_scores(tok),
        ppl_top=ppl_top_scores(base_loss),
        domain_match=domain_match_scores(feats, val_feats),
        ppl_corr=ppl_corr_scores(base_loss, val_base_loss),
    )
