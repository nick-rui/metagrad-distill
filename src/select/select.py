"""Phase-3: select the top-n sequences under a token budget, by any per-seq score.

Used uniformly for every method (classifier pred, oracle labels, perplexity, ...)
so selection is apples-to-apples. Optional dedup penalty discourages near-duplicate
picks (cosine on features).
"""
from __future__ import annotations
import numpy as np


def select_topn(scores, t_seq, budget_frac, n_total_tokens=None, higher_better=True,
                feats=None, dedup_thresh=None):
    """scores [M] (nan = ineligible). Returns selected seq indices for the budget."""
    scores = np.asarray(scores, np.float64)
    M = len(scores)
    tokens_per = t_seq
    total = (n_total_tokens if n_total_tokens is not None else M * tokens_per)
    budget_tokens = int(budget_frac * total)
    n_pick = max(1, budget_tokens // tokens_per)

    order = np.argsort(-scores if higher_better else scores)
    order = order[np.isfinite(scores[order])]

    if dedup_thresh is None or feats is None:
        return order[:n_pick]

    # greedy dedup: skip a candidate too similar to an already-picked one
    F = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    picked, picked_F = [], []
    for i in order:
        if len(picked) >= n_pick:
            break
        if picked_F and np.max(F[i] @ np.asarray(picked_F).T) > dedup_thresh:
            continue
        picked.append(i); picked_F.append(F[i])
    return np.asarray(picked)
