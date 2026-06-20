"""Metagradient scoring: differentiate a held-out target loss Phi w.r.t.
per-sequence training weights, by backpropagating through a short Adam inner loop.

    tau_i = d Phi / d w_i   at w = 1      (metagradient)
    s_i   = -tau_i                        (goodness; Phi is a loss we minimise)

Inner loop is a hand-rolled, fully-differentiable Adam (matching the DPG paper's
"use Adam, not SGD" finding). The unrolled step is wrapped in `jax.checkpoint`
(rematerialisation) to keep the reverse-mode graph affordable. For large T this
is where REPLAY (Engstrom 2025) would slot in; at our small T direct unroll fits.
"""
from __future__ import annotations
from functools import partial
import numpy as np
import jax, jax.numpy as jnp
from . import model_gpt2 as M

tree_map = jax.tree_util.tree_map


def _zeros_like(tree):
    return tree_map(jnp.zeros_like, tree)


@partial(jax.jit, static_argnums=(3, 4), static_argnames=("optimizer",))
def _phi_and_metagrad(params0, seqs, val, cfg, T, lr=1e-3, b1=0.9, b2=0.999,
                      eps=1e-8, optimizer="adam"):
    """Returns (phi, tau) where tau = d phi / d w at w=1."""
    k = seqs.shape[0]
    w0 = jnp.ones(k, jnp.float32)

    def phi_of_w(w):
        def wloss(p):
            pe = M.loss_per_example(p, seqs, cfg)          # [k]
            return jnp.sum(w * pe) / k

        def step(carry, t):
            p, m, v = carry
            g = jax.grad(wloss)(p)
            if optimizer == "sgd":
                p = tree_map(lambda p_, g_: p_ - lr * g_, p, g)
                return (p, m, v), None
            t1 = t + 1.0
            m = tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
            v = tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, v, g)
            bc1 = 1 - b1 ** t1
            bc2 = 1 - b2 ** t1
            # eps INSIDE the sqrt: d/dx sqrt(x) -> inf at x=0, so for params with
            # zero grad (v=0) the metagradient (a 2nd-order grad) would be 0*inf=NaN.
            p = tree_map(lambda p_, m_, v_:
                         p_ - lr * (m_ / bc1) / jnp.sqrt(v_ / bc2 + eps), p, m, v)
            return (p, m, v), None

        carry0 = (params0, _zeros_like(params0), _zeros_like(params0))
        (pT, _, _), _ = jax.lax.scan(jax.checkpoint(step), carry0,
                                     jnp.arange(T, dtype=jnp.float32))
        return M.loss_mean(pT, val, cfg)

    phi, tau = jax.value_and_grad(phi_of_w)(w0)
    return phi, tau


def metagrad_scores(params0, seqs, val, cfg, T=16, lr=1e-3, optimizer="adam",
                    val_bs=256):
    """seqs [k,L] int, val [V,L] int -> (s [k], phi float).

    s_i = -tau_i, higher = better (training on i lowers target loss more).
    """
    seqs = jnp.asarray(seqs, jnp.int32)
    val = jnp.asarray(val[:val_bs], jnp.int32)
    phi, tau = _phi_and_metagrad(params0, seqs, val, cfg, int(T), float(lr),
                                 optimizer=optimizer)
    return -np.asarray(tau, np.float64), float(phi)
