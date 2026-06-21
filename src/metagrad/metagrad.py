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


@partial(jax.jit, static_argnums=(3, 4), static_argnames=("optimizer", "remat_blocks"))
def _phi_and_metagrad(params0, seqs, val, cfg, T, lr=1e-3, b1=0.9, b2=0.999,
                      eps=1e-8, optimizer="adam", remat_blocks=False, wd=0.0, loss_clip=0.0, loss_pow=1.0):
    """Returns (phi, tau) where tau = d phi / d w at w=1.

    ``wd`` adds AdamW-style decoupled weight decay to the inner loop. ``loss_clip``
    caps each sequence's per-example loss before the weighted sum, so high-loss
    (hard/atypical) examples can't dominate the inner-loop gradient by magnitude --
    a targeted fix for the metagradient over-valuing hard data (§2.11-2.13)."""
    k = seqs.shape[0]
    w0 = jnp.ones(k, jnp.float32)

    def phi_of_w(w):
        def wloss(p):
            pe = M.loss_per_example(p, seqs, cfg, remat_blocks)          # [k]
            cap = jnp.where(loss_clip > 0, loss_clip, jnp.inf)           # traced-safe
            pe = jnp.minimum(pe, cap)                                    # hard cap (no-op if clip<=0)
            # soft compression pe**loss_pow (loss_pow<1 de-emphasises high-loss/hard
            # examples *gently*, vs the hard clip — de-bias without flattening). 1.0=off.
            pe = jnp.where(loss_pow != 1.0, jnp.power(jnp.maximum(pe, 1e-6), loss_pow), pe)
            return jnp.sum(w * pe) / k

        def step(carry, t):
            p, m, v = carry
            g = jax.grad(wloss)(p)
            if optimizer == "sgd":
                p = tree_map(lambda p_, g_: p_ - lr * (g_ + wd * p_), p, g)
                return (p, m, v), None
            t1 = t + 1.0
            m = tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
            v = tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, v, g)
            bc1 = 1 - b1 ** t1
            bc2 = 1 - b2 ** t1
            # eps INSIDE the sqrt: d/dx sqrt(x) -> inf at x=0, so for params with
            # zero grad (v=0) the metagradient (a 2nd-order grad) would be 0*inf=NaN.
            # AdamW decoupled decay (wd*p) added to the update.
            p = tree_map(lambda p_, m_, v_:
                         p_ - lr * ((m_ / bc1) / jnp.sqrt(v_ / bc2 + eps) + wd * p_), p, m, v)
            return (p, m, v), None

        carry0 = (params0, _zeros_like(params0), _zeros_like(params0))
        (pT, _, _), _ = jax.lax.scan(jax.checkpoint(step), carry0,
                                     jnp.arange(T, dtype=jnp.float32))
        return M.loss_mean(pT, val, cfg, remat_blocks)

    phi, tau = jax.value_and_grad(phi_of_w)(w0)
    return phi, tau


@partial(jax.jit, static_argnums=(3, 4))
def _phi_and_metagrad_gn(params0, seqs, val, cfg, T, lr=3e-5, b1=0.9, b2=0.999, eps=1e-8):
    """Gradient-normalized metagradient (§2.14): inner update is
    Σ wᵢ·(∇ℓᵢ/‖∇ℓᵢ‖) — normalize each example's gradient DIRECTION to unit norm so
    high-loss/hard examples can't dominate by magnitude, while keeping directional
    value. Needs per-example grads (vmap(grad)) → small k only. Returns (phi, tau)."""
    k = seqs.shape[0]
    w0 = jnp.ones(k, jnp.float32)

    def one_loss(p, seq):
        return M.loss_per_example(p, seq[None], cfg, False)[0]

    def phi_of_w(w):
        def step(carry, t):
            p, m, v = carry
            g_each = jax.vmap(lambda s: jax.grad(one_loss)(p, s))(seqs)         # [k,*param] leaves
            sq = tree_map(lambda L: (L.reshape(k, -1) ** 2).sum(1), g_each)      # [k] per leaf
            norm = jnp.sqrt(sum(jax.tree_util.tree_leaves(sq)) + 1e-12)          # [k] global per-ex
            g_each = tree_map(lambda L: L / norm.reshape([k] + [1]*(L.ndim-1)), g_each)
            g = tree_map(lambda L: jnp.tensordot(w, L, axes=([0], [0])) / k, g_each)
            t1 = t + 1.0
            m = tree_map(lambda m_, g_: b1*m_ + (1-b1)*g_, m, g)
            v = tree_map(lambda v_, g_: b2*v_ + (1-b2)*g_*g_, v, g)
            p = tree_map(lambda p_, m_, v_:
                         p_ - lr*(m_/(1-b1**t1))/jnp.sqrt(v_/(1-b2**t1)+eps), p, m, v)
            return (p, m, v), None
        carry0 = (params0, _zeros_like(params0), _zeros_like(params0))
        (pT, _, _), _ = jax.lax.scan(jax.checkpoint(step), carry0, jnp.arange(T, dtype=jnp.float32))
        return M.loss_mean(pT, val, cfg, False)

    return jax.value_and_grad(phi_of_w)(w0)


@partial(jax.jit, static_argnums=(4, 5))
def _phi_and_metagrad_gn_redu(params0, seqs, val, vw, cfg, T, lr=3e-5, b1=0.9, b2=0.999, eps=1e-8):
    """gradnorm inner loop (bias #1) + reducibility-weighted val Φ (bias #2, §2.15):
    Φ = Σ_x vw_x · L_after(x), vw concentrated on typical/learnable val. vw: [V]."""
    k = seqs.shape[0]; w0 = jnp.ones(k, jnp.float32)

    def one_loss(p, seq):
        return M.loss_per_example(p, seq[None], cfg, False)[0]

    def phi_of_w(w):
        def step(carry, t):
            p, m, v = carry
            g_each = jax.vmap(lambda s: jax.grad(one_loss)(p, s))(seqs)
            sq = tree_map(lambda L: (L.reshape(k, -1) ** 2).sum(1), g_each)
            norm = jnp.sqrt(sum(jax.tree_util.tree_leaves(sq)) + 1e-12)
            g_each = tree_map(lambda L: L / norm.reshape([k] + [1]*(L.ndim-1)), g_each)
            g = tree_map(lambda L: jnp.tensordot(w, L, axes=([0], [0])) / k, g_each)
            t1 = t + 1.0
            m = tree_map(lambda m_, g_: b1*m_ + (1-b1)*g_, m, g)
            v = tree_map(lambda v_, g_: b2*v_ + (1-b2)*g_*g_, v, g)
            p = tree_map(lambda p_, m_, v_:
                         p_ - lr*(m_/(1-b1**t1))/jnp.sqrt(v_/(1-b2**t1)+eps), p, m, v)
            return (p, m, v), None
        carry0 = (params0, _zeros_like(params0), _zeros_like(params0))
        (pT, _, _), _ = jax.lax.scan(jax.checkpoint(step), carry0, jnp.arange(T, dtype=jnp.float32))
        return jnp.sum(vw * M.loss_per_example(pT, val, cfg, False))

    return jax.value_and_grad(phi_of_w)(w0)


@partial(jax.jit, static_argnums=(4, 5), static_argnames=("optimizer", "remat_blocks"))
def phi_at_w(params0, seqs, val, w, cfg, T, lr=1e-3, b1=0.9, b2=0.999,
             eps=1e-8, optimizer="adam", remat_blocks=False):
    """Φ(w): run the SAME inner loop at an explicit weight vector w (not w=1).
    Used to finite-difference ∂Φ/∂w_i and check it against the autodiff τ."""
    k = seqs.shape[0]

    def wloss(p):
        pe = M.loss_per_example(p, seqs, cfg, remat_blocks)
        return jnp.sum(w * pe) / k

    def step(carry, t):
        p, m, v = carry
        g = jax.grad(wloss)(p)
        if optimizer == "sgd":
            return (tree_map(lambda p_, g_: p_ - lr * g_, p, g), m, v), None
        t1 = t + 1.0
        m = tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
        v = tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, v, g)
        p = tree_map(lambda p_, m_, v_:
                     p_ - lr * (m_ / (1 - b1 ** t1)) / jnp.sqrt(v_ / (1 - b2 ** t1) + eps),
                     p, m, v)
        return (p, m, v), None

    carry0 = (params0, _zeros_like(params0), _zeros_like(params0))
    (pT, _, _), _ = jax.lax.scan(jax.checkpoint(step), carry0, jnp.arange(T, dtype=jnp.float32))
    return M.loss_mean(pT, val, cfg, remat_blocks)


def metagrad_scores(params0, seqs, val, cfg, T=16, lr=1e-3, optimizer="adam",
                    val_bs=256, L_inner=None, remat_blocks=False, wd=0.0, loss_clip=0.0,
                    loss_pow=1.0, gradnorm=False, redu_vw=None):
    """seqs [k,L] int, val [V,L] int -> (s [k], phi float).

    s_i = -tau_i, higher = better (training on i lowers target loss more).
    L_inner truncates the per-sequence length used in BOTH the inner loop and
    Phi (logits are [k, L, vocab] — a major memory cost that L_inner cuts).
    ``wd`` = AdamW weight decay; ``loss_clip`` caps per-example loss (§2.13 levers).
    ``gradnorm`` = per-example gradient-direction normalization (§2.14; small k only).
    """
    seqs = jnp.asarray(seqs, jnp.int32)
    val = jnp.asarray(val[:val_bs], jnp.int32)
    if L_inner is not None:
        seqs = seqs[:, :L_inner]
        val = val[:, :L_inner]
    if gradnorm and redu_vw is not None:
        phi, tau = _phi_and_metagrad_gn_redu(params0, seqs, val, jnp.asarray(redu_vw, jnp.float32),
                                             cfg, int(T), float(lr))
    elif gradnorm:
        phi, tau = _phi_and_metagrad_gn(params0, seqs, val, cfg, int(T), float(lr))
    else:
        phi, tau = _phi_and_metagrad(params0, seqs, val, cfg, int(T), float(lr),
                                     optimizer=optimizer, remat_blocks=remat_blocks,
                                     wd=float(wd), loss_clip=float(loss_clip), loss_pow=float(loss_pow))
    return -np.asarray(tau, np.float64), float(phi)
