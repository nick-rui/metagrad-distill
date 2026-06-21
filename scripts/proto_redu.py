"""Bias #2 probe: reducible-loss Φ (+ gradnorm for bias #1). The hard-lean persists
because Φ = plain held-out loss rewards data that lowers loss on the HARD val tail
fast (myopically). Reducible-loss idea (RHO-LOSS, Mindermann 2022): weight the
target toward val content the base model already handles — the *reducible/learnable*
part — and down-weight the irreducibly-hard tail. Then the metagradient should reward
data that improves typical content (mid) rather than data that overfits the hard tail.

  Φ = Σ_x vw_x · L_after(x) / Σ_x vw_x ,  vw_x = softmax(−base_loss(x) / temp)
  temp→∞ : uniform (= plain Φ).  temp small : focus on the easiest/most-typical val.

Reuses gradnorm (per-example grad-norm) so bias #1 is already handled. Reports per-cluster
oracle preference on mgd_diff. Goal: hard−mid -> ≤0 (prefer mid, which wins downstream).
"""
import os, json, argparse, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import numpy as np, csv
import jax, jax.numpy as jnp
from functools import partial
from src.metagrad import model_gpt2 as M

tree_map = jax.tree_util.tree_map
def _zeros_like(t): return tree_map(jnp.zeros_like, t)


@partial(jax.jit, static_argnums=(4,))
def metagrad_redu(params0, seqs, val, vw, cfg, T, lr=3e-5, b1=0.9, b2=0.999, eps=1e-8):
    """gradnorm inner loop + reducibility-weighted val Φ. vw: [V] per-val weights (sum to 1)."""
    k = seqs.shape[0]; w0 = jnp.ones(k, jnp.float32)
    def one_loss(p, seq): return M.loss_per_example(p, seq[None], cfg, False)[0]
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
        per_val = M.loss_per_example(pT, val, cfg, False)        # [V]
        return jnp.sum(vw * per_val)                              # reducibility-weighted Φ
    return jax.value_and_grad(phi_of_w)(w0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="artifacts/data/mgd_diff")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--L_inner", type=int, default=128)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--val_bs", type=int, default=64)
    ap.add_argument("--n_rounds", type=int, default=40)
    ap.add_argument("--temp", type=float, default=0.5)     # softmax temp over -base_val_loss; big=uniform
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    params, cfg = M.load_pretrained("gpt2")
    tok = np.load(os.path.join(a.data_dir, "tokens.npy")).astype(np.int32)
    valnp = np.load(os.path.join(a.data_dir, "val.npy")).astype(np.int32)[:a.val_bs, :a.L_inner]
    val = jnp.asarray(valnp)
    clusters = np.array([r[1] for r in list(csv.reader(open(os.path.join(a.data_dir, "meta.csv"))))[1:]])
    # base loss per val point -> reducibility weights (favor low-base-loss = typical/learnable)
    base_val = np.asarray(jax.jit(lambda p, x: M.loss_per_example(p, x, cfg, False))(params, val))
    z = (-base_val / max(a.temp, 1e-6)); z -= z.max()
    vw = jnp.asarray(np.exp(z) / np.exp(z).sum())
    print(f"vw concentration: max={float(vw.max()):.3f} (uniform={1/a.val_bs:.3f}); "
          f"base_val easy..hard range {base_val.min():.2f}..{base_val.max():.2f}", flush=True)

    rng = np.random.default_rng(0); Mtot = len(tok)
    sum_s = np.zeros(Mtot); cnt = np.zeros(Mtot); phis = []; t0 = time.time()
    for r in range(a.n_rounds):
        idx = rng.choice(Mtot, size=a.k, replace=False)
        seqs = jnp.asarray(tok[idx][:, :a.L_inner])
        phi, tau = metagrad_redu(params, seqs, val, vw, cfg, a.T, lr=a.lr)
        s = -np.asarray(tau, np.float64); zz = (s - s.mean())/(s.std()+1e-8)
        sum_s[idx] += zz; cnt[idx] += 1; phis.append(float(phi))
        if r == 0: print(f"compiled+round0 {time.time()-t0:.0f}s phi={phi:.4f}", flush=True)
    lab = np.where(cnt > 0, sum_s/np.maximum(cnt, 1), np.nan); cov = cnt > 0
    cm = {c: float(np.nanmean(lab[(clusters == c) & cov])) for c in ["easy", "mid", "hard"]}
    out = dict(temp=a.temp, k=a.k, T=a.T, n_rounds=a.n_rounds, mean_phi=round(float(np.mean(phis)), 4),
               cluster_mean=cm, top_cluster=max(cm, key=cm.get),
               hard_minus_mid=round(cm["hard"]-cm["mid"], 4), spread=round(cm["hard"]-cm["easy"], 4))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2); print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
