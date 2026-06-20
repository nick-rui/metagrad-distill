"""PROTOTYPE (small-k): per-example GRADIENT normalization in the inner loop.

The §2.13 finding: loss-magnitude levers (clip, pow) de-bias only by flattening the
signal, because hard examples dominate via gradient MAGNITUDE. The principled fix is
to normalize each example's gradient to unit norm *before* combining — so the inner
update is  sum_i w_i * (grad_i / ||grad_i||)  — removing the magnitude confound while
keeping each example's gradient DIRECTION (its actual training value). The metagradient
d Phi/d w then scores directional value, not gradient size.

Cost: per-example grads = vmap(grad) over k sequences = k param-gradient trees per inner
step, differentiated twice. Only fits at small k / L_inner / T. This prototype answers:
does gradient-normalization de-bias the oracle on mgd_diff (hard-mid -> 0) WHILE keeping
a real per-cluster spread (i.e. without the flattening that killed H1 under clipping)?

  python -m scripts.proto_gradnorm --k 8 --L_inner 64 --T 8 --n_rounds 20 --normalize 1
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


@partial(jax.jit, static_argnums=(3, 4), static_argnames=("normalize",))
def metagrad_gn(params0, seqs, val, cfg, T, lr=3e-5, b1=0.9, b2=0.999, eps=1e-8,
                normalize=True):
    k = seqs.shape[0]
    w0 = jnp.ones(k, jnp.float32)

    def one_loss(p, seq):                      # scalar loss for a single sequence
        return M.loss_per_example(p, seq[None], cfg, False)[0]

    def phi_of_w(w):
        def step(carry, t):
            p, m, v = carry
            # per-example gradients: [k, *param] for every leaf
            g_each = jax.vmap(lambda s: jax.grad(one_loss)(p, s))(seqs)
            if normalize:
                sq = tree_map(lambda L: (L.reshape(k, -1) ** 2).sum(1), g_each)   # [k] per leaf
                norm = jnp.sqrt(sum(jax.tree_util.tree_leaves(sq)) + 1e-12)        # [k] global per-ex
                g_each = tree_map(lambda L: L / norm.reshape([k] + [1]*(L.ndim-1)), g_each)
            # weighted sum over examples -> param-shaped grad
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

    phi, tau = jax.value_and_grad(phi_of_w)(w0)
    return phi, tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="artifacts/data/mgd_diff")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--L_inner", type=int, default=64)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--val_bs", type=int, default=64)
    ap.add_argument("--n_rounds", type=int, default=20)
    ap.add_argument("--normalize", type=int, default=1)   # 1=gradnorm, 0=plain (control)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    params, cfg = M.load_pretrained("gpt2")
    tok = np.load(os.path.join(a.data_dir, "tokens.npy")).astype(np.int32)
    val = jnp.asarray(np.load(os.path.join(a.data_dir, "val.npy")).astype(np.int32)[:a.val_bs, :a.L_inner])
    clusters = np.array([r[1] for r in list(csv.reader(open(os.path.join(a.data_dir, "meta.csv"))))[1:]])
    rng = np.random.default_rng(0); Mtot = len(tok)
    sum_s = np.zeros(Mtot); cnt = np.zeros(Mtot); phis = []
    t0 = time.time()
    for r in range(a.n_rounds):
        idx = rng.choice(Mtot, size=a.k, replace=False)
        seqs = jnp.asarray(tok[idx][:, :a.L_inner])
        phi, tau = metagrad_gn(params, seqs, val, cfg, a.T, lr=a.lr, normalize=bool(a.normalize))
        s = -np.asarray(tau, np.float64)
        z = (s - s.mean()) / (s.std() + 1e-8)
        sum_s[idx] += z; cnt[idx] += 1; phis.append(float(phi))
        if r == 0: print(f"compiled+round0 in {time.time()-t0:.0f}s phi={phi:.3f}", flush=True)
    lab = np.where(cnt > 0, sum_s/np.maximum(cnt, 1), np.nan); cov = cnt > 0
    cm = {c: float(np.nanmean(lab[(clusters == c) & cov])) for c in ["easy", "mid", "hard"]}
    out = dict(normalize=bool(a.normalize), k=a.k, L_inner=a.L_inner, T=a.T, n_rounds=a.n_rounds,
               mean_phi=round(float(np.mean(phis)), 3), cluster_mean=cm,
               top_cluster=max(cm, key=cm.get), hard_minus_mid=round(cm["hard"]-cm["mid"], 4),
               spread=round(cm["hard"]-cm["easy"], 4), sec=round(time.time()-t0, 1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
