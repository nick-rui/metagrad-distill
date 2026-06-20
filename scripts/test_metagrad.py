"""Unit test for the metagradient core (the riskiest component).

Checks, on a tiny random model so it runs in seconds:
  1. gradients actually flow to w (tau is finite, non-zero, shape [k]);
  2. sign sanity: sequences that match the val distribution score HIGHER than
     random/corrupt sequences (training on them lowers val loss more).
"""
import numpy as np, jax, jax.numpy as jnp
from src.metagrad import model_gpt2 as M
from src.metagrad.metagrad import metagrad_scores

def main():
    rng = np.random.default_rng(0)
    V, L = 64, 16
    cfg = M.GPT2Config(vocab=V, n_ctx=L, d=32, n_layer=2, n_head=2)
    params0 = M.init_random(cfg, jax.random.PRNGKey(0))

    # val = a few fixed "target" sequences with structure (repeating motif)
    motif = np.array([3, 8, 1, 20, 5, 14, 2, 9], dtype=np.int32)
    def make(n, kind):
        out = []
        for _ in range(n):
            if kind == "good":      # tiles the val motif -> training reduces val loss
                s = np.tile(motif, L // len(motif) + 1)[:L]
            elif kind == "corrupt": # random tokens
                s = rng.integers(0, V, size=L)
            out.append(s)
        return np.asarray(out, np.int32)

    val = make(8, "good")
    good = make(6, "good")
    corrupt = make(6, "corrupt")
    seqs = np.concatenate([good, corrupt], axis=0)
    labels = ["good"] * len(good) + ["corrupt"] * len(corrupt)

    s, phi = metagrad_scores(params0, seqs, val, cfg, T=4, lr=0.05, val_bs=8)

    s = np.asarray(s)
    print(f"phi(val loss @ w=1) = {phi:.4f}")
    print(f"tau finite: {np.isfinite(s).all()}  | s.shape={s.shape}  | |s|>0: {np.abs(s).sum()>0}")
    sg = s[:len(good)].mean(); sc = s[len(good):].mean()
    print(f"mean s good={sg:+.4e}   mean s corrupt={sc:+.4e}   (want good > corrupt)")
    for lab, val_ in zip(labels, s):
        print(f"  {lab:8} s={val_:+.4e}")

    ok = (np.isfinite(s).all() and np.abs(s).sum() > 0 and s.shape == (len(seqs),)
          and sg > sc)
    print("\nUNIT TEST:", "PASS" if ok else "FAIL")
    assert ok
    return ok

if __name__ == "__main__":
    main()
