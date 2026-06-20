"""Foundational check: is the autodiff metagradient τ_i = ∂Φ/∂w_i actually the
real sensitivity of the target to each sequence's weight? We verify against a
black-box finite difference — re-run the SAME inner loop at w = 1 ± ε·e_i and
central-difference Φ — which makes no assumption about the autodiff being correct.

If τ (autodiff) ≈ τ_fd (finite diff), the oracle measures real training value;
that is the assumption every downstream result rests on.

  python -m scripts.validate_oracle_fd --k 24 --T 16 --eps 0.05 --n_probe 16
"""
import os, json, argparse
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import numpy as np, jax, jax.numpy as jnp
from scipy.stats import spearmanr, pearsonr
from src.metagrad import model_gpt2 as M
from src.metagrad.metagrad import metagrad_scores, phi_at_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="artifacts/data/mgd_v1")
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--L_inner", type=int, default=128)
    ap.add_argument("--val_bs", type=int, default=128)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--n_probe", type=int, default=16)   # how many coords to finite-diff
    ap.add_argument("--out", default="artifacts/ablation/oracle_fd.json")
    a = ap.parse_args()

    params, cfg = M.load_pretrained("gpt2")
    tok = np.load(os.path.join(a.data_dir, "tokens.npy")).astype(np.int32)
    val = np.load(os.path.join(a.data_dir, "val.npy")).astype(np.int32)[:a.val_bs]
    rng = np.random.default_rng(0)
    seqs = tok[rng.choice(len(tok), size=a.k, replace=False)][:, :a.L_inner]
    valL = val[:, :a.L_inner]

    # autodiff metagradient (s = -tau -> tau = -s)
    s, _ = metagrad_scores(params, seqs, val, cfg, T=a.T, lr=a.lr, val_bs=a.val_bs, L_inner=a.L_inner)
    tau_ad = -np.asarray(s, np.float64)

    seqsj = jnp.asarray(seqs); valj = jnp.asarray(valL); ones = jnp.ones(a.k, jnp.float32)
    phi = lambda w: float(phi_at_w(params, seqsj, valj, w, cfg, a.T, a.lr))
    probe = list(range(min(a.n_probe, a.k)))
    tau_fd = np.zeros(len(probe))
    for j, i in enumerate(probe):
        wp = ones.at[i].add(a.eps); wm = ones.at[i].add(-a.eps)
        tau_fd[j] = (phi(wp) - phi(wm)) / (2 * a.eps)     # central diff ≈ dΦ/dw_i
        print(f"  coord {i}: autodiff τ={tau_ad[i]:+.4e}  finite-diff τ={tau_fd[j]:+.4e}", flush=True)

    ad = tau_ad[probe]
    res = dict(k=a.k, T=a.T, lr=a.lr, eps=a.eps, n_probe=len(probe),
               spearman=float(spearmanr(ad, tau_fd).statistic),
               pearson=float(pearsonr(ad, tau_fd).statistic),
               rel_l2=float(np.linalg.norm(ad - tau_fd) / (np.linalg.norm(ad) + 1e-12)),
               sign_agree=float(np.mean(np.sign(ad) == np.sign(tau_fd))))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)
    print("PASS" if res["spearman"] > 0.9 and res["sign_agree"] > 0.9 else "WEAK", flush=True)


if __name__ == "__main__":
    main()
