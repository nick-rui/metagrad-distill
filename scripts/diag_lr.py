"""Diagnostic: is the inner-loop lr in a sane regime, and does the metagradient
recover the ground-truth cluster ordering (good > offdomain > corrupt)?

For one lr: run N rounds of metagrad scoring on real corpus batches, z-score
within round, accumulate per-cluster. Report mean phi (inner-loop val loss; base
GPT-2 PubMed ~3.4 — if phi >> that, the inner loop is diverging) and per-cluster
mean z-scored s with Cohen's d(good-corrupt).
"""
import os, json, argparse, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import numpy as np, csv
from src.metagrad import model_gpt2 as M
from src.metagrad.metagrad import metagrad_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--L_inner", type=int, default=128)
    ap.add_argument("--n_rounds", type=int, default=30)
    ap.add_argument("--val_bs", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    params, cfg = M.load_pretrained("gpt2")
    tok = np.load("artifacts/data/mgd_v1/tokens.npy").astype(np.int32)
    val = np.load("artifacts/data/mgd_v1/val.npy").astype(np.int32)
    meta = list(csv.reader(open("artifacts/data/mgd_v1/meta.csv")))[1:]
    clusters = np.array([r[1] for r in meta])

    rng = np.random.default_rng(0)
    M_total = tok.shape[0]
    sum_s = np.zeros(M_total); cnt = np.zeros(M_total); phis = []
    t0 = time.time()
    for r in range(a.n_rounds):
        idx = rng.choice(M_total, size=a.k, replace=False)
        s, phi = metagrad_scores(params, tok[idx], val, cfg, T=a.T, lr=a.lr,
                                 val_bs=a.val_bs, L_inner=a.L_inner)
        z = (s - s.mean()) / (s.std() + 1e-8)
        sum_s[idx] += z; cnt[idx] += 1; phis.append(float(phi))
    lab = np.where(cnt > 0, sum_s / np.maximum(cnt, 1), np.nan)
    cov = cnt > 0
    cm = {c: float(np.nanmean(lab[(clusters == c) & cov])) for c in ["good", "offdomain", "corrupt"]}
    g = lab[(clusters == 'good') & cov]; cr = lab[(clusters == 'corrupt') & cov]
    pooled = np.sqrt((np.nanvar(g) + np.nanvar(cr)) / 2) + 1e-9
    out = dict(lr=a.lr, T=a.T, k=a.k, n_rounds=a.n_rounds,
               base_pubmed_val_loss=3.45,
               mean_phi=round(float(np.mean(phis)), 3),
               phi_first=round(phis[0], 3), phi_last=round(phis[-1], 3),
               cluster_mean=cm,
               cohens_d_good_minus_corrupt=round((np.nanmean(g) - np.nanmean(cr)) / pooled, 4),
               covered=int(cov.sum()), sec=round(time.time() - t0, 1))
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
