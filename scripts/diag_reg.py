"""Does inner-loop regularisation de-bias the metagradient's bad preference for
HARD data (results.md §2.11-2.12)? Sweep weight-decay (wd) and horizon T on the
difficulty corpus; report the per-cluster mean z-scored score. Baseline (wd=0,T=16):
easy -0.19 / mid 0.00 / hard +0.19 -> prefers hard (which loses to random). We want
a config that flattens the hard-bias (ideally mid >= hard, since mid~random wins).
"""
import os, json, argparse, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import numpy as np, csv
from src.metagrad import model_gpt2 as M
from src.metagrad.metagrad import metagrad_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="artifacts/data/mgd_diff")
    ap.add_argument("--wd", type=float, required=True)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--L_inner", type=int, default=128)
    ap.add_argument("--n_rounds", type=int, default=40)
    ap.add_argument("--val_bs", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    params, cfg = M.load_pretrained("gpt2")
    tok = np.load(os.path.join(a.data_dir, "tokens.npy")).astype(np.int32)
    val = np.load(os.path.join(a.data_dir, "val.npy")).astype(np.int32)
    clusters = np.array([r[1] for r in list(csv.reader(open(os.path.join(a.data_dir, "meta.csv"))))[1:]])
    rng = np.random.default_rng(0); Mtot = len(tok)
    sum_s = np.zeros(Mtot); cnt = np.zeros(Mtot); phis = []
    t0 = time.time()
    for r in range(a.n_rounds):
        idx = rng.choice(Mtot, size=a.k, replace=False)
        s, phi = metagrad_scores(params, tok[idx], val, cfg, T=a.T, lr=a.lr,
                                 val_bs=a.val_bs, L_inner=a.L_inner, wd=a.wd)
        z = (s - s.mean()) / (s.std() + 1e-8)
        sum_s[idx] += z; cnt[idx] += 1; phis.append(float(phi))
    lab = np.where(cnt > 0, sum_s / np.maximum(cnt, 1), np.nan); cov = cnt > 0
    cm = {c: float(np.nanmean(lab[(clusters == c) & cov])) for c in ["easy", "mid", "hard"]}
    top = max(cm, key=cm.get)
    out = dict(wd=a.wd, T=a.T, lr=a.lr, k=a.k, n_rounds=a.n_rounds,
               mean_phi=round(float(np.mean(phis)), 3), cluster_mean=cm,
               top_cluster=top, hard_minus_mid=round(cm["hard"] - cm["mid"], 4),
               sec=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
