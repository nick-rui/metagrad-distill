"""H2 truncation, robustly: score each inner-loop length T in its OWN process so
compiled-graph memory doesn't accumulate across T (the single-process version
OOMs at T=8 under the platform allocator). Worker saves the per-batch score
array for one T; driver spawns one worker per T, then correlates each T's
ranking against the longest reference T.

  driver:  python -m scripts.h2_truncation --Ts 1 2 4 8 16 --k 32 --n_batches 12
  worker:  python -m scripts.h2_truncation --worker --T 8 --scores_out <npy> ...
"""
import os, sys, json, argparse, subprocess
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import numpy as np
from scipy.stats import spearmanr


def _batches(M_total, k, n_batches, seed):
    rng = np.random.default_rng(seed)
    return [rng.choice(M_total, size=k, replace=False) for _ in range(n_batches)]


def worker(a):
    from src.metagrad import model_gpt2 as M
    from src.metagrad.metagrad import metagrad_scores
    tok = np.load(os.path.join(a.data_dir, "tokens.npy"))
    val = np.load(os.path.join(a.data_dir, "val.npy")).astype(np.int32)
    params0, cfg = M.load_pretrained()
    batches = _batches(len(tok), a.k, a.n_batches, a.seed)
    arr = np.full((a.n_batches, a.k), np.nan)
    for bi, idx in enumerate(batches):
        s, _ = metagrad_scores(params0, tok[idx].astype(np.int32), val, cfg,
                               T=a.T, lr=a.lr, val_bs=a.val_bs, L_inner=a.L_inner,
                               optimizer=a.optimizer)
        arr[bi] = s
    np.save(a.scores_out, arr)
    print(f"[worker T={a.T}] saved {a.scores_out}", flush=True)


def driver(a):
    os.makedirs(os.path.dirname(a.out_path) or ".", exist_ok=True)
    tmp = os.path.join(os.path.dirname(a.out_path) or ".", "_h2_scores")
    os.makedirs(tmp, exist_ok=True)
    scores = {}
    for T in a.Ts:
        sp = os.path.join(tmp, f"T{T}.npy")
        cmd = [sys.executable, "-m", "scripts.h2_truncation", "--worker", "--T", str(T),
               "--scores_out", sp, "--k", str(a.k), "--n_batches", str(a.n_batches),
               "--lr", str(a.lr), "--L_inner", str(a.L_inner), "--val_bs", str(a.val_bs),
               "--optimizer", a.optimizer, "--data_dir", a.data_dir, "--seed", str(a.seed)]
        r = subprocess.run(cmd, env=dict(os.environ))
        if r.returncode == 0 and os.path.exists(sp):
            scores[T] = np.load(sp)
            print(f"scored T={T}", flush=True)
        else:
            print(f"T={T} worker failed rc={r.returncode}", flush=True)

    Tref = max(scores) if scores else max(a.Ts)
    rho_vs_ref = {}
    for T in a.Ts:
        if T not in scores or Tref not in scores:
            rho_vs_ref[T] = None; continue
        rhos = [spearmanr(scores[T][bi], scores[Tref][bi]).statistic
                for bi in range(a.n_batches)
                if np.isfinite(scores[T][bi]).all() and np.isfinite(scores[Tref][bi]).all()]
        rho_vs_ref[T] = float(np.nanmean(rhos)) if rhos else None
    res = dict(Ts=list(a.Ts), Tref=Tref, k=a.k, n_batches=a.n_batches, lr=a.lr,
               L_inner=a.L_inner, optimizer=a.optimizer, spearman_vs_ref=rho_vs_ref)
    json.dump(res, open(a.out_path, "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--scores_out", default=None)
    ap.add_argument("--Ts", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--n_batches", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--L_inner", type=int, default=128)
    ap.add_argument("--val_bs", type=int, default=128)
    ap.add_argument("--optimizer", default="adam")
    ap.add_argument("--data_dir", default="artifacts/data/mgd_v1")
    ap.add_argument("--out_path", default="artifacts/ablation/truncation.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    worker(a) if a.worker else driver(a)
