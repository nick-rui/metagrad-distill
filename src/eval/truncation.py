"""H2 — does a short inner loop preserve the ranking of a long one?
Sample fixed batches; score each at several T; correlate truncated-T rankings
against the longest reference T. Also Adam-vs-SGD control at fixed T.
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
from scipy.stats import spearmanr


def run(data_dir, out_path, Ts=(1, 2, 4, 8, 16), k=32, n_batches=12, lr=1e-3,
        L_inner=128, val_bs=128, optimizer="adam", seed=0, wandb_run=None):
    from src.metagrad import model_gpt2 as M
    from src.metagrad.metagrad import metagrad_scores
    tok = np.load(os.path.join(data_dir, "tokens.npy"))
    val = np.load(os.path.join(data_dir, "val.npy")).astype(np.int32)
    params0, cfg = M.load_pretrained()
    rng = np.random.default_rng(seed)
    batches = [rng.choice(len(tok), size=k, replace=False) for _ in range(n_batches)]

    # scores[T] = [n_batches, k]
    scores = {}
    for T in Ts:
        arr = np.full((n_batches, k), np.nan)
        for bi, idx in enumerate(batches):
            try:
                s, _ = metagrad_scores(params0, tok[idx].astype(np.int32), val, cfg,
                                       T=T, lr=lr, val_bs=val_bs, L_inner=L_inner,
                                       optimizer=optimizer)
                arr[bi] = s
            except Exception as e:
                print(f"T={T} batch={bi} failed: {type(e).__name__}", flush=True)
        scores[T] = arr
        print(f"scored T={T}", flush=True)

    Tref = max(Ts)
    rho_vs_ref = {}
    for T in Ts:
        rhos = [spearmanr(scores[T][bi], scores[Tref][bi]).correlation
                for bi in range(n_batches)
                if np.isfinite(scores[T][bi]).all() and np.isfinite(scores[Tref][bi]).all()]
        rho_vs_ref[T] = float(np.nanmean(rhos)) if rhos else None
        if wandb_run is not None and rho_vs_ref[T] is not None:
            wandb_run.log({"T": T, "spearman_vs_ref": rho_vs_ref[T]})
    res = dict(Ts=list(Ts), Tref=Tref, k=k, n_batches=n_batches, optimizer=optimizer,
               spearman_vs_ref=rho_vs_ref)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="artifacts/data/mgd_v1")
    ap.add_argument("--out_path", default="artifacts/ablation/truncation.json")
    ap.add_argument("--Ts", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--n_batches", type=int, default=12)
    ap.add_argument("--optimizer", default="adam")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    run_ = None
    if args.wandb:
        import wandb
        run_ = wandb.init(project="metagrad-distill", group="ablation-truncation",
                          name=f"trunc-{args.optimizer}", config=vars(args))
    run(args.data_dir, args.out_path, tuple(args.Ts), args.k, args.n_batches,
        optimizer=args.optimizer, wandb_run=run_)
    if run_: run_.finish()
