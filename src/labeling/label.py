"""Phase-1 labeling: turn metagradients into stable per-sequence labels.

Each round: reset model to base init (A is stateless), sample k sequences,
compute metagradient goodness s = -tau, z-score within the round (removes the
per-batch offset, GRPO-style group-relative normalisation), accumulate into a
running per-sequence mean. A sequence appears in ~C rounds, so its label is the
mean of C z-scored scores -> a stable, transferable target for the classifier.

Sharded across GPUs by running one process per device over a disjoint slice of
rounds, then merging the running sums.
"""
from __future__ import annotations
import os, json, time, argparse
from collections import defaultdict
import numpy as np


def _save(out_path, sum_s, cnt, phis, meta, done_rounds):
    np.savez(out_path, sum_s=sum_s, cnt=cnt, phis=np.asarray(phis),
             done_rounds=done_rounds, meta=json.dumps(meta))


def run_shard(out_path, data_dir, n_rounds, k, T, lr, L_inner, val_bs, seed,
              base="gpt2", wandb_run=None, log_every=10, ckpt_every=25, loss_clip=0.0,
              gradnorm=False, subset=0):
    import jax
    from src.metagrad import model_gpt2 as M
    from src.metagrad.metagrad import metagrad_scores

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tok = np.load(os.path.join(data_dir, "tokens.npy"))      # uint16 [M,L]
    val = np.load(os.path.join(data_dir, "val.npy")).astype(np.int32)
    M_total = tok.shape[0]
    n_pool = subset if subset > 0 else M_total              # restrict sampling to a subset (small-k coverage)
    params0, cfg = M.load_pretrained(base)
    meta = dict(n_rounds=n_rounds, k=k, T=T, lr=lr, L_inner=L_inner,
                val_bs=val_bs, seed=seed, base=base)

    rng = np.random.default_rng(seed)
    sum_s = np.zeros(M_total, np.float64)
    cnt = np.zeros(M_total, np.int32)
    phis = []
    # resume from checkpoint if present
    start = 0
    if os.path.exists(out_path):
        try:
            d = np.load(out_path, allow_pickle=True)
            sum_s, cnt = d["sum_s"], d["cnt"]
            phis = list(d["phis"]); start = int(d["done_rounds"])
            # advance rng to keep sampling stream consistent
            for _ in range(start):
                rng.choice(n_pool, size=k, replace=False)
            print(f"[shard {seed}] resuming from round {start}", flush=True)
        except Exception as e:
            print(f"[shard {seed}] ckpt load failed ({e}); fresh start", flush=True)
    t_start = time.time()
    for r in range(start, n_rounds):
        idx = rng.choice(n_pool, size=k, replace=False)
        seqs = tok[idx].astype(np.int32)
        s, phi = metagrad_scores(params0, seqs, val, cfg, T=T, lr=lr,
                                 val_bs=val_bs, L_inner=L_inner, loss_clip=loss_clip, gradnorm=gradnorm)
        z = (s - s.mean()) / (s.std() + 1e-8)         # within-round normalisation
        sum_s[idx] += z
        cnt[idx] += 1
        phis.append(float(phi))
        if wandb_run is not None and (r % log_every == 0):
            wandb_run.log({"round": r, "phi": phi, "rounds_per_s":
                           (r + 1) / (time.time() - t_start),
                           "covered_seqs": int((cnt > 0).sum())})
        if r % 25 == 0:
            print(f"[shard {seed}] round {r}/{n_rounds} phi={phi:.4f} "
                  f"covered={int((cnt>0).sum())} "
                  f"{(r+1-start)/(time.time()-t_start):.2f} rounds/s", flush=True)
        if (r + 1) % ckpt_every == 0:
            _save(out_path, sum_s, cnt, phis, meta, r + 1)   # periodic durable checkpoint
    _save(out_path, sum_s, cnt, phis, meta, n_rounds)
    print(f"[shard {seed}] done -> {out_path} ({time.time()-t_start:.0f}s)", flush=True)


def merge_shards(shard_paths, out_path, data_dir):
    """Combine per-shard running sums into final labels."""
    sum_s = cnt = None
    phis = []
    for p in shard_paths:
        d = np.load(p, allow_pickle=True)
        sum_s = d["sum_s"] if sum_s is None else sum_s + d["sum_s"]
        cnt = d["cnt"] if cnt is None else cnt + d["cnt"]
        phis.append(d["phis"])
    label = np.where(cnt > 0, sum_s / np.maximum(cnt, 1), np.nan)
    np.savez(out_path, label=label, cnt=cnt)
    # attach cluster for analysis
    import csv
    meta = list(csv.reader(open(os.path.join(data_dir, "meta.csv"))))[1:]
    clusters = np.array([row[1] for row in meta])
    covered = cnt > 0
    summary = dict(
        n_labeled=int(covered.sum()), M=int(len(label)),
        mean_cnt=float(cnt[covered].mean()), min_cnt=int(cnt[covered].min()),
        max_cnt=int(cnt[covered].max()),
        label_by_cluster={c: float(np.nanmean(label[(clusters == c) & covered]))
                          for c in sorted(set(clusters.tolist()))},
    )
    json.dump(summary, open(out_path.replace(".npz", "_summary.json"), "w"), indent=2)
    print("merged labels:", json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--data_dir", default="artifacts/data/mgd_v1")
    ap.add_argument("--n_rounds", type=int, required=True)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--L_inner", type=int, default=128)
    ap.add_argument("--val_bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss_clip", type=float, default=0.0)
    ap.add_argument("--gradnorm", action="store_true")
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_group", default=None)
    args = ap.parse_args()
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="metagrad-distill", group=args.wandb_group or "labeling",
                         name=f"label-shard{args.seed}", config=vars(args))
    run_shard(args.out_path, args.data_dir, args.n_rounds, args.k, args.T, args.lr,
              args.L_inner, args.val_bs, args.seed, wandb_run=run, loss_clip=args.loss_clip,
              gradnorm=args.gradnorm, subset=args.subset)
    if run is not None:
        run.finish()
