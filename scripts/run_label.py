"""Driver: shard the labeling rounds across all 8 GPUs (one process each), then
merge into final per-sequence labels.

Usage:
  python -m scripts.run_label --total_rounds 800 --k 256 --T 16 --L_inner 128 \
      --tag main --wandb
"""
import os, sys, json, time, argparse, subprocess
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total_rounds", type=int, required=True)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--L_inner", type=int, default=128)
    ap.add_argument("--val_bs", type=int, default=128)
    ap.add_argument("--loss_clip", type=float, default=0.0)
    ap.add_argument("--data_dir", default="artifacts/data/mgd_v1")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--n_gpus", type=int, default=8)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    out_dir = f"artifacts/labels/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    per = -(-args.total_rounds // args.n_gpus)  # ceil
    procs, shard_paths = [], []
    for g in range(args.n_gpus):
        sp = os.path.join(out_dir, f"shard{g}.npz")
        shard_paths.append(sp)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                   XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   XLA_PYTHON_CLIENT_ALLOCATOR="platform",
                   HF_HUB_DISABLE_PROGRESS_BARS="1")
        cmd = [sys.executable, "-m", "src.labeling.label",
               "--out_path", sp, "--data_dir", args.data_dir,
               "--n_rounds", str(per), "--k", str(args.k), "--T", str(args.T),
               "--lr", str(args.lr), "--L_inner", str(args.L_inner),
               "--val_bs", str(args.val_bs), "--seed", str(1000 + g),
               "--loss_clip", str(args.loss_clip)]
        if args.wandb:
            cmd += ["--wandb", "--wandb_group", f"labeling-{args.tag}"]
        log = open(os.path.join(out_dir, f"shard{g}.log"), "w")
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
        print(f"launched shard {g} ({per} rounds) on GPU {g}", flush=True)

    t0 = time.time()
    rc = [p.wait() for p in procs]
    print(f"all shards done in {time.time()-t0:.0f}s, rc={rc}", flush=True)
    if any(rc):
        print("WARNING: some shards failed; check logs", flush=True)

    from src.labeling.label import merge_shards
    ok = [sp for sp, c in zip(shard_paths, rc) if c == 0 and os.path.exists(sp)]
    merge_shards(ok, os.path.join(out_dir, "labels.npz"), args.data_dir)
    json.dump(vars(args), open(os.path.join(out_dir, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
