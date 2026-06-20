"""H5 driver: build cohorts, CPT each (waves of 8 across GPUs), then correlate
aggregate predicted score vs measured lift.
"""
import os, sys, glob, time, argparse, subprocess
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="artifacts/data/mgd_v1")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--features", default="artifacts/features/mgd_v1.npz")
    ap.add_argument("--cohort_dir", default="artifacts/cohorts/sel")
    ap.add_argument("--cpt_dir", default="artifacts/cohorts/cpt")
    ap.add_argument("--n_per", type=int, default=2000)
    ap.add_argument("--n_cohorts", type=int, default=14)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()

    from src.eval.cohorts import build_cohorts, correlate
    build_cohorts(args.data_dir, args.pred, args.cohort_dir, args.n_per, args.n_cohorts)
    os.makedirs(args.cpt_dir, exist_ok=True)
    sel_files = sorted(glob.glob(os.path.join(args.cohort_dir, "cohort*.npz")))

    # run in waves of 8 GPUs
    for w in range(0, len(sel_files), 8):
        wave = sel_files[w:w + 8]
        procs = []
        for gi, sel in enumerate(wave):
            m = os.path.splitext(os.path.basename(sel))[0]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gi),
                       HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false",
                       WANDB_MODE="disabled")  # many runs; keep wandb for main H3
            cmd = [sys.executable, "-m", "src.train_final.cpt", "--sel", sel,
                   "--data_dir", args.data_dir, "--out_dir", args.cpt_dir, "--method", m,
                   "--epochs", str(args.epochs), "--lr", str(args.lr),
                   "--eval_every_tokens", "500000", "--wandb_group", "cohorts"]
            log = open(os.path.join(args.cpt_dir, f"{m}.log"), "w")
            procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
            time.sleep(2)
        rc = [p.wait() for p in procs]
        print(f"wave {w//8} done rc={rc}", flush=True)

    correlate(args.cohort_dir, args.cpt_dir, args.features, args.data_dir,
              "artifacts/cohorts/h5.json")


if __name__ == "__main__":
    main()
