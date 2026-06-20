"""Run final CPT for every selection method, one per GPU in parallel, then build
the report (H3 table + Pareto + efficiency curves). Each method trains with an
identical budget/schedule so the wandb val-ppl-vs-tokens curves are comparable.
"""
import os, sys, glob, time, argparse, subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select_dir", required=True)      # artifacts/select/<tag>
    ap.add_argument("--data_dir", default="artifacts/data/mgd_v1")
    ap.add_argument("--out_dir", required=True)         # artifacts/cpt/<tag>
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--eval_every_tokens", type=int, default=150_000)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--include_full", action="store_true")
    args = ap.parse_args()

    sel_files = sorted(glob.glob(os.path.join(args.select_dir, "*.npz")))
    methods = {os.path.splitext(os.path.basename(f))[0]: f for f in sel_files}
    if not args.include_full:
        methods.pop("full", None)
    if args.methods:
        methods = {m: methods[m] for m in args.methods if m in methods}
    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.path.basename(args.out_dir.rstrip("/"))

    procs = []
    for gi, (m, sel) in enumerate(methods.items()):
        gpu = gi % 8
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false")
        cmd = [sys.executable, "-m", "src.train_final.cpt", "--sel", sel,
               "--data_dir", args.data_dir, "--out_dir", args.out_dir, "--method", m,
               "--epochs", str(args.epochs), "--lr", str(args.lr), "--bs", str(args.bs),
               "--eval_every_tokens", str(args.eval_every_tokens),
               "--wandb_group", f"cpt-{tag}"]
        log = open(os.path.join(args.out_dir, f"{m}.log"), "w")
        procs.append((m, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
        print(f"launched CPT[{m}] on GPU {gpu}", flush=True)
        time.sleep(3)

    rc = {m: p.wait() for m, p in procs}
    print("CPT done:", rc, flush=True)
    from src.eval.report import make_report
    make_report(args.out_dir, args.out_dir.replace("cpt", "report"))


if __name__ == "__main__":
    main()
