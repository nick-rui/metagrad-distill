"""Multi-seed CPT for error bars + significance. Runs each method's CPT at several
seeds (seed varies batch order/dropout), aggregates mean±std, and tests whether the
classifier's improvement is *significantly* different from each baseline (paired by
seed). Turns single-run "≈"/"ties" statements into falsifiable ones.

  python -m scripts.multiseed_cpt --select_dir artifacts/select/hard \
     --data_dir artifacts/data/mgd_diff --methods classifier oracle domain_match ppl_top random \
     --seeds 5 --out artifacts/ablation/multiseed_diff.json
"""
import os, sys, json, glob, time, argparse, subprocess
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out_dir = a.out_dir or os.path.join("artifacts/cpt", "ms_" + os.path.basename(a.select_dir.rstrip("/")))
    os.makedirs(out_dir, exist_ok=True)
    env_base = dict(os.environ, WANDB_MODE="disabled", HF_HUB_DISABLE_PROGRESS_BARS="1",
                    TOKENIZERS_PARALLELISM="false")

    jobs = [(m, s) for m in a.methods for s in range(a.seeds)]
    results = {m: [] for m in a.methods}
    gi = 0
    for batch_start in range(0, len(jobs), 8):       # waves of 8 GPUs
        procs = []
        for (m, s) in jobs[batch_start:batch_start + 8]:
            sel = os.path.join(a.select_dir, f"{m}.npz")
            if not os.path.exists(sel):
                print(f"skip {m}: no {sel}", flush=True); continue
            name = f"{m}_s{s}"
            env = dict(env_base, CUDA_VISIBLE_DEVICES=str(gi % 8)); gi += 1
            cmd = [sys.executable, "-m", "src.train_final.cpt", "--sel", sel,
                   "--data_dir", a.data_dir, "--out_dir", out_dir, "--method", name,
                   "--epochs", str(a.epochs), "--lr", str(a.lr), "--bs", str(a.bs),
                   "--seed", str(s), "--eval_every_tokens", "200000", "--wandb_group", "ms"]
            log = open(os.path.join(out_dir, f"{name}.log"), "w")
            procs.append((m, s, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
        for (m, s, p) in procs:
            p.wait()
            jf = os.path.join(out_dir, f"{m}_s{s}.json")
            if os.path.exists(jf):
                results[m].append((s, json.load(open(jf))["improvement"]))
        print(f"wave done ({batch_start+8}/{len(jobs)})", flush=True)

    # aggregate + paired comparison vs classifier
    agg = {}
    for m, rs in results.items():
        v = np.array([x[1] for x in rs])
        agg[m] = dict(seeds=[x[0] for x in rs], improvements=[round(x[1], 4) for x in rs],
                      mean=float(v.mean()) if len(v) else None,
                      std=float(v.std(ddof=1)) if len(v) > 1 else None, n=len(v))
    cmp = {}
    if "classifier" in results:
        cdict = dict(results["classifier"])
        for m in a.methods:
            if m == "classifier": continue
            mdict = dict(results[m]); seeds = sorted(set(cdict) & set(mdict))
            if len(seeds) < 2: continue
            d = np.array([cdict[s] - mdict[s] for s in seeds])   # paired per seed
            sem = d.std(ddof=1) / np.sqrt(len(d))
            cmp[f"classifier_minus_{m}"] = dict(mean_diff=round(float(d.mean()), 4),
                                                sem=round(float(sem), 4), n=len(d),
                                                significant=bool(abs(d.mean()) > 2 * sem))
    res = dict(methods=agg, classifier_vs=cmp)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
