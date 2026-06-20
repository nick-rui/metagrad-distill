"""H5 — cohort lift prediction (hackathon protocol).
Build cohorts that vary composition (fraction good/offdomain/corrupt); for each,
the dataset-level aggregate of our predicted score should predict the cohort's
CPT lift better than cheap baselines.

This module builds the cohort selection files + a manifest. CPT is run per cohort
(reusing cpt.py via scripts/run_cohorts.py), then `correlate()` ties predicted
aggregate score to measured lift.
"""
from __future__ import annotations
import os, json, glob
import numpy as np


def build_cohorts(data_dir, pred_path, out_dir, n_per=2000, n_cohorts=14, seed=0):
    """Each cohort = n_per sequences at a target good-fraction; the rest split
    evenly between offdomain and corrupt. Saves sel + the aggregate features."""
    import csv
    tok = np.load(os.path.join(data_dir, "tokens.npy"))
    pred = np.load(pred_path)["pred"]
    clusters = np.array([r[1] for r in list(csv.reader(open(os.path.join(data_dir, "meta.csv"))))[1:]])
    idx_by = {c: np.where(clusters == c)[0] for c in ("good", "offdomain", "corrupt")}
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    manifest = {}
    good_fracs = np.linspace(0.0, 1.0, n_cohorts)
    for ci, gf in enumerate(good_fracs):
        n_good = int(round(gf * n_per))
        n_rest = n_per - n_good
        n_off, n_cor = n_rest // 2, n_rest - n_rest // 2
        pick = np.concatenate([
            rng.choice(idx_by["good"], min(n_good, len(idx_by["good"])), replace=False),
            rng.choice(idx_by["offdomain"], n_off, replace=False),
            rng.choice(idx_by["corrupt"], n_cor, replace=False)])
        name = f"cohort{ci:02d}_g{gf:.2f}"
        np.savez(os.path.join(out_dir, f"{name}.npz"), sel=pick)
        manifest[name] = dict(good_frac=float(gf), n=len(pick),
                              mean_pred=float(pred[pick].mean()),
                              frac_highpred=float((pred[pick] > np.median(pred)).mean()))
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    print(f"built {len(manifest)} cohorts in {out_dir}", flush=True)
    return manifest


def correlate(cohort_dir, cpt_dir, features_path, data_dir, out_path):
    """Correlate aggregate predicted score vs measured CPT lift across cohorts,
    against a base-perplexity baseline aggregate."""
    from scipy.stats import spearmanr
    manifest = json.load(open(os.path.join(cohort_dir, "manifest.json")))
    base_loss = np.load(features_path)["base_loss"]
    rows = []
    for name, info in manifest.items():
        cj = os.path.join(cpt_dir, f"{name}.json")
        if not os.path.exists(cj):
            continue
        d = json.load(open(cj))
        sel = np.load(os.path.join(cohort_dir, f"{name}.npz"))["sel"]
        rows.append(dict(name=name, good_frac=info["good_frac"],
                         mean_pred=info["mean_pred"],
                         mean_baseppl=float(np.exp(base_loss[sel].mean())),
                         lift=d["improvement"]))
    if not rows:
        print("no cohort CPT results yet", flush=True); return None
    pred = np.array([r["mean_pred"] for r in rows])
    bppl = np.array([r["mean_baseppl"] for r in rows])
    lift = np.array([r["lift"] for r in rows])
    def r2(x, y):
        b1, b0 = np.polyfit(x, y, 1); yh = b1 * x + b0
        return float(1 - np.sum((y - yh)**2) / np.sum((y - y.mean())**2))
    res = dict(n_cohorts=len(rows),
               ours=dict(spearman=float(spearmanr(pred, lift).correlation), r2=r2(pred, lift)),
               baseline_ppl=dict(spearman=float(spearmanr(bppl, lift).correlation), r2=r2(bppl, lift)),
               rows=rows)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2), flush=True)
    return res
