"""Aggregate results across methods into H3 (downstream win) and H4 (Pareto:
predictive power vs compute cost). Emits a markdown table, a Pareto PNG, and a
CPT-efficiency PNG (val ppl vs tokens), and optionally logs them to wandb.

Compute cost is expressed in approximate "sequence-step" units so methods are
comparable:
  random/length        : ~0           (no model pass)
  ppl_top/ppl_corr     : M            (one base forward per seq)
  domain_match         : M            (one base forward per seq, reuses features)
  classifier (ours)    : R*k*T  (labeling)  +  M (featurize/score)   <-- amortised oracle
  oracle (per-seq MG)  : C_full * M * T      (metagradient for the whole corpus)
  full-corpus          : M (train on everything; upper bound, not a selector)
"""
from __future__ import annotations
import os, json, glob
import numpy as np


def load_cpt(cpt_dir):
    res = {}
    for p in glob.glob(os.path.join(cpt_dir, "*.json")):
        d = json.load(open(p)); res[d["method"]] = d
    return res


def cost_units(method, M, R, k, T, C_full=3):
    return {
        "random": 1.0, "length": 1.0,
        "ppl_top": M, "ppl_corr": M, "domain_match": M,
        "classifier": R * k * T + M,          # labeling + scoring (amortised)
        "oracle": C_full * M * T,             # full-corpus metagradient
        "full": M,
    }.get(method, M)


def make_report(cpt_dir, out_dir, M=50000, R=3125, k=64, T=16, clf_h1=None,
                wandb_run=None):
    os.makedirs(out_dir, exist_ok=True)
    cpt = load_cpt(cpt_dir)
    rows = []
    for m, d in cpt.items():
        rows.append(dict(method=m, final_ppl=d["final_ppl"], improvement=d["improvement"],
                         cost=cost_units(m, M, R, k, T)))
    rows.sort(key=lambda r: r["final_ppl"])

    # markdown table
    md = ["| method | final PubMed ppl | improvement | cost (seq-steps) |",
          "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['method']} | {r['final_ppl']:.3f} | {r['improvement']:+.3f} | {r['cost']:.2e} |")
    md_txt = "\n".join(md)
    open(os.path.join(out_dir, "h3_table.md"), "w").write(md_txt)
    print(md_txt, flush=True)

    # Pareto + efficiency plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Pareto: improvement vs cost (log x)
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in rows:
            ax.scatter(r["cost"], r["improvement"], s=60)
            ax.annotate(r["method"], (r["cost"], r["improvement"]),
                        textcoords="offset points", xytext=(5, 3), fontsize=8)
        ax.set_xscale("log"); ax.set_xlabel("compute cost (seq-steps, log)")
        ax.set_ylabel("PubMed ppl improvement"); ax.set_title("H4 Pareto: power vs cost")
        ax.grid(True, alpha=0.3); fig.tight_layout()
        pareto_png = os.path.join(out_dir, "pareto.png"); fig.savefig(pareto_png, dpi=120)

        # Efficiency curves: val ppl vs tokens
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for m, d in cpt.items():
            if "curve" in d:
                c = np.asarray(d["curve"]); ax2.plot(c[:, 0], c[:, 1], marker="o", ms=3, label=m)
        ax2.set_xlabel("training tokens"); ax2.set_ylabel("held-out PubMed ppl")
        ax2.set_title("CPT efficiency by data selection"); ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3); fig2.tight_layout()
        eff_png = os.path.join(out_dir, "cpt_efficiency.png"); fig2.savefig(eff_png, dpi=120)
        if wandb_run is not None:
            import wandb
            wandb_run.log({"pareto": wandb.Image(pareto_png),
                           "cpt_efficiency": wandb.Image(eff_png)})
    except Exception as e:
        print("plotting skipped:", e, flush=True)

    json.dump(dict(rows=rows, h1=clf_h1), open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpt_dir", required=True)
    ap.add_argument("--out_dir", default="artifacts/report")
    args = ap.parse_args()
    make_report(args.cpt_dir, args.out_dir)
