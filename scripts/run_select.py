"""Build selection files (seq indices) for every method at a token budget, so
final CPT is apples-to-apples. Methods: our classifier, oracle (metagrad labels),
and cheap baselines. Writes artifacts/select/<tag>/<method>.npz with 'sel'.
"""
import os, json, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="artifacts/data/mgd_v1")
    ap.add_argument("--features", default="artifacts/features/mgd_v1.npz")
    ap.add_argument("--labels", required=True)          # oracle metagrad labels
    ap.add_argument("--pred", required=True)            # classifier predictions
    ap.add_argument("--val_features", default="artifacts/features/mgd_v1_val.npz")
    ap.add_argument("--budget_frac", type=float, default=0.10)
    ap.add_argument("--tag", default="b10")
    ap.add_argument("--dedup_thresh", type=float, default=None)
    args = ap.parse_args()

    from src.baselines.scores import all_baseline_scores
    from src.select.select import select_topn

    tok = np.load(os.path.join(args.data_dir, "tokens.npy"))
    t_seq = tok.shape[1]
    feats = np.load(args.features)["feats"]
    base_loss = np.load(args.features)["base_loss"]
    vf = np.load(args.val_features)
    val_feats, val_base_loss = vf["feats"], vf["base_loss"]
    oracle = np.load(args.labels)["label"]
    pred = np.load(args.pred)["pred"]

    scores = all_baseline_scores(tok, feats, base_loss, val_feats, val_base_loss)
    scores["oracle"] = oracle          # nan where unlabeled -> ineligible (ok if full coverage)
    scores["classifier"] = pred        # our method

    out_dir = f"artifacts/select/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    import csv
    clusters = np.array([r[1] for r in list(csv.reader(open(os.path.join(args.data_dir, "meta.csv"))))[1:]])
    summary = {}
    for name, sc in scores.items():
        feats_for_dedup = feats if args.dedup_thresh else None
        sel = select_topn(sc, t_seq, args.budget_frac, higher_better=True,
                          feats=feats_for_dedup, dedup_thresh=args.dedup_thresh)
        np.savez(os.path.join(out_dir, f"{name}.npz"), sel=sel)
        frac = {c: float((clusters[sel] == c).mean()) for c in sorted(set(clusters.tolist()))}
        summary[name] = dict(n_sel=int(len(sel)), tokens=int(len(sel) * t_seq), cluster_frac=frac)
    # full-corpus upper bound = all sequences
    np.savez(os.path.join(out_dir, "full.npz"), sel=np.arange(len(tok)))
    json.dump(dict(budget_frac=args.budget_frac, t_seq=t_seq, methods=summary),
              open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
