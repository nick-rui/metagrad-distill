"""Cross-corpus amortization (#3): train the classifier on corpus A's (features,
oracle-labels), then SCORE corpus B's features with it — testing whether the
learned metagradient-value function transfers to a different, unlabeled corpus
(the 'pay the oracle once, score everything cheaply forever' claim).

Outputs B-pred.npz (key 'pred') consumable by scripts.run_select, plus a small
transfer.json with per-cluster mean predicted score on B.

  python -m scripts.cross_score --train_feats artifacts/features/mgd_v1.npz \
     --train_labels artifacts/labels/main/labels.npz \
     --score_feats artifacts/features/hard.npz --score_meta artifacts/data/mgd_hard/meta.csv \
     --out artifacts/clf/xfer_v1_to_hard/pred.npz
"""
import os, json, argparse, csv
import numpy as np
import lightgbm as lgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_feats", required=True)
    ap.add_argument("--train_labels", required=True)
    ap.add_argument("--score_feats", required=True)
    ap.add_argument("--score_meta", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    Xtr = np.load(a.train_feats)["feats"]
    L = np.load(a.train_labels); label, cnt = L["label"], L["cnt"]
    lab = np.where(np.isfinite(label) & (cnt > 0))[0]
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                          subsample=0.8, colsample_bytree=0.8, min_child_samples=20, n_jobs=-1)
    m.fit(Xtr[lab], label[lab])

    Xb = np.load(a.score_feats)["feats"]
    pred = m.predict(Xb).astype(np.float32)
    np.savez(a.out, pred=pred)

    clusters = np.array([r[1] for r in list(csv.reader(open(a.score_meta)))[1:]])
    by = {c: float(pred[clusters == c].mean()) for c in sorted(set(clusters.tolist()))}
    info = dict(train_feats=a.train_feats, score_feats=a.score_feats,
                n_train=int(len(lab)), n_scored=int(len(pred)), pred_by_cluster=by)
    json.dump(info, open(a.out.replace(".npz", "_info.json"), "w"), indent=2)
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
