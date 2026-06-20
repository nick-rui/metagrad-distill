"""Phase-2: distill oracle metagradient labels into a cheap regressor over
features, then score the whole corpus. Reports H1 (faithfulness): Spearman rho
between predicted and oracle scores on a held-out split.
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
from scipy.stats import spearmanr


def _fit(model_kind, Xtr, ytr):
    if model_kind == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        m = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    elif model_kind == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                              num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                              min_child_samples=20, n_jobs=-1)
    elif model_kind == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        m = make_pipeline(StandardScaler(),
                          MLPRegressor(hidden_layer_sizes=(256, 64), max_iter=200,
                                       early_stopping=True))
    else:
        raise ValueError(model_kind)
    m.fit(Xtr, ytr)
    return m


def train(features_path, labels_path, data_dir, out_dir, model_kind="lgbm",
          test_frac=0.2, seed=0, wandb_run=None):
    os.makedirs(out_dir, exist_ok=True)
    feats = np.load(features_path)["feats"]
    L = np.load(labels_path)
    label, cnt = L["label"], L["cnt"]
    import csv
    clusters = np.array([r[1] for r in list(csv.reader(open(os.path.join(data_dir, "meta.csv"))))[1:]])

    labeled = np.where(np.isfinite(label) & (cnt > 0))[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(labeled)
    n_test = int(len(labeled) * test_frac)
    test_idx, train_idx = labeled[:n_test], labeled[n_test:]

    m = _fit(model_kind, feats[train_idx], label[train_idx])
    pred_test = m.predict(feats[test_idx])
    rho = spearmanr(pred_test, label[test_idx]).correlation
    r2 = float(1 - np.sum((pred_test - label[test_idx])**2) /
               np.sum((label[test_idx] - label[test_idx].mean())**2))

    # score the WHOLE corpus
    pred_all = m.predict(feats).astype(np.float32)
    np.savez(os.path.join(out_dir, "pred.npz"), pred=pred_all)

    # does the classifier recover the good cluster? (mean predicted score per cluster)
    pred_by_cluster = {c: float(pred_all[clusters == c].mean())
                       for c in sorted(set(clusters.tolist()))}
    res = dict(model=model_kind, n_train=len(train_idx), n_test=len(test_idx),
               H1_spearman=float(rho), H1_r2=r2, pred_by_cluster=pred_by_cluster)
    json.dump(res, open(os.path.join(out_dir, "h1.json"), "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)
    if wandb_run is not None:
        wandb_run.log({"H1_spearman": rho, "H1_r2": r2,
                       **{f"pred/{c}": v for c, v in pred_by_cluster.items()}})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="artifacts/features/mgd_v1.npz")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--data_dir", default="artifacts/data/mgd_v1")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="lgbm", choices=["ridge", "lgbm", "mlp"])
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="metagrad-distill", group="classifier",
                         name=f"clf-{args.model}", config=vars(args))
    train(args.features, args.labels, args.data_dir, args.out_dir, args.model, wandb_run=run)
    if run:
        run.finish()
