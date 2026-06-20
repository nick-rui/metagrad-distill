"""Compute cheap per-sequence features over the whole corpus (forward only):
  - mean-pooled final hidden state of base GPT-2  -> [M, 768]  (classifier input)
  - base-model perplexity                          -> [M]       (perplexity baseline)

These are the 'cheap, no-training' signals the classifier maps to oracle scores.
"""
from __future__ import annotations
import os, argparse, time
import numpy as np


def _featurize_array(tok, params, cfg, feat_fn, loss_fn, bs, jnp):
    feats = np.zeros((len(tok), cfg.d), np.float32)
    losses = np.zeros(len(tok), np.float32)
    t0 = time.time()
    for i in range(0, len(tok), bs):
        b = jnp.asarray(tok[i:i + bs].astype(np.int32))
        feats[i:i + bs] = np.asarray(feat_fn(params, b))
        losses[i:i + bs] = np.asarray(loss_fn(params, b))
        if i % (bs * 20) == 0:
            print(f"  {i}/{len(tok)} ({(i+bs)/(time.time()-t0):.0f} seq/s)", flush=True)
    return feats, losses


def featurize(data_dir, out_path, base="gpt2", bs=256):
    import jax, jax.numpy as jnp
    from src.metagrad import model_gpt2 as M
    params, cfg = M.load_pretrained(base)
    feat_fn = jax.jit(lambda p, x: M.hidden_features(p, x, cfg))
    loss_fn = jax.jit(lambda p, x: M.loss_per_example(p, x, cfg, False))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tok = np.load(os.path.join(data_dir, "tokens.npy"))
    feats, losses = _featurize_array(tok, params, cfg, feat_fn, loss_fn, bs, jnp)
    np.savez(out_path, feats=feats, base_loss=losses)
    print(f"saved {out_path}: feats {feats.shape}, ppl mean={np.exp(losses.mean()):.1f}", flush=True)

    val = np.load(os.path.join(data_dir, "val.npy"))
    vf, vl = _featurize_array(val, params, cfg, feat_fn, loss_fn, bs, jnp)
    val_path = out_path.replace(".npz", "_val.npz")
    np.savez(val_path, feats=vf, base_loss=vl)
    print(f"saved {val_path}: feats {vf.shape}, val ppl={np.exp(vl.mean()):.1f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="artifacts/data/mgd_v1")
    ap.add_argument("--out_path", default="artifacts/features/mgd_v1.npz")
    ap.add_argument("--bs", type=int, default=256)
    args = ap.parse_args()
    featurize(args.data_dir, args.out_path, bs=args.bs)
