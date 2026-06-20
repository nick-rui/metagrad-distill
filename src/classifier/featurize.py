"""Compute cheap per-sequence features over the whole corpus (forward only):
  - mean-pooled final hidden state of base GPT-2  -> [M, 768]  (classifier input)
  - base-model perplexity                          -> [M]       (perplexity baseline)

These are the 'cheap, no-training' signals the classifier maps to oracle scores.
"""
from __future__ import annotations
import os, argparse, time
import numpy as np


def featurize(data_dir, out_path, base="gpt2", bs=256):
    import jax, jax.numpy as jnp
    from src.metagrad import model_gpt2 as M
    tok = np.load(os.path.join(data_dir, "tokens.npy"))
    params, cfg = M.load_pretrained(base)

    feat_fn = jax.jit(lambda p, x: M.hidden_features(p, x, cfg))
    loss_fn = jax.jit(lambda p, x: M.loss_per_example(p, x, cfg, False))
    feats = np.zeros((len(tok), cfg.d), np.float32)
    losses = np.zeros(len(tok), np.float32)
    t0 = time.time()
    for i in range(0, len(tok), bs):
        b = jnp.asarray(tok[i:i + bs].astype(np.int32))
        feats[i:i + bs] = np.asarray(feat_fn(params, b))
        losses[i:i + bs] = np.asarray(loss_fn(params, b))
        if i % (bs * 20) == 0:
            print(f"  featurized {i}/{len(tok)} ({(i+bs)/(time.time()-t0):.0f} seq/s)", flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, feats=feats, base_loss=losses)
    print(f"saved {out_path}: feats {feats.shape}, ppl mean={np.exp(losses.mean()):.1f} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="artifacts/data/mgd_v1")
    ap.add_argument("--out_path", default="artifacts/features/mgd_v1.npz")
    ap.add_argument("--bs", type=int, default=256)
    args = ap.parse_args()
    featurize(args.data_dir, args.out_path, bs=args.bs)
