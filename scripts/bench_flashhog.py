"""A/B benchmark: XLA float32 attention vs flash-hog higher-order attention,
inside the *metagradient* (a 2nd-order grad through the unrolled Adam loop).

Reports, per (backend, L_inner): peak GPU memory, compile time, round time, phi,
and whether the metagradient is finite. Then, at the largest L_inner where the
XLA path still fits, reports Spearman rho between the two backends' goodness
scores s -- i.e. does the bf16 flash-hog kernel preserve the *ranking* the
validated float32 path produces (the only thing the downstream classifier needs).

The point flash-hog makes: linear (not quadratic) attention memory lets the
metagradient run at long L_inner that the XLA path OOMs on.

Usage:
  python -m scripts.bench_flashhog --k 64 --T 16 --L_list 128,256,512,1024
"""
import os, time, json, argparse
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import dataclasses
import numpy as np, jax
from scipy.stats import spearmanr
from src.metagrad import model_gpt2 as M
from src.metagrad.metagrad import metagrad_scores


def peak_gb():
    s = jax.local_devices()[0].memory_stats() or {}
    return s.get("peak_bytes_in_use", 0) / 1e9


def run_one(params, cfg, seqs, val, T, L, val_bs, lr=3e-5):
    """compile + timed round; returns (row, s) or (row, None) on failure."""
    try:
        t0 = time.time()
        s, phi = metagrad_scores(params, seqs, val, cfg, T=T, lr=lr, val_bs=val_bs, L_inner=L)
        jax.block_until_ready(s); compile_t = time.time() - t0
        t0 = time.time()
        s, phi = metagrad_scores(params, seqs, val, cfg, T=T, lr=lr, val_bs=val_bs, L_inner=L)
        jax.block_until_ready(s); run_t = time.time() - t0
        row = dict(attn=cfg.attn_impl, L_inner=L, peak_gb=round(peak_gb(), 2),
                   compile_s=round(compile_t, 1), round_s=round(run_t, 2),
                   phi=round(float(phi), 4), s_finite=bool(np.isfinite(s).all()))
        return row, np.asarray(s, np.float64)
    except Exception as e:
        return dict(attn=cfg.attn_impl, L_inner=L,
                    error=type(e).__name__ + ": " + str(e)[:80]), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--val_bs", type=int, default=128)
    ap.add_argument("--L_list", type=str, default="128,256,512")
    ap.add_argument("--out", default="artifacts/bench/flashhog_bench.json")
    args = ap.parse_args()
    Ls = [int(x) for x in args.L_list.split(",")]

    params, cfg0 = M.load_pretrained("gpt2")
    cfg_xla = dataclasses.replace(cfg0, attn_impl="xla")
    cfg_fh = dataclasses.replace(cfg0, attn_impl="flashhog")
    tok = np.load("artifacts/data/mgd_v1/tokens.npy").astype(np.int32)
    val = np.load("artifacts/data/mgd_v1/val.npy").astype(np.int32)
    seqs = tok[:args.k]

    rows, scores = [], {}
    for L in Ls:
        for cfg in (cfg_xla, cfg_fh):
            row, s = run_one(params, cfg, seqs, val, args.T, L, args.val_bs)
            rows.append(row); print(json.dumps(row), flush=True)
            if s is not None:
                scores[(cfg.attn_impl, L)] = s

    # ranking agreement at each L where both backends produced finite scores
    agree = []
    for L in Ls:
        a, b = scores.get(("xla", L)), scores.get(("flashhog", L))
        if a is not None and b is not None:
            rho = float(spearmanr(a, b).statistic)
            agree.append(dict(L_inner=L, spearman_xla_vs_flashhog=round(rho, 4), k=args.k))
            print(json.dumps(agree[-1]), flush=True)

    out = dict(rows=rows, agreement=agree, k=args.k, T=args.T, val_bs=args.val_bs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
