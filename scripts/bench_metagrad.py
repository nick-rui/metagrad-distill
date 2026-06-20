"""Measure peak GPU memory + wall-time of one metagradient round on real GPT-2
small, across (k, T, L_inner). Grounds the feasibility budget for the full plan.
"""
import os, time, json, argparse
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import numpy as np, jax
from src.metagrad import model_gpt2 as M
from src.metagrad.metagrad import metagrad_scores

def peak_gb():
    s = jax.local_devices()[0].memory_stats() or {}
    return s.get("peak_bytes_in_use", 0) / 1e9

def reset_peak():
    d = jax.local_devices()[0]
    try: d.memory_stats() and d.clear_memory_stats()  # best-effort
    except Exception: pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=str, required=True,
                    help="semicolon list of k,T,L_inner e.g. '128,16,256;256,16,128'")
    ap.add_argument("--val_bs", type=int, default=128)
    args = ap.parse_args()

    params, cfg = M.load_pretrained("gpt2")
    tok = np.load("artifacts/data/mgd_v1/tokens.npy").astype(np.int32)
    val = np.load("artifacts/data/mgd_v1/val.npy").astype(np.int32)
    rows = []
    for spec in args.configs.split(";"):
        k, T, L = [int(x) for x in spec.split(",")]
        seqs = tok[:k]
        try:
            t0 = time.time()
            s, phi = metagrad_scores(params, seqs, val, cfg, T=T, lr=1e-3,
                                     val_bs=args.val_bs, L_inner=L)
            jax.block_until_ready(s); compile_t = time.time() - t0
            t0 = time.time()
            s, phi = metagrad_scores(params, seqs, val, cfg, T=T, lr=1e-3,
                                     val_bs=args.val_bs, L_inner=L)
            jax.block_until_ready(s); run_t = time.time() - t0
            row = dict(k=k, T=T, L_inner=L, val_bs=args.val_bs, peak_gb=round(peak_gb(), 2),
                       compile_s=round(compile_t, 1), round_s=round(run_t, 2),
                       phi=round(float(phi), 4), s_finite=bool(np.isfinite(s).all()))
        except Exception as e:
            row = dict(k=k, T=T, L_inner=L, error=type(e).__name__ + ": " + str(e)[:60])
        rows.append(row); print(json.dumps(row), flush=True)
    os.makedirs("artifacts/bench", exist_ok=True)
    json.dump(rows, open("artifacts/bench/metagrad_bench.json", "w"), indent=2)
    print("saved artifacts/bench/metagrad_bench.json")

if __name__ == "__main__":
    main()
