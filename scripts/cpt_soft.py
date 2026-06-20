"""Soft-weighted CPT: instead of a hard top-n cut, use the classifier score s-hat
as a continuous per-sample WEIGHT (the metagradient oracle is literally defined
w.r.t. a continuous weight w_i, so this is the more faithful use of the signal).

Two modes, both matched to the hard run's compute (same # optimizer steps, same bs):
  sample     : each batch is importance-sampled from the FULL corpus with prob
               p_i ∝ softmax(s-hat / temp)  (soft, stochastic selection)
  lossweight : sample the full corpus uniformly, scale each example's loss by a
               normalised weight w_i ∝ softmax(s-hat / temp)

Compare final held-out PubMed ppl improvement to the hard top-n classifier/oracle.
temp→0 recovers hard top-n; large temp → uniform (= random).

  python -m scripts.cpt_soft --data_dir artifacts/data/mgd_hard \
     --pred artifacts/clf/hard/pred.npz --steps 471 --temp 0.3 --mode sample --method soft_hard_t0.3
"""
import os, json, argparse, math, time
import numpy as np
from src.train_final.cpt import evaluate


def softmax(x, temp):
    z = (x - x.max()) / max(temp, 1e-6)
    e = np.exp(z); return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out_dir", default="artifacts/cpt/soft")
    ap.add_argument("--method", required=True)
    ap.add_argument("--steps", type=int, required=True)   # match hard run's optimizer steps
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--mode", choices=["sample", "lossweight"], default="sample")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--eval_every", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    from transformers import GPT2LMHeadModel
    torch.manual_seed(a.seed); device = "cuda"; os.makedirs(a.out_dir, exist_ok=True)

    tok = np.load(os.path.join(a.data_dir, "tokens.npy")).astype(np.int64)
    val = np.load(os.path.join(a.data_dir, "val.npy")).astype(np.int64)
    shat = np.load(a.pred)["pred"].astype(np.float64)
    p = softmax(shat, a.temp)                          # corpus-level sampling/weight dist
    w_full = (p * len(p))                              # mean-1 weights for lossweight mode

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    rng = np.random.default_rng(a.seed)
    ppl0 = math.exp(evaluate(model, val, device)); curve = [(0, ppl0)]
    print(f"[{a.method}] base ppl={ppl0:.3f} mode={a.mode} temp={a.temp} steps={a.steps}", flush=True)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        if a.mode == "sample":
            idx = rng.choice(len(tok), size=a.bs, replace=False, p=p)
            b = torch.as_tensor(tok[idx], dtype=torch.long, device=device)
            loss = model(b, labels=b).loss
        else:  # lossweight: uniform sample, weight the per-example loss
            idx = rng.integers(0, len(tok), size=a.bs)
            b = torch.as_tensor(tok[idx], dtype=torch.long, device=device)
            logits = model(b).logits[:, :-1, :]
            tgt = b[:, 1:]
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none"
            ).view(tgt.shape).mean(1)                   # [bs] per-example loss
            wt = torch.as_tensor(w_full[idx], dtype=ce.dtype, device=device)
            loss = (ce * wt).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % a.eval_every == 0:
            ppl = math.exp(evaluate(model, val, device)); curve.append((step, ppl))
            print(f"[{a.method}] step={step} val_ppl={ppl:.3f}", flush=True)
    ppl_final = math.exp(evaluate(model, val, device)); curve.append((a.steps, ppl_final))
    res = dict(method=a.method, mode=a.mode, temp=a.temp, steps=a.steps,
               base_ppl=ppl0, final_ppl=ppl_final, improvement=ppl0 - ppl_final,
               wall_s=round(time.time() - t0, 1), curve=curve)
    json.dump(res, open(os.path.join(a.out_dir, f"{a.method}.json"), "w"), indent=2)
    print(f"[{a.method}] DONE {ppl0:.3f} -> {ppl_final:.3f} (impr {ppl0-ppl_final:+.3f})", flush=True)


if __name__ == "__main__":
    main()
