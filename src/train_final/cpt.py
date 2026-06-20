"""Phase-4: continued pretraining (CPT) on a selected token set, with held-out
PubMed perplexity logged vs tokens-seen to wandb. Run once per selection method
with an identical budget/schedule so the curves are directly comparable
("which data selection makes CPT more efficient?").
"""
from __future__ import annotations
import os, json, argparse, math, time
import numpy as np


def evaluate(model, val_ids, device, bs=32):
    import torch
    model.eval()
    tot, ntok = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val_ids), bs):
            b = torch.as_tensor(val_ids[i:i + bs], dtype=torch.long, device=device)
            out = model(b, labels=b)
            n = (b.shape[0] * (b.shape[1] - 1))
            tot += out.loss.item() * n; ntok += n
    model.train()
    return tot / ntok


def cpt(sel_path, data_dir, out_dir, method, lr=3e-5, bs=32, epochs=2,
        eval_every_tokens=200_000, seed=0, wandb_group="cpt", base="gpt2",
        max_steps=None):
    import torch
    from transformers import GPT2LMHeadModel
    torch.manual_seed(seed)
    device = "cuda"
    os.makedirs(out_dir, exist_ok=True)

    tok = np.load(os.path.join(data_dir, "tokens.npy"))
    val = np.load(os.path.join(data_dir, "val.npy")).astype(np.int64)
    sel = np.load(sel_path)["sel"]
    train_ids = tok[sel].astype(np.int64)
    t_seq = tok.shape[1]

    model = GPT2LMHeadModel.from_pretrained(base).to(device)
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95))

    import wandb
    run = wandb.init(project="metagrad-distill", group=wandb_group, name=f"cpt-{method}",
                     config=dict(method=method, lr=lr, bs=bs, epochs=epochs,
                                 n_seqs=len(sel), n_tokens=int(len(sel) * t_seq),
                                 seed=seed, base=base))
    rng = np.random.default_rng(seed)
    ppl0 = math.exp(evaluate(model, val, device))
    run.log({"val_ppl": ppl0, "tokens": 0, "step": 0})
    print(f"[{method}] base val ppl={ppl0:.3f} | n_sel={len(sel)} tokens={len(sel)*t_seq}", flush=True)

    tokens_seen, step, next_eval = 0, 0, eval_every_tokens
    curve = [(0, ppl0)]
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(len(train_ids))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            b = torch.as_tensor(train_ids[idx], dtype=torch.long, device=device)
            out = model(b, labels=b)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            tokens_seen += b.shape[0] * b.shape[1]; step += 1
            if tokens_seen >= next_eval:
                ppl = math.exp(evaluate(model, val, device))
                run.log({"val_ppl": ppl, "tokens": tokens_seen, "step": step,
                         "train_loss": out.loss.item()})
                curve.append((tokens_seen, ppl))
                print(f"[{method}] tok={tokens_seen} step={step} val_ppl={ppl:.3f}", flush=True)
                next_eval += eval_every_tokens
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    ppl_final = math.exp(evaluate(model, val, device))
    run.log({"val_ppl": ppl_final, "tokens": tokens_seen, "step": step})
    curve.append((tokens_seen, ppl_final))
    res = dict(method=method, base_ppl=ppl0, final_ppl=ppl_final,
              improvement=ppl0 - ppl_final, n_tokens=int(len(sel) * t_seq),
              wall_s=round(time.time() - t0, 1), curve=curve)
    json.dump(res, open(os.path.join(out_dir, f"{method}.json"), "w"), indent=2)
    run.summary.update({"final_ppl": ppl_final, "improvement": ppl0 - ppl_final})
    run.finish()
    print(f"[{method}] DONE base={ppl0:.3f} -> final={ppl_final:.3f}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", required=True)
    ap.add_argument("--data_dir", default="artifacts/data/mgd_v1")
    ap.add_argument("--out_dir", default="artifacts/cpt")
    ap.add_argument("--method", required=True)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--eval_every_tokens", type=int, default=200_000)
    ap.add_argument("--wandb_group", default="cpt")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cpt(args.sel, args.data_dir, args.out_dir, args.method, args.lr, args.bs,
        args.epochs, args.eval_every_tokens, seed=args.seed, wandb_group=args.wandb_group,
        max_steps=args.max_steps)
