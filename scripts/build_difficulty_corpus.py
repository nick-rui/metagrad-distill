"""Difficulty-stratified all-clean-PubMed corpus — the test for the metagradient's
one true edge over feature matching: MARGINAL value given what the base model
already knows. Every sequence is clean PubMed (so domain_match sees ~identical
features and is blind to difficulty); we stratify by base GPT-2 loss:

  easy : lowest-loss tercile  (model already predicts it -> low marginal value, lowest ppl => ppl_top's pick)
  mid  : middle tercile
  hard : highest-loss tercile (clean but model struggles -> high marginal value)

Top 1% loss is trimmed as outliers. Target Phi = held-out clean PubMed. If the
metagradient/classifier prefer `hard` and beat ppl_top (picks easy) AND
domain_match (blind to difficulty), that is value-add no cheap baseline can match.

Outputs artifacts/data/mgd_diff/{tokens.npy,val.npy,meta.csv,stats.json}.
"""
from __future__ import annotations
import os, json, csv, argparse
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
from src.data.corpus import _tokenizer, _pack_stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mgd_diff")
    ap.add_argument("--t_seq", type=int, default=256)
    ap.add_argument("--M", type=int, default=50000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--pubmed_docs", type=int, default=150000)
    ap.add_argument("--trim_top", type=float, default=0.01)   # drop top-1% loss as outliers
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import jax, jax.numpy as jnp
    from src.metagrad import model_gpt2 as M
    from datasets import load_dataset

    rng = np.random.default_rng(a.seed)
    tok = _tokenizer(); eos = tok.eos_token_id
    out_dir = f"artifacts/data/{a.name}"; os.makedirs(out_dir, exist_ok=True)

    # need extra to survive the outlier trim
    need = int((a.M + a.n_val) / (1 - a.trim_top)) + 2000
    print(f"[diff] streaming PubMed for {need} clean seqs...", flush=True)
    pm = load_dataset("MedRAG/pubmed", streaming=True, split="train")
    seqs, ndocs = _pack_stream((ex["contents"] for ex in pm), tok, eos, need, a.t_seq, a.pubmed_docs)
    rng.shuffle(seqs)

    params, cfg = M.load_pretrained("gpt2")
    loss_fn = jax.jit(lambda p, x: M.loss_per_example(p, x, cfg, False))
    losses = np.zeros(len(seqs), np.float32)
    for i in range(0, len(seqs), 512):
        losses[i:i+512] = np.asarray(loss_fn(params, jnp.asarray(seqs[i:i+512].astype(np.int32))))
        if i % 5120 == 0: print(f"  base-loss {i}/{len(seqs)}", flush=True)

    # hold out val first (random), then trim outliers, then tercile the rest by loss
    val = seqs[:a.n_val]; pool = seqs[a.n_val:]; ploss = losses[a.n_val:]
    keep = ploss <= np.quantile(ploss, 1 - a.trim_top)
    pool, ploss = pool[keep][:a.M], ploss[keep][:a.M]
    order = np.argsort(ploss); pool, ploss = pool[order], ploss[order]
    third = len(pool) // 3
    cluster = np.array(["easy"]*third + ["mid"]*(third) + ["hard"]*(len(pool)-2*third))

    # shuffle corpus order, keep aligned
    perm = rng.permutation(len(pool))
    tokens = pool[perm].astype(np.uint16); cluster = cluster[perm]
    np.save(os.path.join(out_dir, "tokens.npy"), tokens)
    np.save(os.path.join(out_dir, "val.npy"), val.astype(np.uint16))
    with open(os.path.join(out_dir, "meta.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["seq_id", "cluster", "source"])
        for j in range(len(tokens)): w.writerow([j, cluster[j], "pubmed"])

    ppl_by = {c: float(np.exp(ploss[(np.array(["easy"]*third+["mid"]*third+["hard"]*(len(pool)-2*third)))==c].mean()))
              for c in ["easy", "mid", "hard"]}
    stats = dict(name=a.name, t_seq=a.t_seq, M=len(tokens), V=len(val),
                 tokens_total=int(len(tokens)*a.t_seq), pubmed_docs_used=int(ndocs),
                 cluster_counts={c: int((cluster == c).sum()) for c in ["easy", "mid", "hard"]},
                 base_ppl_by_cluster=ppl_by, trim_top=a.trim_top, vocab="gpt2", eos_id=int(eos),
                 note="all clean PubMed, stratified by base GPT-2 loss; tests marginal-value/difficulty")
    json.dump(stats, open(os.path.join(out_dir, "stats.json"), "w"), indent=2)
    print("[diff] done:", json.dumps(stats["cluster_counts"]), "base_ppl:", json.dumps(ppl_by), flush=True)


if __name__ == "__main__":
    main()
