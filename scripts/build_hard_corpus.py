"""Hard, all-in-domain corpus to isolate the metagradient's value-add over cheap
baselines. EVERYTHING is PubMed-derived, so `domain_match` (feature similarity to
target) and `ppl_top` (low base perplexity) can no longer separate good from bad
by surface cues. Training value is decorrelated from both:

  clean      : real PubMed abstracts            -> HIGH training value      (target-like)
  repetitive : a PubMed window tiled to fill L  -> LOW value, *lowest* ppl   (fools ppl_top)
  noised     : clean with 20% tokens randomised -> LOW value, clean-like feats (fools domain_match)

Target Phi = held-out CLEAN PubMed. A method that truly measures training value
(the metagradient) should pick `clean` and avoid both `repetitive` and `noised`;
ppl_top should over-pick `repetitive`, domain_match should over-pick `noised`.

Output layout matches mgd_v1: artifacts/data/mgd_hard/{tokens.npy,val.npy,meta.csv,stats.json}
"""
from __future__ import annotations
import argparse, json, os, csv
from dataclasses import dataclass, asdict
import numpy as np
from src.data.corpus import _tokenizer, _pack_stream


@dataclass
class HardConfig:
    name: str = "mgd_hard"
    t_seq: int = 256
    n_clean: int = 20000
    n_repetitive: int = 15000
    n_noised: int = 15000
    n_val: int = 2000
    rep_window: int = 16        # tile this many tokens to fill the sequence
    noise_frac: float = 0.20    # fraction of positions randomised in `noised`
    pubmed_docs: int = 150000
    seed: int = 0
    out_root: str = "artifacts/data"


def build(cfg: HardConfig):
    from datasets import load_dataset
    rng = np.random.default_rng(cfg.seed)
    tok = _tokenizer(); eos = tok.eos_token_id; vocab = tok.vocab_size
    out_dir = os.path.join(cfg.out_root, cfg.name); os.makedirs(out_dir, exist_ok=True)

    need = cfg.n_val + cfg.n_clean + cfg.n_repetitive + cfg.n_noised
    print("[hard] streaming PubMed...", flush=True)
    pm = load_dataset("MedRAG/pubmed", streaming=True, split="train")
    allseq, ndocs = _pack_stream((ex["contents"] for ex in pm), tok, eos, need, cfg.t_seq, cfg.pubmed_docs)
    assert len(allseq) >= need, f"only packed {len(allseq)} / {need}"
    rng.shuffle(allseq)

    i = 0
    val = allseq[i:i + cfg.n_val]; i += cfg.n_val
    clean = allseq[i:i + cfg.n_clean]; i += cfg.n_clean
    rep_src = allseq[i:i + cfg.n_repetitive]; i += cfg.n_repetitive
    noise_src = allseq[i:i + cfg.n_noised]; i += cfg.n_noised

    # repetitive: tile the first `rep_window` tokens across the whole sequence
    W, L = cfg.rep_window, cfg.t_seq
    reps = np.tile(rep_src[:, :W], (1, (L + W - 1) // W))[:, :L].astype(np.uint16)

    # noised: replace noise_frac of positions with uniform-random in-vocab tokens
    noised = noise_src.copy()
    mask = rng.random(noised.shape) < cfg.noise_frac
    noised[mask] = rng.integers(0, vocab, size=int(mask.sum()), dtype=np.uint16)

    parts = [("clean", clean), ("repetitive", reps), ("noised", noised)]
    tokens = np.concatenate([p[1] for p in parts], 0).astype(np.uint16)
    clusters, sources = [], []
    for cname, arr in parts:
        clusters += [cname] * len(arr); sources += ["pubmed-" + cname] * len(arr)
    perm = rng.permutation(len(tokens))
    tokens = tokens[perm]; clusters = np.asarray(clusters)[perm]; sources = np.asarray(sources)[perm]

    np.save(os.path.join(out_dir, "tokens.npy"), tokens)
    np.save(os.path.join(out_dir, "val.npy"), val.astype(np.uint16))
    with open(os.path.join(out_dir, "meta.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["seq_id", "cluster", "source"])
        for j in range(len(tokens)): w.writerow([j, clusters[j], sources[j]])

    stats = dict(**asdict(cfg), M=len(tokens), V=len(val),
                 tokens_total=int(len(tokens) * cfg.t_seq), pubmed_docs_used=int(ndocs),
                 cluster_counts={c: int((clusters == c).sum()) for c in ("clean", "repetitive", "noised")},
                 vocab="gpt2", eos_id=int(eos),
                 note="all-in-domain; clean=value, repetitive=low-ppl-low-value, noised=cleanfeat-low-value")
    json.dump(stats, open(os.path.join(out_dir, "stats.json"), "w"), indent=2)
    print("[hard] done:", json.dumps(stats["cluster_counts"]), "M=", stats["M"], flush=True)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k, v in asdict(HardConfig()).items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    a = ap.parse_args()
    build(HardConfig(**vars(a)))
