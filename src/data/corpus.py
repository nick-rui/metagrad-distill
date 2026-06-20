"""Build the MetaGrad-Distill corpus: a mixture of clusters with a target val set.

Clusters
--------
- ``good``      : on-target biomedical text (PubMed, MedRAG/pubmed `contents`).
- ``offdomain`` : off-domain web text (allenai/c4 `text`).
- ``corrupt``   : token-shuffled `good` sequences (fluent-domain but destroyed structure).

Target metric Phi = LM loss on a held-out PubMed split (disjoint from the corpus).
A working selection method should preferentially pick the ``good`` cluster.

Outputs (under ``artifacts/data/<name>/``)
- ``tokens.npy``   uint16 [M, T_seq]    corpus sequences
- ``meta.csv``     seq_id, cluster, source
- ``val.npy``      uint16 [V, T_seq]    held-out target-domain val sequences
- ``stats.json``   corpus statistics
"""
from __future__ import annotations
import argparse, json, os
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class DataConfig:
    name: str = "mgd_v1"
    t_seq: int = 256
    n_good: int = 20000        # corpus good sequences (target == #packed seqs)
    n_offdomain: int = 20000
    n_corrupt: int = 10000     # made by shuffling tokens of good seqs
    n_val: int = 2000          # held-out target-domain val sequences
    # how many raw docs to stream (packed density ~ docs*tokens/t_seq)
    pubmed_docs: int = 90000
    c4_docs: int = 60000
    seed: int = 0
    out_root: str = "artifacts/data"


def _tokenizer():
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    return tok


def _pack_stream(texts, tok, eos_id, n_seqs, t_seq, max_docs):
    """Stream `texts`, tokenize, pack into [n_seqs, t_seq] with EOS between docs."""
    buf, out, ndocs = [], [], 0
    for t in texts:
        if not t or len(t) < 40:
            continue
        ids = tok.encode(t)
        ids.append(eos_id)
        buf.extend(ids)
        ndocs += 1
        while len(buf) >= t_seq:
            out.append(buf[:t_seq])
            buf = buf[t_seq:]
            if len(out) >= n_seqs:
                return np.asarray(out, dtype=np.uint16), ndocs
        if ndocs >= max_docs:
            break
    return np.asarray(out, dtype=np.uint16), ndocs


def build(cfg: DataConfig):
    from datasets import load_dataset
    rng = np.random.default_rng(cfg.seed)
    tok = _tokenizer()
    eos = tok.eos_token_id
    out_dir = os.path.join(cfg.out_root, cfg.name)
    os.makedirs(out_dir, exist_ok=True)

    # --- good: PubMed (corpus + disjoint val). Stream once, split by doc index. ---
    print("[data] streaming PubMed (good + val)...", flush=True)
    pm = load_dataset("MedRAG/pubmed", streaming=True, split="train")
    pm_text = (ex["contents"] for ex in pm)
    # need enough packed seqs for both corpus-good and val
    good_all, ndocs_pm = _pack_stream(
        pm_text, tok, eos, cfg.n_good + cfg.n_val + cfg.n_corrupt, cfg.t_seq, cfg.pubmed_docs)
    assert len(good_all) >= cfg.n_good + cfg.n_val, f"only packed {len(good_all)} pubmed seqs"
    rng.shuffle(good_all)
    val = good_all[: cfg.n_val]
    good = good_all[cfg.n_val : cfg.n_val + cfg.n_good]
    corrupt_src = good_all[cfg.n_val + cfg.n_good : cfg.n_val + cfg.n_good + cfg.n_corrupt]

    # --- corrupt: shuffle tokens within each sequence (destroy structure, keep unigram stats) ---
    corrupt = corrupt_src.copy()
    for i in range(len(corrupt)):
        rng.shuffle(corrupt[i])

    # --- offdomain: C4 web ---
    print("[data] streaming C4 (offdomain)...", flush=True)
    c4 = load_dataset("allenai/c4", name="en", streaming=True, split="train")
    c4_text = (ex["text"] for ex in c4)
    off, ndocs_c4 = _pack_stream(c4_text, tok, eos, cfg.n_offdomain, cfg.t_seq, cfg.c4_docs)

    # --- assemble corpus ---
    parts = [("good", good), ("offdomain", off), ("corrupt", corrupt)]
    tokens = np.concatenate([p[1] for p in parts], axis=0).astype(np.uint16)
    clusters, sources = [], []
    for cname, arr in parts:
        clusters += [cname] * len(arr)
        sources += [{"good": "pubmed", "corrupt": "pubmed-shuffled", "offdomain": "c4"}[cname]] * len(arr)
    # shuffle corpus order, keep aligned metadata
    perm = rng.permutation(len(tokens))
    tokens = tokens[perm]
    clusters = np.asarray(clusters)[perm]
    sources = np.asarray(sources)[perm]

    np.save(os.path.join(out_dir, "tokens.npy"), tokens)
    np.save(os.path.join(out_dir, "val.npy"), val.astype(np.uint16))
    import csv
    with open(os.path.join(out_dir, "meta.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["seq_id", "cluster", "source"])
        for i in range(len(tokens)):
            w.writerow([i, clusters[i], sources[i]])

    stats = dict(
        **asdict(cfg), M=len(tokens), V=len(val),
        tokens_total=int(len(tokens) * cfg.t_seq),
        cluster_counts={c: int((clusters == c).sum()) for c in ("good", "offdomain", "corrupt")},
        pubmed_docs_used=int(ndocs_pm), c4_docs_used=int(ndocs_c4),
        vocab="gpt2", eos_id=int(eos),
    )
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("[data] done:", json.dumps(stats["cluster_counts"]), "M=", stats["M"],
          "V=", stats["V"], "tokens=", stats["tokens_total"], flush=True)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k, v in asdict(DataConfig()).items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args = ap.parse_args()
    build(DataConfig(**vars(args)))
