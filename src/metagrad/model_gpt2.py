"""Minimal GPT-2 in pure JAX (functional params), for metagradient unrolling.

Params are a plain pytree of jnp arrays so we can differentiate through an
unrolled optimizer cleanly. Matches HF `gpt2` weights (Conv1D weights stored
as [in, out]; gelu_new tanh approximation; tied output embedding).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import jax, jax.numpy as jnp


@dataclass(frozen=True)
class GPT2Config:
    vocab: int = 50257
    n_ctx: int = 1024
    d: int = 768
    n_layer: int = 12
    n_head: int = 12

    @property
    def head_dim(self):
        return self.d // self.n_head


# ---------------------------------------------------------------- model fns
def gelu_new(x):
    return 0.5 * x * (1.0 + jnp.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * g + b


def _attn(x, blk, n_head):
    B, T, d = x.shape
    hd = d // n_head
    qkv = x @ blk["attn_c_attn_w"] + blk["attn_c_attn_b"]      # [B,T,3d]
    q, k, v = jnp.split(qkv, 3, axis=-1)
    sh = lambda t: t.reshape(B, T, n_head, hd).transpose(0, 2, 1, 3)  # [B,nh,T,hd]
    q, k, v = sh(q), sh(k), sh(v)
    att = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(hd).astype(x.dtype)  # [B,nh,T,T]
    mask = jnp.tril(jnp.ones((T, T), bool))
    att = jnp.where(mask, att, -1e9)   # moderate (not finfo.min) for stable 2nd-order grad
    att = jax.nn.softmax(att, axis=-1)
    o = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, d)       # [B,T,d]
    return o @ blk["attn_c_proj_w"] + blk["attn_c_proj_b"]


def _mlp(x, blk):
    h = gelu_new(x @ blk["mlp_c_fc_w"] + blk["mlp_c_fc_b"])
    return h @ blk["mlp_c_proj_w"] + blk["mlp_c_proj_b"]


def forward(params, ids, cfg: GPT2Config):
    """ids [B,T] int -> logits [B,T,vocab]."""
    B, T = ids.shape
    x = params["wte"][ids] + params["wpe"][:T]
    for blk in params["blocks"]:
        x = x + _attn(layernorm(x, blk["ln1_g"], blk["ln1_b"]), blk, cfg.n_head)
        x = x + _mlp(layernorm(x, blk["ln2_g"], blk["ln2_b"]), blk)
    x = layernorm(x, params["lnf_g"], params["lnf_b"])
    return x @ params["wte"].T


def loss_per_example(params, ids, cfg: GPT2Config):
    """Mean next-token CE per sequence -> [B]."""
    logits = forward(params, ids, cfg)[:, :-1, :]
    tgt = ids[:, 1:]
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    tok = jnp.take_along_axis(logp, tgt[..., None], axis=-1)[..., 0]  # [B,T-1]
    return -tok.mean(axis=-1)


def loss_mean(params, ids, cfg: GPT2Config):
    return loss_per_example(params, ids, cfg).mean()


# ---------------------------------------------------------------- params I/O
def _block(np_get, i):
    g = lambda n: jnp.asarray(np_get(f"h.{i}.{n}"), jnp.float32)
    return dict(
        ln1_g=g("ln_1.weight"), ln1_b=g("ln_1.bias"),
        ln2_g=g("ln_2.weight"), ln2_b=g("ln_2.bias"),
        attn_c_attn_w=g("attn.c_attn.weight"), attn_c_attn_b=g("attn.c_attn.bias"),
        attn_c_proj_w=g("attn.c_proj.weight"), attn_c_proj_b=g("attn.c_proj.bias"),
        mlp_c_fc_w=g("mlp.c_fc.weight"), mlp_c_fc_b=g("mlp.c_fc.bias"),
        mlp_c_proj_w=g("mlp.c_proj.weight"), mlp_c_proj_b=g("mlp.c_proj.bias"),
    )


def load_pretrained(name="gpt2"):
    """Load HF gpt2 weights into our pytree. Returns (params, cfg)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    path = hf_hub_download(name, "model.safetensors")
    store = {}
    with safe_open(path, framework="numpy") as f:
        for k in f.keys():
            if k.endswith(".attn.bias") or k.endswith(".attn.masked_bias"):
                continue  # causal-mask buffer, recomputed
            store[k] = f.get_tensor(k)
    cfg_map = {"gpt2": (12, 768, 12), "gpt2-medium": (24, 1024, 16),
               "gpt2-large": (36, 1280, 20), "gpt2-xl": (48, 1600, 25)}
    n_layer, d, n_head = cfg_map[name]
    cfg = GPT2Config(vocab=store["wte.weight"].shape[0], d=d, n_layer=n_layer, n_head=n_head)
    get = lambda n: store[n]
    params = dict(
        wte=jnp.asarray(get("wte.weight"), jnp.float32),
        wpe=jnp.asarray(get("wpe.weight"), jnp.float32),
        blocks=[_block(get, i) for i in range(n_layer)],
        lnf_g=jnp.asarray(get("ln_f.weight"), jnp.float32),
        lnf_b=jnp.asarray(get("ln_f.bias"), jnp.float32),
    )
    return params, cfg


def init_random(cfg: GPT2Config, key, scale=0.02):
    """Small random init (for unit tests / toy models)."""
    ks = jax.random.split(key, 4 + cfg.n_layer)
    n = lambda k, shape: jax.random.normal(k, shape, jnp.float32) * scale
    blocks = []
    for i in range(cfg.n_layer):
        bk = jax.random.split(ks[4 + i], 6)
        blocks.append(dict(
            ln1_g=jnp.ones(cfg.d), ln1_b=jnp.zeros(cfg.d),
            ln2_g=jnp.ones(cfg.d), ln2_b=jnp.zeros(cfg.d),
            attn_c_attn_w=n(bk[0], (cfg.d, 3 * cfg.d)), attn_c_attn_b=jnp.zeros(3 * cfg.d),
            attn_c_proj_w=n(bk[1], (cfg.d, cfg.d)), attn_c_proj_b=jnp.zeros(cfg.d),
            mlp_c_fc_w=n(bk[2], (cfg.d, 4 * cfg.d)), mlp_c_fc_b=jnp.zeros(4 * cfg.d),
            mlp_c_proj_w=n(bk[3], (4 * cfg.d, cfg.d)), mlp_c_proj_b=jnp.zeros(cfg.d),
        ))
    return dict(
        wte=n(ks[0], (cfg.vocab, cfg.d)), wpe=n(ks[1], (cfg.n_ctx, cfg.d)),
        blocks=blocks, lnf_g=jnp.ones(cfg.d), lnf_b=jnp.zeros(cfg.d),
    )
