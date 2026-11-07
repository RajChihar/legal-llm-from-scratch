"""Architecture A: decoder-only LLaMA-style (GQA, RoPE, SwiGLU).

Copied VERBATIM from the Phase-1 notebook that produced the checkpoint (decoder-only-fine-tuned.ipynb / decoder-only-transformer-llama-style.ipynb),
so parameter names match model_final.pt. Do not edit: Phase-2 code wraps these modules instead."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import RMSNorm, RotaryEmbedding, SwiGLU, _ffn_hidden, init_weights_, lm_loss

# =============================================================================
#  ARCHITECTURE A - decoder-only LLaMA-style Transformer
# =============================================================================

class GQAttention(nn.Module):
    """Causal multi-head attention with grouped-query K/V heads and an optional KV-cache.

    cache: dict per layer used only during generation. Prefill (empty cache, T>1) is causal;
    a single-token decode step attends to every cached position, so it needs no mask."""
    def __init__(self, d_model, n_head, n_kv_head, dropout=0.0, bias=False):
        super().__init__()
        assert d_model % n_head == 0 and n_head % n_kv_head == 0
        self.n_head, self.n_kv, self.hd = n_head, n_kv_head, d_model // n_head
        self.wq = nn.Linear(d_model, n_head * self.hd, bias=bias)
        self.wkv = nn.Linear(d_model, 2 * n_kv_head * self.hd, bias=bias)
        self.proj = nn.Linear(n_head * self.hd, d_model, bias=bias)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, rope, pos0=0, cache=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.hd).transpose(1, 2)              # [B,H,T,hd]
        k, v = self.wkv(x).view(B, T, 2, self.n_kv, self.hd).permute(2, 0, 3, 1, 4)  # [B,Hkv,T,hd] each
        q, k = rope(q, pos0), rope(k, pos0)
        if cache is not None:
            if "k" in cache:
                assert T == 1, "cached decoding feeds one token at a time"
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        if self.n_kv != self.n_head:                       # share each K/V head across a group of query heads
            rep = self.n_head // self.n_kv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.resid_drop(self.proj(y.transpose(1, 2).reshape(B, T, -1)))


class Block(nn.Module):
    """Pre-norm residual block: x + Attn(Norm(x)); x + SwiGLU(Norm(x))."""
    def __init__(self, d_model, n_head, n_kv_head, ffn_hidden, dropout, bias):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d_model), RMSNorm(d_model)
        self.attn = GQAttention(d_model, n_head, n_kv_head, dropout, bias)
        self.ffn = SwiGLU(d_model, ffn_hidden, dropout, bias)

    def forward(self, x, rope, pos0=0, cache=None):
        x = x + self.attn(self.norm1(x), rope, pos0, cache)
        return x + self.ffn(self.norm2(x))


class LlamaLM(nn.Module):
    """
    forward(idx [B,T], targets [B,T] | None, pos0=0, caches=None, last_only=False) -> (logits, loss | None)
      * training: causal LM over every position, loss = CE(logits, targets)
      * generation: pass `caches` (one dict per layer) and the absolute position `pos0` of idx[:, 0]
    """
    def __init__(self, vocab_size, block_size, pad_id=None, n_layer=12, n_head=10, n_kv_head=2, d_model=640,
                 ffn_mult=8 / 3, dropout=0.0, rope_theta=10000.0, tie_weights=True, bias=False, **kw):
        super().__init__()
        if kw:
            print(f"[LlamaLM] ignoring unknown config keys: {sorted(kw)}")
        self.block_size, self.pad_id, self.n_layer = block_size, pad_id, n_layer
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(d_model // n_head, block_size, rope_theta)
        hidden = _ffn_hidden(d_model, ffn_mult)
        self.blocks = nn.ModuleList([Block(d_model, n_head, n_kv_head, hidden, dropout, bias) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        init_weights_(self, n_layer)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None, pos0=0, caches=None, last_only=False):
        B, T = idx.shape
        assert pos0 + T <= self.block_size, f"position {pos0 + T} > block_size {self.block_size}"
        x = self.drop(self.tok_emb(idx))
        for i, blk in enumerate(self.blocks):
            x = blk(x, self.rope, pos0, None if caches is None else caches[i])
        x = self.norm_f(x)
        if last_only:
            x = x[:, -1:]
        logits = self.lm_head(x)
        loss = None if targets is None else lm_loss(logits, targets)
        return logits, loss

MODEL_CLASS = LlamaLM
