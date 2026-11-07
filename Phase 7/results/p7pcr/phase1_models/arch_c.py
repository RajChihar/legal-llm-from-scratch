"""Architecture C: encoder/hybrid prefix-LM + MLM (UniLM-style).

Copied VERBATIM from the Phase-1 notebook that produced the checkpoint (C/encoder-hybrid.ipynb),
so parameter names match model_final.pt. Do not edit: Phase-2 code wraps these modules instead."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import RMSNorm, RotaryEmbedding, SwiGLU, _ffn_hidden, init_weights_, lm_loss

# =============================================================================
#  ARCHITECTURE C - encoder/hybrid LM (UniLM-style shared stack)
# =============================================================================

class HybridSelfAttention(nn.Module):
    """Self-attention driven by an explicit keep-mask (True = may attend), or plain causal / full attention.
    `cache` is used during generation: prefill stores the prompt K/V, decode steps append one token."""
    def __init__(self, d_model, n_head, dropout=0.0, bias=False):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.hd = n_head, d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, rope, attn_mask=None, causal=True, pos0=0, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.hd).permute(2, 0, 3, 1, 4)
        q, k = rope(q, pos0), rope(k, pos0)
        if cache is not None:
            if "k" in cache:
                assert T == 1, "cached decoding feeds one token at a time"
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                           is_causal=causal and attn_mask is None and T > 1,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.resid_drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))


class Block(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, dropout, bias):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d_model), RMSNorm(d_model)
        self.attn = HybridSelfAttention(d_model, n_head, dropout, bias)
        self.ffn = SwiGLU(d_model, ffn_hidden, dropout, bias)

    def forward(self, x, rope, attn_mask, causal, pos0=0, cache=None):
        x = x + self.attn(self.norm1(x), rope, attn_mask, causal, pos0, cache)
        return x + self.ffn(self.norm2(x))


class HybridLM(nn.Module):
    """
    forward(idx [B,T], prefix_len [B] | None, seg [B,T] | None, targets [B,T] | None,
            pos0=0, caches=None, last_only=False) -> (logits, loss | None)
      * attention rule for sample b: query i may attend key j  iff  j < prefix_len[b]  or  j <= i
        prefix_len = 0 -> causal LM;  prefix_len = T -> fully bidirectional encoder (MLM samples)
      * loss = CE(logits, targets, ignore_index=-100) (targets already encode which positions are supervised)
    """
    def __init__(self, vocab_size, block_size, pad_id=None, n_layer=10, n_head=10, d_model=640, ffn_mult=8 / 3,
                 dropout=0.0, rope_theta=10000.0, tie_weights=True, bias=False, n_segments=3, **kw):
        super().__init__()
        if kw:
            print(f"[HybridLM] ignoring unknown config keys: {sorted(kw)}")
        self.block_size, self.pad_id, self.n_layer = block_size, pad_id, n_layer
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.seg_emb = nn.Embedding(n_segments, d_model)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(d_model // n_head, block_size, rope_theta)
        hidden = _ffn_hidden(d_model, ffn_mult)
        self.blocks = nn.ModuleList([Block(d_model, n_head, hidden, dropout, bias) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        init_weights_(self, n_layer)
        nn.init.zeros_(self.seg_emb.weight)                       # starts as a no-op, learned from there
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    @staticmethod
    def hybrid_mask(prefix_len, T):
        """Bool [B,1,T,T], True = query i may attend key j: (j < prefix_len[b]) or (j <= i)."""
        pos = torch.arange(T, device=prefix_len.device)
        in_prefix = pos[None, None, :] < prefix_len[:, None, None]      # [B,1,T]  key j in prefix
        causal = pos[None, :, None] >= pos[None, None, :]               # [1,T,T]  j <= i
        return (in_prefix | causal)[:, None]

    def forward(self, idx, prefix_len=None, seg=None, targets=None, pos0=0, caches=None, last_only=False):
        B, T = idx.shape
        assert pos0 + T <= self.block_size, f"position {pos0 + T} > block_size {self.block_size}"
        if prefix_len is None:
            prefix_len = torch.zeros(B, dtype=torch.long, device=idx.device)
        prefix_len = prefix_len.to(idx.device).long().clamp(0, T)
        if bool((prefix_len == 0).all()):
            attn_mask, causal = None, True              # pure causal -> fused is_causal kernel
        elif bool((prefix_len == T).all()):
            attn_mask, causal = None, False             # pure bidirectional -> no mask at all
        else:
            attn_mask, causal = self.hybrid_mask(prefix_len, T), False
        if seg is None:
            seg = (torch.arange(T, device=idx.device)[None, :] >= prefix_len[:, None]).long()
        x = self.drop(self.tok_emb(idx) + self.seg_emb(seg))
        for i, blk in enumerate(self.blocks):
            x = blk(x, self.rope, attn_mask, causal, pos0, None if caches is None else caches[i])
        x = self.norm_f(x)
        if last_only:
            x = x[:, -1:]
        logits = self.lm_head(x)
        loss = None if targets is None else lm_loss(logits, targets)
        return logits, loss

MODEL_CLASS = HybridLM
