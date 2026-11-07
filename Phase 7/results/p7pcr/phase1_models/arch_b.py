"""Architecture B: encoder-decoder.

Copied VERBATIM from the Phase-1 notebook that produced the checkpoint (B/notebookc976077ad3.ipynb),
so parameter names match model_final.pt. Do not edit: Phase-2 code wraps these modules instead."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import RMSNorm, RotaryEmbedding, SwiGLU, _ffn_hidden, init_weights_, lm_loss

# =============================================================================
#  ARCHITECTURE B - encoder-decoder Transformer
# =============================================================================

class SelfAttention(nn.Module):
    """Multi-head self-attention with RoPE. `key_mask` [B,S] bool (True = real token) for the encoder;
    `causal=True` for the decoder. `cache` (decoder generation only) holds past K/V."""
    def __init__(self, d_model, n_head, dropout=0.0, bias=False):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.hd = n_head, d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, rope, pos0=0, key_mask=None, causal=False, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.hd).permute(2, 0, 3, 1, 4)
        q, k = rope(q, pos0), rope(k, pos0)
        if cache is not None:
            if "k" in cache:
                assert T == 1, "cached decoding feeds one token at a time"
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        mask = None if key_mask is None else key_mask[:, None, None, :]           # [B,1,1,S]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal and T > 1 and mask is None,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.resid_drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))


class CrossAttention(nn.Module):
    """Decoder queries attend over encoder memory (no positional rotation; padding keys masked out)."""
    def __init__(self, d_model, n_head, dropout=0.0, bias=False):
        super().__init__()
        self.n_head, self.hd = n_head, d_model // n_head
        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, memory, mem_mask, cache=None):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        if cache is not None and "ck" in cache:
            k, v = cache["ck"], cache["cv"]                                       # memory K/V computed once
        else:
            S = memory.size(1)
            k, v = self.kv(memory).view(B, S, 2, self.n_head, self.hd).permute(2, 0, 3, 1, 4)
            if cache is not None:
                cache["ck"], cache["cv"] = k, v
        mask = None if mem_mask is None else mem_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.resid_drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, dropout, bias):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d_model), RMSNorm(d_model)
        self.attn = SelfAttention(d_model, n_head, dropout, bias)
        self.ffn = SwiGLU(d_model, ffn_hidden, dropout, bias)

    def forward(self, x, rope, key_mask):
        x = x + self.attn(self.norm1(x), rope, 0, key_mask=key_mask, causal=False)
        return x + self.ffn(self.norm2(x))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, dropout, bias):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = RMSNorm(d_model), RMSNorm(d_model), RMSNorm(d_model)
        self.self_attn = SelfAttention(d_model, n_head, dropout, bias)
        self.cross_attn = CrossAttention(d_model, n_head, dropout, bias)
        self.ffn = SwiGLU(d_model, ffn_hidden, dropout, bias)

    def forward(self, x, rope, memory, mem_mask, pos0=0, cache=None):
        x = x + self.self_attn(self.norm1(x), rope, pos0, causal=True, cache=cache)
        x = x + self.cross_attn(self.norm2(x), memory, mem_mask, cache=cache)
        return x + self.ffn(self.norm3(x))


class EncDecLM(nn.Module):
    """
    forward(src [B,S], src_mask [B,S] bool, dec_in [B,T], targets [B,T] | None) -> (logits [B,T,V], loss | None)
    encode(src, src_mask) -> memory [B,S,C];  decode(dec_in, memory, src_mask, pos0, caches, last_only)
    """
    def __init__(self, vocab_size, enc_block, dec_block, pad_id=None, n_enc_layer=6, n_dec_layer=6,
                 n_head=9, d_model=576, ffn_mult=8 / 3, dropout=0.0, rope_theta=10000.0, bias=False, **kw):
        super().__init__()
        if kw:
            print(f"[EncDecLM] ignoring unknown config keys: {sorted(kw)}")
        self.enc_block, self.dec_block, self.pad_id = enc_block, dec_block, pad_id
        self.tok_emb = nn.Embedding(vocab_size, d_model)          # shared by encoder, decoder and LM head
        self.drop = nn.Dropout(dropout)
        hd = d_model // n_head
        self.enc_rope = RotaryEmbedding(hd, enc_block, rope_theta)
        self.dec_rope = RotaryEmbedding(hd, dec_block, rope_theta)
        hidden = _ffn_hidden(d_model, ffn_mult)
        self.enc_blocks = nn.ModuleList([EncoderBlock(d_model, n_head, hidden, dropout, bias) for _ in range(n_enc_layer)])
        self.dec_blocks = nn.ModuleList([DecoderBlock(d_model, n_head, hidden, dropout, bias) for _ in range(n_dec_layer)])
        self.enc_norm, self.dec_norm = RMSNorm(d_model), RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        init_weights_(self, n_enc_layer + n_dec_layer)
        self.lm_head.weight = self.tok_emb.weight

    def encode(self, src, src_mask=None):
        assert src.size(1) <= self.enc_block, f"source length {src.size(1)} > enc_block {self.enc_block}"
        if src_mask is not None and bool(src_mask.all()):
            src_mask = None                                        # no padding -> plain (fast) attention
        x = self.drop(self.tok_emb(src))
        for blk in self.enc_blocks:
            x = blk(x, self.enc_rope, src_mask)
        return self.enc_norm(x)

    def decode(self, dec_in, memory, mem_mask=None, pos0=0, caches=None, last_only=False):
        assert pos0 + dec_in.size(1) <= self.dec_block, f"decoder position {pos0 + dec_in.size(1)} > dec_block"
        if mem_mask is not None and bool(mem_mask.all()):
            mem_mask = None
        x = self.drop(self.tok_emb(dec_in))
        for i, blk in enumerate(self.dec_blocks):
            x = blk(x, self.dec_rope, memory, mem_mask, pos0, None if caches is None else caches[i])
        x = self.dec_norm(x)
        if last_only:
            x = x[:, -1:]
        return self.lm_head(x)

    def forward(self, src, src_mask, dec_in, targets=None):
        memory = self.encode(src, src_mask)
        logits = self.decode(dec_in, memory, src_mask)
        loss = None if targets is None else lm_loss(logits, targets)
        return logits, loss

MODEL_CLASS = EncDecLM
