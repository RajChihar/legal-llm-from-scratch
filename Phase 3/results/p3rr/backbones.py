"""Load a Phase-1 `model_final.pt` and expose it as a token encoder: ids [B,T] -> hidden states [B,T,C].

All four architectures share the block layout  x += attn(norm1(x)); x += ffn(norm2(x)), so one forward loop
serves all of them. Attention is recomputed here from each block's own projection layers so that the mask
can be chosen per experiment:

    attention="causal"         token i sees tokens <= i            (how A and D were pretrained)
    attention="bidirectional"  token i sees every real token       (how B's encoder / C's prefix were pretrained)
    D additionally keeps its sliding window: |i - j| < window (bidirectional) or 0 <= i - j < window (causal).

Padding is on the right and masked out as keys. The LM head is dropped (the task uses its own head).
(Copied unchanged from the Phase-2 pipeline.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .phase1_models import MODEL_CLASSES

ARCH_OF_CLASS = {"LlamaLM": "A", "EncDecLM": "B", "HybridLM": "C", "EfficientLM": "D"}


def _clean_state_dict(sd):
    """Strip prefixes added by torch.compile (`_orig_mod.`) or DataParallel (`module.`)."""
    out = {}
    for k, v in sd.items():
        for p in ("_orig_mod.", "module."):
            while k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


def attention_forward(attn, x, rope, mask, is_causal=False):
    """Self-attention of any Phase-1 attention module. `mask`: bool, broadcastable to [B,1,T,T] (True = attend),
    or None; `is_causal` uses the fused causal kernel (no materialised mask)."""
    B, T, _ = x.shape
    H, hd = attn.n_head, attn.hd
    if hasattr(attn, "qkv"):                                     # B (SelfAttention) and C (HybridSelfAttention)
        q, k, v = attn.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
    else:                                                        # A (GQAttention) and D (LocalMQAttention)
        q = attn.wq(x).view(B, T, H, hd).transpose(1, 2)
        k, v = attn.wkv(x).view(B, T, 2, attn.n_kv, hd).permute(2, 0, 3, 1, 4)
        if attn.n_kv != H:
            rep = H // attn.n_kv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    q, k = rope(q, 0), rope(k, 0)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal and mask is None,
                                       dropout_p=attn.dropout if attn.training else 0.0)
    return attn.resid_drop(attn.proj(y.transpose(1, 2).reshape(B, T, -1)))


class NERBackbone(nn.Module):
    def __init__(self, arch, tok_emb, blocks, norm, rope, max_len, attention="bidirectional", window=None,
                 seg_emb=None, seg_id=0, grad_ckpt=False):
        super().__init__()
        assert attention in ("causal", "bidirectional")
        self.arch, self.attention, self.window, self.max_len = arch, attention, window, max_len
        self.tok_emb, self.blocks, self.norm, self.rope = tok_emb, blocks, norm, rope
        self.seg_emb, self.seg_id = seg_emb, seg_id
        self.d_model = tok_emb.weight.size(1)
        self.grad_ckpt = grad_ckpt
        self.soft_prompt = None                                  # set by peft.apply_method(..., "prompt")

    def build_mask(self, keep):
        """keep [B,T] bool (False = padding) -> (mask or None, is_causal).

        Fast paths (no T x T mask is materialised, so SDPA can use its fused kernels):
          causal, no window : padding is on the right, so a real query never sees a pad key under the causal
                              rule -> plain is_causal=True
          bidirectional     : key-padding mask [B,1,1,T], or None when the batch has no padding
        Only D's sliding window needs an explicit band mask [B,1,T,T].
        """
        no_pad = bool(keep.all())
        if self.window is None:
            if self.attention == "causal":
                return None, True
            return (None if no_pad else keep[:, None, None, :]), False
        T = keep.size(1)
        i = torch.arange(T, device=keep.device)
        rel = i[:, None] - i[None, :]                            # query - key
        allow = rel.abs() < self.window
        if self.attention == "causal":
            allow &= rel >= 0
        return allow[None, None] & keep[:, None, None, :], False

    def forward(self, ids, keep):
        x = self.tok_emb(ids)
        if self.seg_emb is not None:
            x = x + self.seg_emb.weight[self.seg_id]
        n_p = 0
        if self.soft_prompt is not None:                         # prompt tuning: prepend learned vectors
            n_p = self.soft_prompt.size(0)
            x = torch.cat([self.soft_prompt[None].expand(x.size(0), -1, -1).to(x.dtype), x], 1)
            keep = torch.cat([torch.ones(keep.size(0), n_p, dtype=torch.bool, device=keep.device), keep], 1)
        assert x.size(1) <= self.max_len, f"sequence {x.size(1)} > positional table {self.max_len}"
        mask, causal = self.build_mask(keep)
        for blk in self.blocks:
            if self.grad_ckpt and self.training:
                x = checkpoint(self._block, blk, x, mask, causal, use_reentrant=False)
            else:
                x = self._block(blk, x, mask, causal)
        x = self.norm(x)
        return x[:, n_p:]

    def _block(self, blk, x, mask, causal=False):
        x = x + attention_forward(blk.attn, blk.norm1(x), self.rope, mask, causal)
        return x + blk.ffn(blk.norm2(x))


def load_phase1(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cls_name = ck.get("model_class")
    kw = dict(ck["model_kw"])
    if cls_name is None:                                         # fall back to the config keys
        cls_name = ("EncDecLM" if "enc_block" in kw else "HybridLM" if "n_segments" in kw
                    else "EfficientLM" if "window" in kw else "LlamaLM")
    model = MODEL_CLASSES[cls_name](**kw)
    missing, unexpected = model.load_state_dict(_clean_state_dict(ck["model"]), strict=False)
    missing = [m for m in missing if not m.endswith("lm_head.weight")]      # tied to tok_emb
    assert not missing and not unexpected, f"checkpoint mismatch: missing={missing} unexpected={unexpected}"
    return model, cls_name, kw


def load_backbone(ckpt_path, attention="bidirectional", keep_layers=None, c_segment=0, grad_ckpt=False):
    """Returns (NERBackbone, info). keep_layers=k keeps only the first k blocks (smaller KD student)."""
    model, cls_name, kw = load_phase1(ckpt_path)
    arch = ARCH_OF_CLASS[cls_name]
    if arch == "B":
        blocks, norm, rope, max_len, window, seg = model.enc_blocks, model.enc_norm, model.enc_rope, kw["enc_block"], None, None
    elif arch == "D":
        blocks, norm, rope, max_len, window, seg = model.blocks, model.norm_f, model.rope, model.max_pos, model.window, None
    else:
        blocks, norm, rope, max_len, window = model.blocks, model.norm_f, model.rope, kw["block_size"], None
        seg = model.seg_emb if arch == "C" else None
    n_total_layers = len(blocks)
    if keep_layers:
        blocks = nn.ModuleList(list(blocks)[:keep_layers])
    bb = NERBackbone(arch, model.tok_emb, blocks, norm, rope, max_len, attention, window, seg, c_segment, grad_ckpt)
    del model
    info = {"arch": arch, "class": cls_name, "model_kw": {k: v for k, v in kw.items()}, "phase1_layers": n_total_layers,
            "layers_used": len(blocks), "d_model": bb.d_model, "max_len": max_len, "window": window,
            "attention": attention, "params": sum(p.numel() for p in bb.parameters())}
    return bb, info
