"""Fine-tuning / parameter-efficient adaptation methods, applied in place to a Phase-1 backbone.
(Phase-2 module + rsLoRA and IA3 for Phase 3.)

    full     every backbone parameter trainable
    partial  only the last `train_last_k` blocks (+ final norm) trainable
    frozen   backbone frozen, only the head trains (linear probe: measures raw Phase-1 transfer)
    lora     W x + (alpha/r) B A x                           (Hu et al., 2021)
             rslora=True: scale alpha/sqrt(r) instead of alpha/r  (Kalajdzievski, 2023, rank-stabilised LoRA)
    ia3      learned rescaling vectors: k, v (attention) and the FFN hidden activation   (Liu et al., 2022)
    lorma    LoRMA+ : (I + (alpha/r) B A) W x                 (Bihany, Patel & Modi, 2025)
             LoRMApi: R_pi(B A) W x, rows of BA cyclically rotated (row i by i) -> full rank
    adapter  Houlsby bottleneck after the attention and FFN output projections: y + U act(D y)
    bitfit   zero-initialised bias added to every block linear + RMSNorm gains (the Phase-1 linears have no
             biases, so BitFit's "train only the biases" becomes "add and train only biases")
    prompt   n learned vectors prepended to the input embeddings (Lester et al., 2021)
    prefix   prefix tuning (Li & Liang, 2021): every layer gets n learned virtual-token states; the layer's frozen
             norm + K/V projection turn them into extra keys/values that all real tokens attend to. Learning the
             states instead of raw K/V is the re-parameterisation that keeps training stable.

The task head is always trainable. New PEFT parameters are named lora_*, lorma_*, adapter_*, bitfit_*, ia3_* or
soft_prompt so the trainer can give them their own learning rate.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NEW_PARAM_TAGS = ("lora_", "lorma_", "adapter_", "bitfit_", "ia3_", "soft_prompt", "prefix_h")


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16, dropout=0.0, rslora=False):
        super().__init__()
        self.base, self.scale = base, alpha / (math.sqrt(r) if rslora else r)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))        # B = 0 -> starts as identity
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B)


class LoRMALinear(nn.Module):
    """Multiplicative low-rank update of the frozen output h = W0 x (left multiplication, d_out x d_out).

    variant "plus": h' = h + (alpha/r) B (A h)          B [d,r] = 0, A [r,d] random  -> identity at init.
                    Re-ordered as in the paper: never forms the d x d matrix; cost O(d r) per token.
    variant "pi"  : h' = R(B A) h, R rotates row i of BA cyclically by i positions (rank inflation).
                    Init B[:,0] = 1, A[0,0] = 1, rest of A zero -> BA has ones in column 0 -> R(BA) = I.
    """
    def __init__(self, base, r=8, alpha=16, variant="plus"):
        super().__init__()
        self.base, self.variant, self.scale = base, variant, alpha / r
        d = base.out_features
        self.d = d
        self.lorma_A = nn.Parameter(torch.empty(r, d))
        self.lorma_B = nn.Parameter(torch.empty(d, r))
        if variant == "plus":
            nn.init.kaiming_uniform_(self.lorma_A, a=math.sqrt(5))
            nn.init.zeros_(self.lorma_B)
        elif variant == "pi":
            nn.init.zeros_(self.lorma_A); self.lorma_A.data[0, 0] = 1.0
            nn.init.normal_(self.lorma_B, std=0.02); self.lorma_B.data[:, 0] = 1.0
        else:
            raise ValueError(variant)

    def rotated(self):
        """R(BA)[i, j] = (BA)[i, (j - i) mod d], built with a strided view of [BA | BA] (no index tensor)."""
        M = self.lorma_B @ self.lorma_A                                        # [d, d]
        d = self.d
        M2 = torch.cat([M, M], 1).contiguous()                                 # [d, 2d]
        return M2.as_strided((d, d), (2 * d - 1, 1), storage_offset=d)         # element (i,j) -> M2[i, d+j-i]

    def forward(self, x):
        h = self.base(x)
        if self.variant == "plus":
            return h + self.scale * F.linear(F.linear(h, self.lorma_A), self.lorma_B)
        return F.linear(h, self.rotated().to(h.dtype))


class HoulsbyAdapter(nn.Module):
    """Wraps an output projection: y = base(x); y + up(gelu(down(y))). up = 0 at init -> identity."""
    def __init__(self, base, bottleneck=64):
        super().__init__()
        self.base = base
        d = base.out_features
        self.adapter_down = nn.Linear(d, bottleneck)
        self.adapter_up = nn.Linear(bottleneck, d)
        nn.init.normal_(self.adapter_down.weight, std=0.02); nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight); nn.init.zeros_(self.adapter_up.bias)

    def forward(self, x):
        y = self.base(x)
        return y + self.adapter_up(F.gelu(self.adapter_down(y)))


class IA3Linear(nn.Module):
    """IA3 (Liu et al., 2022): element-wise learned scaling, initialised to 1 (identity).

    mode "out": y = base(x) * s, s = 1 + (l - 1) * mask. mask selects which outputs get a learned scale:
                for a fused qkv projection only the k and v blocks (IA3 rescales keys and values, not queries).
    mode "in" : y = base(x * l)   (scales the FFN hidden activation before the down projection)."""
    def __init__(self, base, mode="out", mask=None):
        super().__init__()
        self.base, self.mode = base, mode
        n = base.out_features if mode == "out" else base.in_features
        self.ia3_l = nn.Parameter(torch.ones(n))
        self.register_buffer("ia3_mask", torch.ones(n) if mask is None else mask.float(), persistent=False)

    def forward(self, x):
        if self.mode == "in":
            return self.base(x * self.ia3_l.to(x.dtype))
        s = 1 + (self.ia3_l - 1) * self.ia3_mask
        return self.base(x) * s.to(x.dtype)


class BitFitLinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.bitfit_bias = nn.Parameter(torch.zeros(base.out_features))

    def forward(self, x):
        return self.base(x) + self.bitfit_bias


def _block_linears(blk, which=("attn", "ffn")):
    """(parent_module, attribute_name, kind) for every linear inside a Phase-1 block."""
    out = []
    if "attn" in which:
        for a in ("wq", "wkv", "qkv", "proj"):
            if hasattr(blk.attn, a):
                out.append((blk.attn, a, "attn_out" if a == "proj" else "attn_in"))
    if "ffn" in which:
        out.append((blk.ffn, "w12", "ffn_in"))
        out.append((blk.ffn, "proj", "ffn_out"))
    return out


def apply_method(bb, cfg):
    """cfg: the `method` section of the run config. Freezes / wraps backbone modules in place."""
    name = cfg["name"]
    for p in bb.parameters():
        p.requires_grad = name == "full"
    if name in ("full", "frozen"):
        return bb
    if name == "partial":
        k = cfg.get("train_last_k", 4)
        for blk in list(bb.blocks)[-k:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in bb.norm.parameters():
            p.requires_grad = True
        return bb
    if name == "prefix":
        n = cfg.get("n_prefix", 16)
        g = torch.Generator().manual_seed(cfg.get("seed", 0))
        bb.prefix_h = nn.ParameterList([
            nn.Parameter(bb.tok_emb.weight[torch.randint(0, 20000, (n,), generator=g)].detach().clone())
            for _ in bb.blocks])
        return bb
    if name == "prompt":
        bb.soft_prompt = nn.Parameter(bb.tok_emb.weight[torch.randint(0, 20000, (cfg.get("n_prompt", 20),))].detach().clone())
        return bb
    targets = cfg.get("targets", ["attn", "ffn"])
    for blk in bb.blocks:
        if name in ("lora", "lorma"):
            for parent, attr, kind in _block_linears(blk, targets):
                base = getattr(parent, attr)
                if name == "lora":
                    setattr(parent, attr, LoRALinear(base, cfg.get("r", 8), cfg.get("alpha", 16), cfg.get("dropout", 0.05),
                                                     cfg.get("rslora", False)))
                else:
                    if cfg.get("variant", "plus") == "pi" and base.out_features > cfg.get("pi_max_dim", 2048):
                        continue                                  # d x d rotation too large for this layer
                    setattr(parent, attr, LoRMALinear(base, cfg.get("r", 8), cfg.get("alpha", 16), cfg.get("variant", "plus")))
        elif name == "ia3":
            a = blk.attn
            if hasattr(a, "qkv"):                                # fused [q | k | v], each d_model wide
                d = a.qkv.out_features // 3
                mask = torch.cat([torch.zeros(d), torch.ones(2 * d)])
                a.qkv = IA3Linear(a.qkv, "out", mask)
            else:                                                # wkv = [k | v]
                a.wkv = IA3Linear(a.wkv, "out")
            blk.ffn.proj = IA3Linear(blk.ffn.proj, "in")
        elif name == "adapter":
            for parent, attr, kind in _block_linears(blk):
                if kind in ("attn_out", "ffn_out"):
                    setattr(parent, attr, HoulsbyAdapter(getattr(parent, attr), cfg.get("bottleneck", 64)))
        elif name == "bitfit":
            for parent, attr, kind in _block_linears(blk):
                setattr(parent, attr, BitFitLinear(getattr(parent, attr)))
            for p in (blk.norm1.weight, blk.norm2.weight):
                p.requires_grad = True
        else:
            raise ValueError(f"unknown method {name}")
    if name == "bitfit":
        bb.norm.weight.requires_grad = True
    if cfg.get("train_norms", False):
        for blk in bb.blocks:
            blk.norm1.weight.requires_grad = blk.norm2.weight.requires_grad = True
        bb.norm.weight.requires_grad = True
    return bb


def is_new_param(name):
    return any(t in name for t in NEW_PARAM_TAGS)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": train, "trainable_pct": round(100 * train / total, 3)}
