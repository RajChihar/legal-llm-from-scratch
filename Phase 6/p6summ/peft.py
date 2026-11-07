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
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B)

NF4_CODES = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
             -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
             0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
             0.7229568362236023, 1.0)

def nf4_quantize(w, block=64):
    flat = w.detach().float().reshape(-1)
    assert flat.numel() % block == 0 and block % 2 == 0, f"{flat.numel()} weights not divisible into blocks of {block}"
    blocks = flat.view(-1, block)
    absmax = blocks.abs().amax(1).clamp(min=1e-12)
    codes = torch.tensor(NF4_CODES, dtype=torch.float32, device=flat.device)
    mids = (codes[1:] + codes[:-1]) / 2
    idx = torch.bucketize((blocks / absmax[:, None]).reshape(-1), mids).to(torch.uint8)
    return (idx[0::2] << 4) | idx[1::2], absmax

def nf4_dequantize(packed, absmax, shape, block=64):
    codes = torch.tensor(NF4_CODES, dtype=torch.float32, device=packed.device)
    idx = torch.stack([packed >> 4, packed & 15], 1).reshape(-1).long()
    return (codes[idx].view(-1, block) * absmax[:, None]).view(shape)

class QLoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16, dropout=0.0, block=64):
        super().__init__()
        assert base.bias is None, "Phase-1 linears have no bias"
        self.shape, self.block = tuple(base.weight.shape), block
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight_numel = base.weight.numel()
        packed, absmax = nf4_quantize(base.weight, block)
        self.register_buffer("nf4_packed", packed, persistent=False)
        self.register_buffer("nf4_absmax", absmax, persistent=False)
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def dequantized(self):
        return nf4_dequantize(self.nf4_packed, self.nf4_absmax, self.shape, self.block)

    def forward(self, x):
        h = F.linear(x, self.dequantized().to(x.dtype))
        return h + self.scale * F.linear(F.linear(self.drop(x), self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))

class LoRMALinear(nn.Module):
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
        M = self.lorma_B @ self.lorma_A
        d = self.d
        M2 = torch.cat([M, M], 1).contiguous()
        return M2.as_strided((d, d), (2 * d - 1, 1), storage_offset=d)

    def forward(self, x):
        h = self.base(x)
        if self.variant == "plus":
            return h + self.scale * F.linear(F.linear(h, self.lorma_A), self.lorma_B)
        return F.linear(h, self.rotated().to(h.dtype))

class HoulsbyAdapter(nn.Module):
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

class AdaLoRALinear(nn.Module):
    def __init__(self, base, r=12, alpha=24, dropout=0.0):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.lora_P = nn.Parameter(torch.randn(base.out_features, r) * 0.02)
        self.lora_E = nn.Parameter(torch.zeros(r))
        self.lora_Q = nn.Parameter(torch.randn(r, base.in_features) * 0.02)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.linear(self.drop(x), self.lora_Q.to(x.dtype) * self.lora_E.to(x.dtype)[:, None])
        return self.base(x) + self.scale * F.linear(h, self.lora_P.to(x.dtype))

    def orth_penalty(self):
        P, Q = self.lora_P.float(), self.lora_Q.float()
        I = torch.eye(self.r, device=P.device)
        return ((P.t() @ P - I) ** 2).sum() + ((Q @ Q.t() - I) ** 2).sum()

class AdaLoRAController:
    def __init__(self, model, total_steps, target_r=4, init_warmup=0.1, final_warmup=0.2, delta_t=10, beta1=0.85,
                 beta2=0.85, orth_reg=0.1):
        self.mods = [m for m in model.modules() if isinstance(m, AdaLoRALinear)]
        assert self.mods, "no AdaLoRA modules in the model"
        self.T, self.dt, self.b1, self.b2, self.orth_reg = total_steps, max(1, delta_t), beta1, beta2, orth_reg
        self.t_init, self.t_final = int(init_warmup * total_steps), int(final_warmup * total_steps)
        self.b0 = sum(m.r for m in self.mods)
        self.bT = min(self.b0, target_r * len(self.mods))
        self.ipt, self.unc, self.final_mask = {}, {}, None
        self.budget = self.b0

    def orth_loss(self):
        return self.orth_reg * sum(m.orth_penalty() for m in self.mods) / len(self.mods)

    def schedule(self, t):
        if t < self.t_init:
            return self.b0
        if t >= self.T - self.t_final:
            return self.bT
        span = max(1, self.T - self.t_init - self.t_final)
        return int(self.bT + (self.b0 - self.bT) * (1 - (t - self.t_init) / span) ** 3)

    @torch.no_grad()
    def _score(self, p):
        if p.grad is None:
            return torch.zeros_like(p, dtype=torch.float32)
        ipt = (p.float() * p.grad.float()).abs()
        k = id(p)
        if k not in self.ipt:
            self.ipt[k], self.unc[k] = torch.zeros_like(ipt), torch.zeros_like(ipt)
        self.ipt[k].mul_(self.b1).add_(ipt, alpha=1 - self.b1)
        self.unc[k].mul_(self.b2).add_((ipt - self.ipt[k]).abs(), alpha=1 - self.b2)
        return self.ipt[k] * self.unc[k]

    @torch.no_grad()
    def update(self, step):
        grads = [p.grad for m in self.mods for p in (m.lora_P, m.lora_E, m.lora_Q) if p.grad is not None]
        if not all(bool(torch.isfinite(g).all()) for g in grads):
            return
        scores = [self._score(m.lora_E) + self._score(m.lora_P).mean(0) + self._score(m.lora_Q).mean(1) for m in self.mods]
        self.budget = self.schedule(step)
        if step >= self.T - self.t_final:
            if self.final_mask is None:
                self.final_mask = self._mask(scores, self.bT)
            for m, keep in zip(self.mods, self.final_mask):
                m.lora_E.masked_fill_(~keep, 0.0)
            return
        if step >= self.t_init and step % self.dt == 0:
            for m, keep in zip(self.mods, self._mask(scores, self.budget)):
                m.lora_E.masked_fill_(~keep, 0.0)

    def _mask(self, scores, budget):
        allv = torch.cat(scores)
        if budget >= len(allv):
            return [torch.ones_like(s, dtype=torch.bool) for s in scores]
        thr = torch.kthvalue(allv, len(allv) - budget).values
        return [s > thr for s in scores]

    def ranks(self):
        return [int((m.lora_E != 0).sum()) for m in self.mods]

def _block_linears(blk, which=("attn", "ffn")):
    out = []
    if "attn" in which:
        for an in ("attn", "self_attn", "cross_attn"):
            a = getattr(blk, an, None)
            if a is None:
                continue
            for name in ("wq", "wkv", "qkv", "q", "kv", "proj"):
                if hasattr(a, name):
                    out.append((a, name, "attn_out" if name == "proj" else "attn_in"))
    if "ffn" in which:
        out.append((blk.ffn, "w12", "ffn_in"))
        out.append((blk.ffn, "proj", "ffn_out"))
    return out

def apply_method(model, cfg):
    name = cfg["name"]
    for p in model.parameters():
        p.requires_grad = name == "full"
    if name in ("full", "frozen"):
        return model
    blocks = model.peft_blocks()
    if name == "partial":
        k = cfg.get("train_last_k", 4)
        for blk in model.last_blocks(k):
            for p in blk.parameters():
                p.requires_grad = True
        for p in model.final_norm().parameters():
            p.requires_grad = True
        return model
    if name == "prefix":
        assert model.arch in ("A", "C"), "prefix tuning is implemented for the decoder (A) and prefix-LM (C) models"
        n = cfg.get("n_prefix", 16)
        g = torch.Generator().manual_seed(cfg.get("seed", 0))
        model.prefix_h = nn.ParameterList([
            nn.Parameter(model.tok_emb.weight[torch.randint(0, min(20000, model.tok_emb.num_embeddings), (n,), generator=g)].detach().clone())
            for _ in model.blocks])
        return model
    if name == "prompt":
        assert model.arch in ("A", "C", "D"), "prompt tuning is implemented for A, C and D"
        g = torch.Generator().manual_seed(cfg.get("seed", 0))
        model.soft_prompt = nn.Parameter(model.tok_emb.weight[torch.randint(0, min(20000, model.tok_emb.num_embeddings), (cfg.get("n_prompt", 20),),
                                                                            generator=g)].detach().clone())
        return model
    targets = cfg.get("targets", ["attn", "ffn"])
    for blk in blocks:
        if name in ("lora", "lorma", "qlora", "adalora"):
            for parent, attr, kind in _block_linears(blk, targets):
                base = getattr(parent, attr)
                if name == "qlora":
                    new = QLoRALinear(base, cfg.get("r", 8), cfg.get("alpha", 16), cfg.get("dropout", 0.05),
                                      cfg.get("nf4_block", 64))
                elif name == "lora":
                    new = LoRALinear(base, cfg.get("r", 8), cfg.get("alpha", 16), cfg.get("dropout", 0.05),
                                     cfg.get("rslora", False))
                elif name == "adalora":
                    new = AdaLoRALinear(base, cfg.get("r", 12), cfg.get("alpha", 24), cfg.get("dropout", 0.05))
                else:
                    if cfg.get("variant", "plus") == "pi" and base.out_features > cfg.get("pi_max_dim", 2048):
                        continue
                    new = LoRMALinear(base, cfg.get("r", 8), cfg.get("alpha", 16), cfg.get("variant", "plus"))
                setattr(parent, attr, new)
        elif name == "ia3":
            for an in ("attn", "self_attn", "cross_attn"):
                a = getattr(blk, an, None)
                if a is None:
                    continue
                if hasattr(a, "qkv"):
                    d = a.qkv.out_features // 3
                    a.qkv = IA3Linear(a.qkv, "out", torch.cat([torch.zeros(d), torch.ones(2 * d)]))
                elif hasattr(a, "wkv"):
                    a.wkv = IA3Linear(a.wkv, "out")
                elif hasattr(a, "kv"):
                    a.kv = IA3Linear(a.kv, "out")
            blk.ffn.proj = IA3Linear(blk.ffn.proj, "in")
        elif name in ("houlsby", "adapter", "pfeiffer"):
            kinds = ("ffn_out",) if name == "pfeiffer" else ("attn_out", "ffn_out")
            for parent, attr, kind in _block_linears(blk):
                if kind in kinds:
                    setattr(parent, attr, HoulsbyAdapter(getattr(parent, attr), cfg.get("bottleneck", 64)))
        elif name == "bitfit":
            for parent, attr, kind in _block_linears(blk):
                setattr(parent, attr, BitFitLinear(getattr(parent, attr)))
            for nm in ("norm1", "norm2", "norm3"):
                if hasattr(blk, nm):
                    getattr(blk, nm).weight.requires_grad = True
        else:
            raise ValueError(f"unknown method {name}")
    if name == "bitfit":
        for p in model.final_norm().parameters():
            p.requires_grad = True
    if cfg.get("train_norms", False):
        for blk in blocks:
            for nm in ("norm1", "norm2", "norm3"):
                if hasattr(blk, nm):
                    getattr(blk, nm).weight.requires_grad = True
        for p in model.final_norm().parameters():
            p.requires_grad = True
    return model

def is_new_param(name):
    return any(t in name for t in NEW_PARAM_TAGS)

def count_params(model):
    q = sum(m.weight_numel for m in model.modules() if isinstance(m, QLoRALinear))
    total = sum(p.numel() for p in model.parameters()) + q
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": train, "trainable_pct": round(100 * train / max(1, total), 3), "nf4_weights": q}
