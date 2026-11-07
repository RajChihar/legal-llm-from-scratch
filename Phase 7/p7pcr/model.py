import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import load_backbone
from .peft import apply_method

def masked_softmax(scores, mask):
    return torch.softmax(scores.masked_fill(~mask, float("-inf")), -1)

class Retriever(nn.Module):
    def __init__(self, backbone, rcfg):
        super().__init__()
        self.backbone = backbone
        C = backbone.d_model
        self.pool, self.agg, self.sim = rcfg.get("pool", "mean"), rcfg.get("agg", "mean"), rcfg.get("sim", "cos")
        assert self.pool in ("mean", "attn", "last") and self.agg in ("mean", "max", "attn") and self.sim in ("cos", "maxsim")
        self.head = nn.Module()
        if self.pool == "attn":
            self.head.tok_score = nn.Linear(C, 1)
            nn.init.zeros_(self.head.tok_score.weight); nn.init.zeros_(self.head.tok_score.bias)
        d = rcfg.get("proj") or 0
        if d:
            self.head.proj = nn.Linear(C, d, bias=False)
            nn.init.normal_(self.head.proj.weight, std=C ** -0.5)
        if self.agg == "attn":
            self.head.chunk_score = nn.Linear(d or C, 1)
            nn.init.zeros_(self.head.chunk_score.weight); nn.init.zeros_(self.head.chunk_score.bias)
        tau = float(rcfg.get("tau", 0.05))
        self.head.log_tau = nn.Parameter(torch.tensor(tau).log(), requires_grad=bool(rcfg.get("learn_tau", False)))
        self.dim = d or C

    @property
    def tau(self):
        return self.head.log_tau.exp().clamp(0.005, 1.0)

    def chunk_vectors(self, ids, keep, chunk_bs=None):
        step = chunk_bs or ids.size(0)
        out = []
        for i in range(0, ids.size(0), step):
            H = self.backbone(ids[i:i + step], keep[i:i + step])
            k = keep[i:i + step]
            with torch.autocast("cuda", enabled=False):
                Hf = H.float()
                if self.pool == "mean":
                    m = k.float()[..., None]
                    v = (Hf * m).sum(1) / m.sum(1).clamp(min=1)
                elif self.pool == "attn":
                    w = masked_softmax(self.head.tok_score(Hf).squeeze(-1), k)
                    v = torch.einsum("nt,ntc->nc", w, Hf)
                else:
                    v = Hf[torch.arange(Hf.size(0), device=Hf.device), k.sum(1) - 1]
                if hasattr(self.head, "proj"):
                    v = self.head.proj(v)
                out.append(F.normalize(v, dim=-1))
        return torch.cat(out)

    def doc_reps(self, cvec, doc, slot, B):
        K = int(slot.max()) + 1
        mask = torch.zeros(B, K, dtype=torch.bool, device=cvec.device)
        mask[doc, slot] = True
        X = cvec.new_zeros(B, K, cvec.size(1))
        X[doc, slot] = cvec
        with torch.autocast("cuda", enabled=False):
            if self.agg == "mean":
                m = mask.float()[..., None]
                d = (X * m).sum(1) / m.sum(1).clamp(min=1)
            elif self.agg == "max":
                d = X.masked_fill(~mask[..., None], float("-inf")).max(1).values
            else:
                w = masked_softmax(self.head.chunk_score(X).squeeze(-1), mask)
                d = torch.einsum("bk,bkc->bc", w, X)
        return {"doc": F.normalize(d, dim=-1), "chunks": X, "mask": mask}

    def encode(self, batch, chunk_bs=None):
        cv = self.chunk_vectors(batch["ids"], batch["keep"], chunk_bs)
        return self.doc_reps(cv, batch["doc"], batch["slot"], batch["n_docs"])

    def scores(self, q, c, sim=None):
        return similarity(q, c, sim or self.sim) / self.tau

def similarity(q, c, sim="cos", block=64):
    if sim == "cos":
        return q["doc"].float() @ c["doc"].float().t()
    out = []
    for i in range(0, q["doc"].size(0), block):
        Q, Qm = q["chunks"][i:i + block].float(), q["mask"][i:i + block]
        s = torch.einsum("akd,bjd->abkj", Q, c["chunks"].float())
        s = s.masked_fill(~c["mask"][None, :, None, :], float("-inf")).max(-1).values
        qm = Qm[:, None, :].float()
        out.append((s.masked_fill(qm == 0, 0) * qm).sum(-1) / qm.sum(-1).clamp(min=1))
    return torch.cat(out)

def ranking_loss(S, target, allowed, positives, name="infonce", own_negs=None):
    if name == "infonce":
        return F.cross_entropy(S.masked_fill(~allowed, float("-inf")), target)
    if name == "multipos":
        A = allowed | positives
        lse_all = torch.logsumexp(S.masked_fill(~A, float("-inf")), -1)
        lse_pos = torch.logsumexp(S.masked_fill(~positives, float("-inf")), -1)
        return (lse_all - lse_pos).mean()
    if name == "pairwise":
        sp = S.gather(1, target[:, None])
        diff = (sp - S).masked_fill(~own_negs, 0.0)
        n = own_negs.float().sum().clamp(min=1)
        return (F.softplus(-diff) * own_negs.float()).sum() / n
    raise ValueError(f"unknown loss {name}")

def kd_loss(S, T, allowed):
    Sm, Tm = S.float().masked_fill(~allowed, -1e9), T.float().masked_fill(~allowed, -1e9)
    lp_t, lp_s = F.log_softmax(Tm, -1), F.log_softmax(Sm, -1)
    kl = torch.where(allowed, lp_t.exp() * (lp_t - lp_s), torch.zeros_like(lp_s))
    return kl.sum(-1).mean()

def build_model(cfg):
    bcfg = cfg["backbone"]
    bb, info = load_backbone(bcfg["ckpt"], attention=bcfg.get("attention", "bidirectional"),
                             keep_layers=bcfg.get("keep_layers"), c_segment=bcfg.get("c_segment", 0),
                             grad_ckpt=bcfg.get("grad_ckpt", False))
    apply_method(bb, cfg["method"])
    return Retriever(bb, cfg["rep"]), info

def trainable_state(model):
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep or k.startswith("head.")}
