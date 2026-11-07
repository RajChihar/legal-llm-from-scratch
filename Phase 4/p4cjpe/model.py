import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .backbones import load_backbone
from .peft import apply_method

def masked_softmax(scores, mask):
    return torch.softmax(scores.masked_fill(~mask, float("-inf")), -1)

class DocHead(nn.Module):
    def __init__(self, d_model, cfg, n_classes=2):
        super().__init__()
        self.chunk_pool = cfg.get("chunk_pool", "attn")
        self.agg = cfg.get("agg", "attn")
        assert self.chunk_pool in ("mean", "attn"), self.chunk_pool
        assert self.agg in ("mean", "max", "attn", "gru"), self.agg
        self.tok_score = nn.Linear(d_model, 1) if self.chunk_pool == "attn" else None
        self.n_pos = cfg.get("pos_buckets", 8) if cfg.get("use_pos", True) else 0
        self.pos_emb = nn.Embedding(self.n_pos, d_model) if self.n_pos else None
        if self.pos_emb is not None:
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        if self.tok_score is not None:
            nn.init.zeros_(self.tok_score.weight); nn.init.zeros_(self.tok_score.bias)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(cfg.get("dropout", 0.1))
        dim = d_model
        if self.agg == "gru":
            hid = cfg.get("hidden", 256)
            self.rnn = nn.GRU(d_model, hid, batch_first=True, bidirectional=True)
            dim = 2 * hid
        self.chunk_score = nn.Sequential(nn.Linear(dim, cfg.get("att_hidden", 128)), nn.Tanh(),
                                         nn.Linear(cfg.get("att_hidden", 128), 1)) if self.agg in ("attn", "gru") else None
        self.out = nn.Linear(dim, n_classes)

    def pool_tokens(self, H, keep):
        Hf = H.float()
        if self.tok_score is None:
            m = keep.float()[..., None]
            return (Hf * m).sum(1) / m.sum(1).clamp(min=1)
        w = masked_softmax(self.tok_score(Hf).squeeze(-1), keep)
        return torch.einsum("nt,ntc->nc", w, Hf)

    def forward(self, cvec, doc, slot, cpos, B):
        K = int(slot.max().item()) + 1
        C = cvec.size(1)
        if self.pos_emb is not None:
            cvec = cvec + self.pos_emb(cpos.clamp(0, self.n_pos - 1))
        V = cvec.new_zeros(B, K, C)
        V[doc, slot] = cvec
        mask = torch.zeros(B, K, dtype=torch.bool, device=cvec.device)
        mask[doc, slot] = True
        x = self.drop(self.norm(V))
        if self.agg == "gru":
            packed = pack_padded_sequence(x, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
            y, _ = self.rnn(packed)
            x, _ = pad_packed_sequence(y, batch_first=True, total_length=K)
            x = self.drop(x)
        w = None
        if self.agg == "mean":
            m = mask.float()[..., None]
            d = (x * m).sum(1) / m.sum(1).clamp(min=1)
        elif self.agg == "max":
            d = x.masked_fill(~mask[..., None], float("-inf")).max(1).values
        else:
            w = masked_softmax(self.chunk_score(x).squeeze(-1), mask)
            d = torch.einsum("bk,bkc->bc", w, x)
        return self.out(self.drop(d)), w

class CJPEModel(nn.Module):
    def __init__(self, backbone, head_cfg):
        super().__init__()
        self.backbone = backbone
        self.head = DocHead(backbone.d_model, head_cfg)
        self.register_buffer("class_w", torch.ones(2))

    def forward(self, ids, keep, doc, slot, cpos, B, chunk_bs=None):
        step = chunk_bs if chunk_bs else ids.size(0)
        cvs = []
        for i in range(0, ids.size(0), step):
            H = self.backbone(ids[i:i + step], keep[i:i + step])
            with torch.autocast("cuda", enabled=False):
                cvs.append(self.head.pool_tokens(H, keep[i:i + step]))
        with torch.autocast("cuda", enabled=False):
            logits, w = self.head(torch.cat(cvs), doc, slot, cpos, B)
        return logits.float(), w

    def loss(self, logits, y, cfg, teacher=None):
        ce = F.cross_entropy(logits, y, weight=self.class_w, label_smoothing=cfg.get("label_smoothing", 0.0))
        if teacher is None:
            return ce
        T, a = cfg.get("kd_T", 2.0), cfg.get("kd_alpha", 0.5)
        s = F.log_softmax(logits / T, -1)
        t = F.log_softmax(teacher.float() / T, -1)
        kl = F.kl_div(s, t, log_target=True, reduction="batchmean")
        return a * ce + (1 - a) * T * T * kl

def build_model(cfg):
    bcfg = cfg["backbone"]
    bb, info = load_backbone(bcfg["ckpt"], attention=bcfg.get("attention", "bidirectional"),
                             keep_layers=bcfg.get("keep_layers"), c_segment=bcfg.get("c_segment", 0),
                             grad_ckpt=bcfg.get("grad_ckpt", False))
    apply_method(bb, cfg["method"])
    return CJPEModel(bb, cfg["head"]), info

def trainable_state(model):
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k in keep or k.startswith("head.") or k == "class_w"}
