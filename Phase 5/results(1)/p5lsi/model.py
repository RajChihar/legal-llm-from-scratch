"""LSIModel = Phase-1 backbone (+ PEFT) -> chunk representations -> document -> one logit per statute (multi-label).
(Phase-4 hierarchical model; the 2-way softmax became n_labels independent sigmoids.)

A document is longer than the backbone's context, so it is represented by up to K chunks of <= L tokens
(selection: train.py `select_chunks`), each encoded separately:  H_k = backbone(chunk_k)  [L, C].

Two heads (config head.agg):
  shared document vector (agg = mean | max | attn | gru)                         - the Phase-4 head
      c_k = pool(H_k) (+ chunk-position embedding);  d = agg(c_1..c_K);  logits = W d + b            [B, n_labels]
  label-wise attention (agg = "label_attn")                                     - new for Phase 5
      every statute l has its own query q_l and attends over ALL tokens of the document:
          a_l = softmax_t(q_l . h_t) over the real tokens of every chunk,  d_l = sum_t a_l(t) h_t,  logit_l = w_l . d_l + b_l
      so each statute reads the passage that is evidence for it (statute evidence sits in different places of the
      facts). The softmax over the whole document is computed exactly from per-chunk log-sum-exps, so chunks can
      still be encoded in slices.
The output bias starts at log(p / (1 - p)) of each statute's training frequency, so the model starts by predicting
the prior instead of 0.5 for 100 mostly-negative labels.

Losses (sum over labels, mean over documents):
  bce   binary cross-entropy (+ optional pos_weight per statute)
  asl   asymmetric loss (Ridnik et al., 2021): negatives are down-weighted by p^gamma_neg and probabilities below
        `asl_clip` are discarded (easy negatives of rare statutes stop dominating); positives use gamma_pos.
  KD    a * task + (1 - a) * T^2 * KL(Bernoulli(sigmoid(z_t/T)) || Bernoulli(sigmoid(z_s/T))) per statute
        (Hinton et al., 2015, with independent sigmoids instead of one softmax).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .backbones import load_backbone
from .peft import apply_method


def masked_softmax(scores, mask):
    """scores [..., n] fp32, mask [..., n] bool (at least one True per row)."""
    return torch.softmax(scores.masked_fill(~mask, float("-inf")), -1)


class DocHead(nn.Module):
    def __init__(self, d_model, cfg, n_labels):
        super().__init__()
        self.chunk_pool = cfg.get("chunk_pool", "attn")
        self.agg = cfg.get("agg", "attn")
        assert self.chunk_pool in ("mean", "attn"), self.chunk_pool
        assert self.agg in ("mean", "max", "attn", "gru", "label_attn"), self.agg
        self.n_labels = n_labels
        self.drop = nn.Dropout(cfg.get("dropout", 0.1))
        if self.agg == "label_attn":
            self.tok_norm = nn.LayerNorm(d_model)
            self.label_q = nn.Parameter(torch.randn(n_labels, d_model) * 0.02)
            self.label_w = nn.Parameter(torch.randn(n_labels, d_model) * 0.02)
            self.label_b = nn.Parameter(torch.zeros(n_labels))
            return
        self.tok_score = nn.Linear(d_model, 1) if self.chunk_pool == "attn" else None
        self.n_pos = cfg.get("pos_buckets", 8) if cfg.get("use_pos", True) else 0
        self.pos_emb = nn.Embedding(self.n_pos, d_model) if self.n_pos else None
        if self.pos_emb is not None:
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        if self.tok_score is not None:
            nn.init.zeros_(self.tok_score.weight); nn.init.zeros_(self.tok_score.bias)   # starts as mean pooling
        self.norm = nn.LayerNorm(d_model)
        dim = d_model
        if self.agg == "gru":
            hid = cfg.get("hidden", 256)
            self.rnn = nn.GRU(d_model, hid, batch_first=True, bidirectional=True)
            dim = 2 * hid
        self.chunk_score = nn.Sequential(nn.Linear(dim, cfg.get("att_hidden", 128)), nn.Tanh(),
                                         nn.Linear(cfg.get("att_hidden", 128), 1)) if self.agg in ("attn", "gru") else None
        self.out = nn.Linear(dim, n_labels)

    def init_bias(self, prior):
        """prior [n_labels] training frequency -> output bias log(p/(1-p))."""
        p = torch.as_tensor(prior, dtype=torch.float32).clamp(1e-4, 0.5)
        with torch.no_grad():
            (self.label_b if self.agg == "label_attn" else self.out.bias).copy_(torch.log(p / (1 - p)))

    # ---- per chunk (runs on slices of chunks)
    def pool_tokens(self, H, keep):
        """H [N,T,C], keep [N,T] -> chunk summary.
        shared head: [N,C] fp32;  label_attn: (lse [N,Lb], v [N,Lb,C]) = per-chunk log-sum-exp of the label scores
        and the label-attention-weighted token average inside the chunk."""
        Hf = H.float()
        if self.agg == "label_attn":
            Hn = self.tok_norm(Hf)
            s = torch.einsum("ntc,lc->nlt", Hn, self.label_q) / math.sqrt(Hn.size(-1))
            s = s.masked_fill(~keep[:, None, :], float("-inf"))
            lse = torch.logsumexp(s, -1)                                       # [N, Lb]
            v = torch.einsum("nlt,ntc->nlc", torch.softmax(s, -1), Hn)         # [N, Lb, C]
            return lse, v
        if self.tok_score is None:
            m = keep.float()[..., None]
            return (Hf * m).sum(1) / m.sum(1).clamp(min=1)
        w = masked_softmax(self.tok_score(Hf).squeeze(-1), keep)
        return torch.einsum("nt,ntc->nc", w, Hf)

    # ---- per document
    def forward(self, cvec, doc, slot, cpos, B):
        """cvec: chunk summaries of all chunks; doc/slot/cpos [N] long -> logits [B, n_labels], weights or None."""
        K = int(slot.max().item()) + 1
        mask = torch.zeros(B, K, dtype=torch.bool, device=doc.device)
        mask[doc, slot] = True
        if self.agg == "label_attn":
            lse, v = cvec
            Lb, C = v.shape[1], v.shape[2]
            S = lse.new_full((B, K, Lb), float("-inf"))
            S[doc, slot] = lse
            V = v.new_zeros(B, K, Lb, C)
            V[doc, slot] = v
            w = torch.softmax(S, 1)                    # chunk share of each label's document-level attention
            d = torch.einsum("bkl,bklc->blc", w, V)    # = attention over every token of the document
            logits = (self.drop(d) * self.label_w).sum(-1) + self.label_b
            return logits, w.mean(-1)
        C = cvec.size(1)
        if self.pos_emb is not None:
            cvec = cvec + self.pos_emb(cpos.clamp(0, self.n_pos - 1))
        V = cvec.new_zeros(B, K, C)
        V[doc, slot] = cvec
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


def _cat_summaries(parts):
    if isinstance(parts[0], tuple):
        return tuple(torch.cat(x) for x in zip(*parts))
    return torch.cat(parts)


class LSIModel(nn.Module):
    def __init__(self, backbone, head_cfg, n_labels):
        super().__init__()
        self.backbone = backbone
        self.n_labels = n_labels
        self.head = DocHead(backbone.d_model, head_cfg, n_labels)
        self.register_buffer("pos_weight", torch.ones(n_labels))

    def forward(self, ids, keep, doc, slot, cpos, B, chunk_bs=None):
        """ids/keep [N,T] (all chunks of B documents). chunk_bs: encode chunks in slices (inference memory)."""
        step = chunk_bs if chunk_bs else ids.size(0)
        parts = []
        for i in range(0, ids.size(0), step):
            H = self.backbone(ids[i:i + step], keep[i:i + step])
            with torch.autocast("cuda", enabled=False):                # pooling + head in fp32
                parts.append(self.head.pool_tokens(H, keep[i:i + step]))
        with torch.autocast("cuda", enabled=False):
            logits, w = self.head(_cat_summaries(parts), doc, slot, cpos, B)
        return logits.float(), w

    def task_loss(self, logits, y, cfg):
        """logits, y [B, n_labels] fp32 -> scalar (sum over labels, mean over documents)."""
        name = cfg.get("name", "bce")
        if name == "bce":
            pw = self.pos_weight if cfg.get("pos_weight", "none") != "none" else None
            return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw, reduction="none").sum(1).mean()
        if name == "asl":
            gp, gn, clip = cfg.get("gamma_pos", 0.0), cfg.get("gamma_neg", 4.0), cfg.get("asl_clip", 0.05)
            p = torch.sigmoid(logits)
            p_neg = (p - clip).clamp(min=0) if clip else p                  # probability shifting for negatives
            log_p = F.logsigmoid(logits)
            log_1m = torch.log((1 - p_neg).clamp(min=1e-8))
            lp = y * log_p * ((1 - p) ** gp if gp else 1.0)
            ln = (1 - y) * log_1m * (p_neg ** gn if gn else 1.0)
            return -(lp + ln).sum(1).mean()
        raise ValueError(f"unknown loss {name}")

    def loss(self, logits, y, cfg, teacher=None):
        """cfg: loss section + kd_alpha / kd_T. teacher: teacher logits [B, n_labels] or None."""
        task = self.task_loss(logits, y, cfg)
        if teacher is None:
            return task
        T, a = cfg.get("kd_T", 2.0), cfg.get("kd_alpha", 0.5)
        zs, zt = logits / T, teacher.float() / T
        pt = torch.sigmoid(zt)
        # binary KL(t || s) per label = t (log t - log s) + (1 - t) (log(1-t) - log(1-s))
        kl = pt * (F.logsigmoid(zt) - F.logsigmoid(zs)) + (1 - pt) * (F.logsigmoid(-zt) - F.logsigmoid(-zs))
        return a * task + (1 - a) * T * T * kl.sum(1).mean()


def build_model(cfg, n_labels):
    bcfg = cfg["backbone"]
    bb, info = load_backbone(bcfg["ckpt"], attention=bcfg.get("attention", "bidirectional"),
                             keep_layers=bcfg.get("keep_layers"), c_segment=bcfg.get("c_segment", 0),
                             grad_ckpt=bcfg.get("grad_ckpt", False))
    apply_method(bb, cfg["method"])
    return LSIModel(bb, cfg["head"], n_labels), info


def trainable_state(model):
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k in keep or k.startswith("head.") or k == "pos_weight"}
