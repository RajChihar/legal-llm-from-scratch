"""RRModel = Phase-1 backbone (+ PEFT) -> sentence vectors -> sentence-level context -> per-sentence label scores.

Input: a window of consecutive sentences of one document, concatenated as tokens (no separator: the text reads
like the running judgment the model was pretrained on). sent_idx[b, t] says which sentence token t belongs to.

    token states   H = backbone(ids)                     [B, T, C]   (tokens see the other sentences in the window)
    sentence vecs  V_s = mean_{t in s} H_t                [B, S, C]
                   + position embedding (relative position of the sentence in the document, in buckets)
                   + domain embedding (CL / IT)
    context        BiGRU over the sentences of the window (or none)
    scores         Linear -> [B, S, K]; optionally a linear-chain CRF over the label sequence

Losses: CE (optional class weights, label smoothing), KD = a*CE + (1-a)*T^2*KL(teacher_T || student_T)
(Hinton et al., 2015) for the linear head; sentence-level CRF negative log-likelihood for the CRF head.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .backbones import load_backbone
from .peft import apply_method


class SentenceHead(nn.Module):
    def __init__(self, d_model, n_labels, cfg, n_domains=2):
        super().__init__()
        self.n_pos = cfg.get("pos_buckets", 20) if cfg.get("use_pos", True) else 0
        self.pos_emb = nn.Embedding(self.n_pos, d_model) if self.n_pos else None
        self.dom_emb = nn.Embedding(n_domains, d_model) if cfg.get("use_domain", True) else None
        for e in (self.pos_emb, self.dom_emb):
            if e is not None:
                nn.init.normal_(e.weight, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(cfg.get("dropout", 0.1))
        self.context = cfg.get("context", "gru")
        hid = cfg.get("hidden", 256)
        if self.context == "gru":
            self.rnn = nn.GRU(d_model, hid, batch_first=True, bidirectional=True)
            out_dim = 2 * hid
        elif self.context == "none":
            self.rnn, out_dim = None, d_model
        else:
            raise ValueError(f"unknown context {self.context}")
        self.out = nn.Linear(out_dim, n_labels)

    def forward(self, V, smask, pos, dom):
        """V [B,S,C] float32, smask [B,S] bool (right-padded), pos [B,S] long, dom [B] long -> [B,S,K]."""
        if self.pos_emb is not None:
            V = V + self.pos_emb(pos.clamp(0, self.n_pos - 1))
        if self.dom_emb is not None:
            V = V + self.dom_emb(dom)[:, None]
        x = self.drop(self.norm(V))
        if self.rnn is not None:
            lengths = smask.sum(1).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            y, _ = self.rnn(packed)
            x, _ = pad_packed_sequence(y, batch_first=True, total_length=V.size(1))
            x = self.drop(x)
        return self.out(x)


class CRF(nn.Module):
    """Linear-chain CRF over the sentence labels of a window (no hard constraints: any role may follow any)."""
    def __init__(self, K):
        super().__init__()
        self.K = K
        self.trans = nn.Parameter(torch.zeros(K, K))           # from -> to
        self.start = nn.Parameter(torch.zeros(K))
        self.end = nn.Parameter(torch.zeros(K))

    def nll(self, emis, tags, mask):
        """emis [B,S,K] fp32, tags [B,S] (pads any value), mask [B,S] bool right-padded -> mean NLL per sentence."""
        B, S, K = emis.shape
        tags = tags.clamp(min=0)
        score = self.start[tags[:, 0]] + emis[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, S):
            m = mask[:, t]
            s_t = self.trans[tags[:, t - 1], tags[:, t]] + emis[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            score = score + s_t * m
        lengths = mask.sum(1)
        last = tags.gather(1, (lengths - 1).clamp(min=0)[:, None]).squeeze(1)
        score = score + self.end[last]
        alpha = self.start[None] + emis[:, 0]
        for t in range(1, S):
            nxt = torch.logsumexp(alpha[:, :, None] + self.trans[None], dim=1) + emis[:, t]
            alpha = torch.where(mask[:, t, None], nxt, alpha)
        logZ = torch.logsumexp(alpha + self.end[None], dim=1)
        return ((logZ - score) / lengths.clamp(min=1)).mean()

    def params_np(self):
        return tuple(x.detach().float().cpu().numpy() for x in (self.trans, self.start, self.end))


def _lse(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def viterbi_np(emis, trans, start, end):
    """emis [N,K] -> best label index sequence (whole document)."""
    N, K = emis.shape
    score = start + emis[0]
    back = np.zeros((N, K), dtype=np.int64)
    for t in range(1, N):
        cand = score[:, None] + trans
        back[t] = cand.argmax(0)
        score = cand.max(0) + emis[t]
    score = score + end
    path = [int(score.argmax())]
    for t in range(N - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


def crf_log_marginals_np(emis, trans, start, end):
    """Forward-backward over the whole document: log P(label_t = k | document)  [N,K]."""
    N, K = emis.shape
    a = np.zeros((N, K)); b = np.zeros((N, K))
    a[0] = start + emis[0]
    for t in range(1, N):
        a[t] = _lse(a[t - 1][:, None] + trans, 0) + emis[t]
    b[N - 1] = end
    for t in range(N - 2, -1, -1):
        b[t] = _lse(trans + (emis[t + 1] + b[t + 1])[None], 1)
    logZ = _lse(a[N - 1] + end, 0)
    return (a + b - logZ).astype(np.float32)


class RRModel(nn.Module):
    def __init__(self, backbone, labels, head_cfg):
        super().__init__()
        self.backbone, self.labels = backbone, list(labels)
        K = len(labels)
        self.head = SentenceHead(backbone.d_model, K, head_cfg)
        self.crf = CRF(K) if head_cfg.get("type", "linear") == "crf" else None
        self.register_buffer("class_w", torch.ones(K))

    def sentence_vectors(self, ids, keep, sent_idx, S):
        H = self.backbone(ids, keep)                                   # [B,T,C]
        M = (sent_idx[:, None, :] == torch.arange(S, device=ids.device)[None, :, None]).to(H.dtype)   # [B,S,T]
        V = torch.bmm(M, H).float()
        return V / M.float().sum(-1, keepdim=True).clamp(min=1)

    def emissions(self, ids, keep, sent_idx, smask, pos, dom):
        V = self.sentence_vectors(ids, keep, sent_idx, smask.size(1))
        with torch.autocast("cuda", enabled=False):
            return self.head(V, smask, pos, dom).float()

    def loss(self, emis, labels, smask, cfg, teacher=None):
        """emis [B,S,K] fp32, labels [B,S] (-100 = pad), teacher [B,S,K] log-probs or None."""
        if self.crf is not None:
            return self.crf.nll(emis, labels, smask)
        K = emis.size(-1)
        ce = F.cross_entropy(emis.view(-1, K), labels.view(-1), weight=self.class_w, ignore_index=-100,
                             label_smoothing=cfg.get("label_smoothing", 0.0))
        if teacher is None:
            return ce
        T, a = cfg.get("kd_T", 2.0), cfg.get("kd_alpha", 0.5)
        ok = labels != -100
        s = F.log_softmax(emis[ok] / T, -1)
        t = F.log_softmax(teacher[ok].float() / T, -1)                 # teacher log-probs, re-tempered
        kl = F.kl_div(s, t, log_target=True, reduction="batchmean")
        return a * ce + (1 - a) * T * T * kl


def build_model(cfg, labels):
    bcfg = cfg["backbone"]
    bb, info = load_backbone(bcfg["ckpt"], attention=bcfg.get("attention", "bidirectional"),
                             keep_layers=bcfg.get("keep_layers"), c_segment=bcfg.get("c_segment", 0),
                             grad_ckpt=bcfg.get("grad_ckpt", False))
    apply_method(bb, cfg["method"])
    return RRModel(bb, labels, cfg["head"]), info


def trainable_state(model):
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k in keep or k.startswith("head.") or k.startswith("crf.") or k == "class_w"}
