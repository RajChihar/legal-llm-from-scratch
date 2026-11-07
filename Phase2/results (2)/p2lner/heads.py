"""Prediction heads on top of the backbone's hidden states.

linear : dropout + Linear(d_model -> n_tags); token-level softmax, decoded with BIO-constrained Viterbi.
crf    : same emissions + a linear-chain CRF (learned tag-transition scores, BIO-illegal transitions fixed
         to -1e4). Trained with the sequence negative log-likelihood, decoded with Viterbi.
"""
import torch
import torch.nn as nn

from .spans import bio_constraints

NEG = -1e4


class LinearHead(nn.Module):
    def __init__(self, d_model, n_tags, dropout=0.1, hidden=0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        if hidden:
            self.proj = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_tags))
        else:
            self.proj = nn.Linear(d_model, n_tags)

    def forward(self, h):
        return self.proj(self.drop(h))


class CRF(nn.Module):
    def __init__(self, tagset):
        super().__init__()
        K = tagset.n
        self.K = K
        self.trans = nn.Parameter(torch.zeros(K, K))           # from -> to
        self.start = nn.Parameter(torch.zeros(K))
        self.end = nn.Parameter(torch.zeros(K))
        s, t = bio_constraints(tagset)
        self.register_buffer("start_mask", torch.tensor(s).clamp(min=NEG))
        self.register_buffer("trans_mask", torch.tensor(t).clamp(min=NEG))

    def full_trans(self):
        return self.trans + self.trans_mask, self.start + self.start_mask, self.end

    def nll(self, emis, tags, mask):
        """emis [B,T,K] float32, tags [B,T] (pads may be anything), mask [B,T] bool, right-padded.
        Returns the mean NLL per sequence divided by its length (token-normalised).
        Training windows can start or end inside an entity (first gold tag I-X), so the "no I at the start"
        constraint is not applied here, only when decoding whole documents."""
        trans, _, end = self.full_trans()
        start = self.start
        B, T, K = emis.shape
        tags = tags.clamp(min=0)
        # score of the gold path
        score = start[tags[:, 0]] + emis[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, T):
            m = mask[:, t]
            s_t = trans[tags[:, t - 1], tags[:, t]] + emis[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            score = score + s_t * m
        lengths = mask.sum(1)
        last = tags.gather(1, (lengths - 1).clamp(min=0)[:, None]).squeeze(1)
        score = score + end[last]
        # log partition (forward algorithm)
        alpha = start[None] + emis[:, 0]                        # [B,K]
        for t in range(1, T):
            nxt = torch.logsumexp(alpha[:, :, None] + trans[None], dim=1) + emis[:, t]
            alpha = torch.where(mask[:, t, None], nxt, alpha)
        logZ = torch.logsumexp(alpha + end[None], dim=1)
        return ((logZ - score) / lengths.clamp(min=1)).mean()
