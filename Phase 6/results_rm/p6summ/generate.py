"""Decoding (config section `gen`): greedy, beam search or sampling, all with a KV cache.

Logit processing at every step (in this order):
    special tokens <pad> <bos> <sep> <mask> are never generated; <eot> (end of summary) is blocked until
    `min_new` tokens exist (length control)
    repetition penalty  logit / p (positive) or * p (negative) for tokens among the last `rep_window` generated
    no-repeat n-gram    a token that would complete an n-gram already present in the hypothesis gets -inf
                        (vectorised: compare the last n-1 tokens with every earlier (n-1)-gram)
then greedy = argmax, sample = temperature / top-k / top-p, beam = `beams` hypotheses per document scored by
sum of log-probabilities / length^length_penalty.
Generation stops at <eot> or after `max_new` tokens.
"""
import time

import numpy as np
import torch

from .model import map_state

GEN_DEFAULTS = {"method": "greedy", "beams": 4, "max_new": 640, "min_new": 64, "no_repeat_ngram": 4,
                "rep_penalty": 1.0, "rep_window": 512, "length_penalty": 1.0, "temperature": 0.7, "top_k": 50,
                "top_p": 0.9, "batch_size": 16, "seed": 0}


def process_logits(logits, hist, step, g, sp):
    """logits [R,V] float32 (modified in place), hist [R,t] generated tokens so far."""
    logits[:, sp["banned"]] = -float("inf")
    if step < g["min_new"]:
        logits[:, sp["eot"]] = -float("inf")
    t = hist.size(1)
    if t and g.get("rep_penalty", 1.0) != 1.0:
        ctx = hist[:, -g["rep_window"]:]
        sc = torch.gather(logits, 1, ctx)
        logits.scatter_(1, ctx, torch.where(sc > 0, sc / g["rep_penalty"], sc * g["rep_penalty"]))
    n = g.get("no_repeat_ngram", 0)
    if n and t >= n:
        win = hist.unfold(1, n - 1, 1)                           # [R, t-n+2, n-1] every (n-1)-gram
        match = (win[:, :-1] == hist[:, None, t - n + 1:]).all(-1)     # earlier (n-1)-grams == current suffix
        nxt = hist[:, n - 1:]                                    # the token that followed each of them
        logits.scatter_(1, torch.where(match, nxt, sp["banned"][0]), -float("inf"))
    return logits


def _sample(logits, g, gen):
    logits = logits / max(g["temperature"], 1e-5)
    if g.get("top_k"):
        v, _ = torch.topk(logits, min(g["top_k"], logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")
    if g.get("top_p") and g["top_p"] < 1.0:
        sl, si = torch.sort(logits, descending=True)
        probs = torch.softmax(sl, -1)
        sl[probs.cumsum(-1) - probs > g["top_p"]] = -float("inf")        # always keeps the top token
        logits = torch.full_like(logits, -float("inf")).scatter(1, si, sl)
    return torch.multinomial(torch.softmax(logits, -1), 1, generator=gen)[:, 0]


@torch.no_grad()
def _greedy_or_sample(model, state, g, sp, gen):
    logits = state.pop("logits")
    R = logits.size(0)
    dev = logits.device
    hist = torch.zeros(R, 0, dtype=torch.long, device=dev)
    done = torch.zeros(R, dtype=torch.bool, device=dev)
    for step in range(g["max_new"]):
        lg = process_logits(logits, hist, step, g, sp)
        tok = lg.argmax(-1) if g["method"] == "greedy" else _sample(lg, g, gen)
        tok = torch.where(done, torch.full_like(tok, sp["eot"]), tok)
        hist = torch.cat([hist, tok[:, None]], 1)
        done |= tok == sp["eot"]
        if bool(done.all()):
            break
        logits = model.step(state, tok)
    out = []
    for row in hist.tolist():
        out.append(row[:row.index(sp["eot"])] if sp["eot"] in row else row)
    return out


@torch.no_grad()
def _beam(model, state, g, sp):
    nb, lp = g["beams"], g.get("length_penalty", 1.0)
    logits = state.pop("logits")
    B, V = logits.shape
    dev = logits.device
    rep = torch.arange(B, device=dev).repeat_interleave(nb)
    state = map_state(state, lambda t: t.index_select(0, rep))
    logits = logits.index_select(0, rep)
    scores = torch.full((B, nb), -float("inf"), device=dev)
    scores[:, 0] = 0.0
    hist = torch.zeros(B * nb, 0, dtype=torch.long, device=dev)
    finished = [[] for _ in range(B)]                            # (normalised score, tokens)
    done = [False] * B
    for step in range(g["max_new"]):
        lp_ = torch.log_softmax(process_logits(logits, hist, step, g, sp), -1)
        cand = (scores.view(-1, 1) + lp_).view(B, nb * V)
        top_s, top_i = cand.topk(2 * nb, dim=1)
        top_s, top_i = top_s.tolist(), top_i.tolist()
        new_scores = torch.full((B, nb), -float("inf"), device=dev)
        src = torch.zeros(B * nb, dtype=torch.long, device=dev)
        toks = torch.full((B * nb,), sp["eot"], dtype=torch.long, device=dev)
        src_l, tok_l, sc_l = [], [], []
        for b in range(B):
            rows, toks_b, scs = [], [], []
            if not done[b]:
                for s, i in zip(top_s[b], top_i[b]):
                    if s == -float("inf"):
                        break
                    beam, tok = divmod(i, V)
                    r = b * nb + beam
                    if tok == sp["eot"]:
                        finished[b].append((s / (step + 1) ** lp, hist[r].tolist()))
                    else:
                        rows.append(r); toks_b.append(tok); scs.append(s)
                    if len(rows) == nb:
                        break
                finished[b] = sorted(finished[b], key=lambda x: -x[0])[:nb]
                if len(finished[b]) >= nb and (not scs or max(scs) / (step + 1) ** lp <= finished[b][-1][0]):
                    done[b] = True
            while len(rows) < nb:                                # padding beams (finished document / too few)
                rows.append(b * nb); toks_b.append(sp["eot"]); scs.append(-float("inf"))
            src_l += rows; tok_l += toks_b; sc_l += scs
        src = torch.tensor(src_l, device=dev)
        toks = torch.tensor(tok_l, device=dev)
        scores = torch.tensor(sc_l, device=dev).view(B, nb)
        hist = torch.cat([hist.index_select(0, src), toks[:, None]], 1)
        if all(done):
            break
        state = map_state(state, lambda t: t.index_select(0, src))
        logits = model.step(state, toks)
    out = []
    for b in range(B):
        if not done[b]:                                          # max length reached: running beams compete too
            for k in range(nb):
                s = float(scores[b, k])
                if s > -float("inf"):
                    row = hist[b * nb + k].tolist()
                    finished[b].append((s / max(1, len(row)) ** lp, row))
        best = max(finished[b], key=lambda x: x[0])[1] if finished[b] else []
        out.append([t for t in best if t != sp["eot"]])
    return out


def special_ids(tok):
    return {"eot": tok.special_to_id["<|endoftext|>"],
            "banned": [tok.pad_id, tok.bos_id, tok.special_to_id["<|sep|>"], tok.mask_id]}


@torch.no_grad()
def generate(model, inputs, gcfg, tok, device="cuda", log=None, amp_dtype=None):
    """inputs: list of model inputs (see Summarizer.prefill), one per document -> list of generated token lists.
    Documents are sorted by input length so a batch holds similar lengths (less cache padding)."""
    g = {**GEN_DEFAULTS, **(gcfg or {})}
    sp = special_ids(tok)
    sp["banned"] = torch.tensor(sp["banned"], device=device)
    model.eval()
    size = (lambda x: sum(len(c) for c in x)) if model.arch == "B" else len
    order = np.argsort([-size(x) for x in inputs], kind="stable")
    bs = max(1, g["batch_size"] // (g["beams"] if g["method"] == "beam" else 1))
    out = [None] * len(inputs)
    gen = torch.Generator(device=device).manual_seed(g.get("seed", 0)) if g["method"] == "sample" else None
    t0, n_tok = time.time(), 0
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None and device == "cuda"):
            state = model.prefill([inputs[i] for i in idx])
            res = _beam(model, state, g, sp) if g["method"] == "beam" else _greedy_or_sample(model, state, g, sp, gen)
        for i, r in zip(idx, res):
            out[i] = r
            n_tok += len(r)
        if log and (k // bs) % 10 == 0:
            log(f"[gen] {min(k + bs, len(order))}/{len(order)} documents | {time.time() - t0:.0f}s")
    dt = time.time() - t0
    return out, {"seconds": round(dt, 1), "docs_per_sec": round(len(inputs) / max(dt, 1e-6), 3),
                 "tokens_per_sec": round(n_tok / max(dt, 1e-6), 1), "mean_new_tokens": round(n_tok / max(1, len(inputs)), 1)}
