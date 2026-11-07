import bisect
from collections import Counter

import numpy as np

class TagSet:
    def __init__(self, labels):
        self.labels = list(labels)
        self.tags = ["O"] + [f"{p}-{l}" for l in self.labels for p in ("B", "I")]
        self.n = len(self.tags)

    def B(self, i): return 1 + 2 * i
    def I(self, i): return 2 + 2 * i
    def label_of(self, t): return None if t == 0 else self.labels[(t - 1) // 2]
    def is_B(self, t): return t > 0 and (t - 1) % 2 == 0
    def is_I(self, t): return t > 0 and (t - 1) % 2 == 1

def token_views(text, enc):
    cs, ce, ws = [], [], []
    for s, e in zip(enc.starts, enc.ends):
        t = text[s:e]
        st = t.strip()
        if not st:
            cs.append(s); ce.append(e); ws.append(True)
        else:
            cs.append(s + (len(t) - len(t.lstrip()))); ce.append(e - (len(t) - len(t.rstrip()))); ws.append(False)
    return np.array(cs), np.array(ce), np.array(ws, dtype=bool)

def spans_to_tags(text, enc, gold, tagset):
    cs, ce, ws = token_views(text, enc)
    N = len(cs)
    tags = np.zeros(N, dtype=np.int64)
    taken = np.zeros(N, dtype=bool)
    lab_idx = {l: i for i, l in enumerate(tagset.labels)}
    skipped = 0
    starts = enc.starts
    for s, e, l in sorted(gold, key=lambda x: -(x[1] - x[0])):
        if l not in lab_idx:
            continue
        lo = max(0, bisect.bisect_right(starts, s) - 1)
        hi = bisect.bisect_left(starts, e)
        idx = [i for i in range(lo, min(hi + 1, N)) if not ws[i] and cs[i] < e and ce[i] > s]
        if not idx:
            skipped += 1
            continue
        a, b = idx[0], idx[-1]
        if taken[a:b + 1].any():
            skipped += 1
            continue
        k = lab_idx[l]
        tags[a:b + 1] = tagset.I(k)
        tags[a] = tagset.B(k)
        taken[a:b + 1] = True
    return tags, skipped

def tags_to_spans(cs, ce, ws, tags, tagset):
    out = []
    cur = None
    for i in range(len(tags)):
        if ws[i]:
            continue
        t = int(tags[i])
        if t == 0:
            if cur: out.append(tuple(cur)); cur = None
            continue
        lab = tagset.label_of(t)
        if tagset.is_B(t) or cur is None or cur[2] != lab:
            if cur: out.append(tuple(cur))
            cur = [int(cs[i]), int(ce[i]), lab]
        else:
            cur[1] = int(ce[i])
    if cur:
        out.append(tuple(cur))
    return out

def bio_constraints(tagset):
    K = tagset.n
    trans = np.zeros((K, K), dtype=np.float32)
    start = np.zeros(K, dtype=np.float32)
    for j in range(K):
        if tagset.is_I(j):
            start[j] = -np.inf
            for i in range(K):
                if not (i > 0 and tagset.label_of(i) == tagset.label_of(j)):
                    trans[i, j] = -np.inf
    return start, trans

def viterbi(emis, trans, start, end=None):
    N, K = emis.shape
    if N == 0:
        return np.zeros(0, dtype=np.int64)
    score = start + emis[0]
    back = np.zeros((N, K), dtype=np.int64)
    for t in range(1, N):
        cand = score[:, None] + trans
        back[t] = cand.argmax(0)
        score = cand.max(0) + emis[t]
    if end is not None:
        score = score + end
    path = np.zeros(N, dtype=np.int64)
    path[-1] = int(score.argmax())
    for t in range(N - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path

def decode_doc(text, enc, scores, tagset, strategy="bio_viterbi", trans=None, start=None, end=None):
    cs, ce, ws = token_views(text, enc)
    if strategy == "greedy":
        tags = scores.argmax(-1)
    else:
        s0, tr0 = bio_constraints(tagset)
        tr = tr0 if trans is None else tr0 + trans
        st = s0 if start is None else s0 + start
        tags = viterbi(scores.astype(np.float32), tr, st, end)
    return tags_to_spans(cs, ce, ws, tags, tagset)

def learn_trim_rules(docs, max_rate=0.002):
    first, last, n = Counter(), Counter(), 0
    seen_chars = set()
    for d in docs.values():
        t = d["text"]
        seen_chars.update(c for c in set(t) if not c.isalnum() and not c.isspace())
        for s, e, _ in d["gold"]:
            first[t[s]] += 1; last[t[e - 1]] += 1; n += 1
    n = max(n, 1)
    lead = {c for c in seen_chars if first[c] / n <= max_rate}
    trail = {c for c in seen_chars if last[c] / n <= max_rate}
    return {"lead": "".join(sorted(lead)), "trail": "".join(sorted(trail))}

def trim_spans(text, spans, rules):
    if not rules:
        return spans
    lead, trail = set(rules["lead"]), set(rules["trail"])
    out = []
    for s, e, l in spans:
        while s < e and (text[s] in lead or text[s].isspace()):
            s += 1
        while e > s and (text[e - 1] in trail or text[e - 1].isspace()):
            e -= 1
        if e > s:
            out.append((s, e, l))
    return out

def reachability(docs, tok, tagset):
    from .metrics import strict_scores
    pred, bad, skipped = {}, [], 0
    for did, d in docs.items():
        enc = tok.encode_with_offsets(d["text"])
        tags, sk = spans_to_tags(d["text"], enc, d["gold"], tagset)
        skipped += sk
        cs, ce, ws = token_views(d["text"], enc)
        pred[did] = tags_to_spans(cs, ce, ws, tags, tagset)
        ps = set(pred[did])
        for g in d["gold"]:
            if tuple(g) not in ps:
                s, e, l = g
                bad.append((did, s, e, l, d["text"][max(0, s - 15):s] + "[" + d["text"][s:e] + "]" + d["text"][e:e + 15]))
    gold = {k: d["gold"] for k, d in docs.items()}
    return strict_scores(gold, pred, tagset.labels), bad, skipped
