import numpy as np

def _spans_to_tokens(tokens, spans, gap_ids):
    out, prev_end = [], None
    for s, e in spans:
        if prev_end is not None and s != prev_end and gap_ids is not None and len(gap_ids):
            out.append(np.asarray(gap_ids, dtype=tokens.dtype))
        out.append(tokens[s:e])
        prev_end = e
    return np.concatenate(out) if out else tokens[:0]

def select_spans(n, budget, strategy="headtail", sent_end=None, sent_score=None, head_frac=0.5, k=4, lead=256,
                 gap_len=0):
    if n <= budget:
        return [(0, n)]
    if strategy == "head":
        return [(0, budget)]
    if strategy == "tail":
        return [(n - budget, n)]
    if strategy == "headtail":
        b = budget - gap_len
        h = int(round(b * head_frac))
        return [(0, h), (n - (b - h), n)] if 0 < h < b else ([(0, b)] if h >= b else [(n - b, n)])
    if strategy == "uniform":
        k = max(1, min(k, budget // 16))
        w = (budget - (k - 1) * gap_len) // k
        starts = np.linspace(0, n - w, k).round().astype(int)
        spans = []
        for s in starts:
            s = int(s)
            if spans and s < spans[-1][1]:
                s = spans[-1][1]
            if s + w <= n:
                spans.append((s, s + w))
        return _merge(spans)
    if strategy == "salient":
        assert sent_end is not None and sent_score is not None, "salient selection needs sentence boundaries"
        ends = np.asarray(sent_end, dtype=np.int64)
        starts = np.concatenate([[0], ends[:-1]])
        m = len(ends)
        chosen = np.zeros(m, dtype=bool)
        chosen[ends <= min(lead, budget)] = True
        used = int(ends[chosen].max()) if chosen.any() else 0
        for i in np.argsort(-np.asarray(sent_score), kind="stable"):
            if chosen[i]:
                continue
            ln = int(ends[i] - starts[i])
            nb = int(i > 0 and chosen[i - 1]) + int(i + 1 < m and chosen[i + 1])
            d_gap = (1 if chosen.any() else 0) - nb
            if used + ln + gap_len * d_gap <= budget:
                chosen[i] = True
                used += ln + gap_len * d_gap
        if not chosen.any():
            return [(0, budget)]
        spans = _merge([(int(starts[i]), int(ends[i])) for i in np.flatnonzero(chosen)])
        assert _cost(spans, gap_len) == used, (_cost(spans, gap_len), used)
        return spans
    raise ValueError(f"unknown context strategy {strategy}")

def _merge(spans):
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out

def _cost(spans, gap_len):
    return sum(e - s for s, e in spans) + gap_len * sum(1 for a, b in zip(spans, spans[1:]) if b[0] != a[1])

def select_source(tokens, budget, strategy="headtail", sent_end=None, sent_score=None, head_frac=0.5, k=4, lead=256,
                  gap_ids=None):
    gl = len(gap_ids) if gap_ids is not None else 0
    spans = select_spans(len(tokens), budget, strategy, sent_end, sent_score, head_frac, k, lead, gl)
    out = _spans_to_tokens(np.asarray(tokens), spans, gap_ids)
    assert len(out) <= budget, (len(out), budget, strategy, spans)
    return out

def source_for(split, i, ccfg, budget, gap_ids):
    return select_source(split["tokens"][i], budget, ccfg["strategy"], split["sent_end"][i], split["sent_score"][i],
                         ccfg.get("head_frac", 0.5), ccfg.get("k", 4), ccfg.get("lead", 256), gap_ids)

def fid_chunks(src, chunk_len):
    return [src[i:i + chunk_len] for i in range(0, max(1, len(src)), chunk_len)] or [src]
