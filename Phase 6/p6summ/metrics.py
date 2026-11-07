import re
from collections import Counter

import numpy as np

_TOK_RE = re.compile(r"[a-z0-9]+")

def _tokens(text):
    return _TOK_RE.findall(text.lower())

def _ngrams(t, n):
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))

def _prf(overlap, n_pred, n_ref):
    p = overlap / n_pred if n_pred else 0.0
    r = overlap / n_ref if n_ref else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

def _lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]

def rouge_fallback(pred, ref, lcs=True):
    p, r = _tokens(pred), _tokens(ref)
    out = {}
    for n in (1, 2):
        a, b = _ngrams(p, n), _ngrams(r, n)
        out[f"rouge{n}"] = _prf(sum((a & b).values()), max(0, len(p) - n + 1), max(0, len(r) - n + 1))
    if lcs:
        out["rougeL"] = _prf(_lcs(p, r), len(p), len(r))
    return out

_SCORER = None

def rouge_scores(preds, refs, use_package=True):
    global _SCORER
    scorer = None
    if use_package:
        try:
            if _SCORER is None:
                from rouge_score import rouge_scorer
                _SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            scorer = _SCORER
        except ImportError:
            scorer = None
    per = {k: [] for k in ("rouge1", "rouge2", "rougeL")}
    for p, r in zip(preds, refs):
        if scorer is not None:
            s = scorer.score(r, p)
            for k in per:
                per[k].append((s[k].precision, s[k].recall, s[k].fmeasure))
        else:
            s = rouge_fallback(p, r)
            for k in per:
                per[k].append(s[k])
    out = {"rouge_impl": "rouge_score" if scorer is not None else "fallback"}
    for k, v in per.items():
        a = np.array(v) if v else np.zeros((0, 3))
        out[f"{k}_p"], out[f"{k}_r"], out[f"{k}_f"] = (float(a[:, i].mean()) if len(a) else 0.0 for i in range(3))
    out["per_doc"] = {f"{k}_f": [float(x[2]) for x in v] for k, v in per.items()}
    return out

_BERT = {}

def bertscore(preds, refs, model_type=None, lang="en", batch_size=32, device=None):
    try:
        from bert_score import BERTScorer
    except ImportError:
        return None
    key = (model_type, lang, str(device))
    if key not in _BERT:
        _BERT[key] = BERTScorer(model_type=model_type, lang=lang if model_type is None else None, device=device)
    P, R, F = _BERT[key].score(list(preds), list(refs), batch_size=batch_size)
    return {"bertscore_p": float(P.mean()), "bertscore_r": float(R.mean()), "bertscore_f": float(F.mean()),
            "per_doc_f": [float(x) for x in F]}

def free_bertscore():
    _BERT.clear()

_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")

def redundancy(text):
    t = _tokens(text)
    g = list(zip(t, t[1:], t[2:]))
    rep3 = 1 - len(set(g)) / len(g) if g else 0.0
    sents = [s.strip().lower() for s in _SENT_SPLIT.split(text) if len(s.split()) >= 4]
    rep_s = 1 - len(set(sents)) / len(sents) if sents else 0.0
    return rep3, rep_s

def novelty(summary, source, n):
    s = list(_ngrams(_tokens(summary), n))
    if not s:
        return 0.0
    src = set(_ngrams(_tokens(source), n))
    return sum(x not in src for x in s) / len(s)

_FACT_PATTERNS = [
    r"\b(?:section|sec|article|art|rule|order|clause|schedule|regulation)s?\.?\s*\(?\d+[a-z]?\)?(?:\s*\(\d+\))?",
    r"\b[a-z][a-z ,.'&()-]{2,60}?\bact\b,?\s*(?:of\s*)?\d{4}",
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|"
    r"december)[,]?\s+\d{4}\b",
    r"\b(?:rs\.?|rupees|inr)\s*[\d,]+(?:\.\d+)?",
    r"\b(?:no\.?|nos\.?)\s*\d+(?:\s*(?:of|/)\s*\d{2,4})?",
    r"\b(?:18|19|20)\d{2}\b",
]
_FACT_RE = re.compile("|".join(f"(?:{p})" for p in _FACT_PATTERNS))

def legal_facts(text):
    out = set()
    for m in _FACT_RE.finditer(text.lower()):
        f = re.sub(r"[^a-z0-9]+", " ", m.group(0)).strip()
        f = re.sub(r"^(sec|sections?|s)\b", "section", f)
        f = re.sub(r"^(art|articles?)\b", "article", f)
        if f:
            out.add(f)
    return out

def fact_scores(pred, source, ref):
    gp, src, rf = legal_facts(pred), legal_facts(source), legal_facts(ref)
    prec = len(gp & src) / len(gp) if gp else float("nan")
    rec = len(gp & rf) / len(rf) if rf else float("nan")
    return prec, rec, len(gp)

def analysis(preds, refs, sources):
    rows = []
    for p, r, s in zip(preds, refs, sources):
        rep3, reps = redundancy(p)
        fp, fr, nf = fact_scores(p, s, r)
        rows.append({"words": len(p.split()), "ref_words": len(r.split()), "rep_3gram": rep3, "rep_sent": reps,
                     "novel_1gram": novelty(p, s, 1), "novel_2gram": novelty(p, s, 2), "novel_3gram": novelty(p, s, 3),
                     "fact_precision": fp, "fact_recall": fr, "facts_per_summary": nf})
    out = {}
    for k in rows[0] if rows else []:
        v = np.array([x[k] for x in rows], dtype=float)
        out[k] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    return out

def length_bucket_report(per_doc, ref_lens, edges=(0, 300, 600, 1000, 10 ** 9)):
    per_doc, ref_lens = np.asarray(per_doc), np.asarray(ref_lens)
    out = []
    for a, b in zip(edges, edges[1:]):
        m = (ref_lens >= a) & (ref_lens < b)
        if m.any():
            out.append({"ref_words": f"{a}-{b if b < 10 ** 9 else 'inf'}", "docs": int(m.sum()),
                        "score": round(float(per_doc[m].mean()), 4)})
    return out
