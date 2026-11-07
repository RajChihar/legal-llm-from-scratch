from collections import Counter

import numpy as np

def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f

def flatten(gold, pred, ids=None):
    g, p = [], []
    for d in (ids if ids is not None else gold):
        assert len(gold[d]) == len(pred[d]), f"{d}: {len(gold[d])} gold vs {len(pred[d])} predicted sentences"
        g += list(gold[d]); p += list(pred[d])
    return g, p

def scores(gold, pred, labels, ids=None):
    g, p = flatten(gold, pred, ids)
    tp, fp, fn = Counter(), Counter(), Counter()
    for a, b in zip(g, p):
        if a == b:
            tp[a] += 1
        else:
            fp[b] += 1; fn[a] += 1
    support = Counter(g)
    per = {}
    for l in labels:
        pr, rc, f = _prf(tp[l], fp[l], fn[l])
        per[l] = {"p": pr, "r": rc, "f1": f, "support": support[l], "predicted": tp[l] + fp[l]}
    present = [l for l in labels if support[l] or tp[l] + fp[l]]
    mean = lambda key, ls: float(np.mean([per[l][key] for l in ls])) if ls else 0.0
    return {"macro_f1": mean("f1", present), "macro_p": mean("p", present), "macro_r": mean("r", present),
            "macro_f1_vocab": mean("f1", labels), "accuracy": sum(tp.values()) / max(1, len(g)),
            "n_sentences": len(g), "labels_in_average": len(present), "per_label": per}

def confusion(gold, pred, labels, ids=None):
    g, p = flatten(gold, pred, ids)
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for a, b in zip(g, p):
        m[idx[a], idx[b]] += 1
    return m

def top_confusions(cm, labels, k=12):
    pairs = [(int(cm[i, j]), labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels)) if i != j and cm[i, j]]
    return [f"{a} -> {b}: {n}" for n, a, b in sorted(pairs, reverse=True)[:k]]

def full_report(gold, pred, labels, domain_of):
    ids = list(gold)
    rep = {"all": scores(gold, pred, labels, ids)}
    for dom in sorted(set(domain_of.values())):
        di = [d for d in ids if domain_of[d] == dom]
        if di:
            rep[dom] = scores(gold, pred, labels, di)
    cm = confusion(gold, pred, labels, ids)
    rep["confusion"] = cm.tolist()
    rep["top_confusions"] = top_confusions(cm, labels)
    at_change = inside = 0
    for d in ids:
        gs, ps = gold[d], pred[d]
        for i, (a, b) in enumerate(zip(gs, ps)):
            if a != b:
                if (i > 0 and gs[i - 1] != a) or (i + 1 < len(gs) and gs[i + 1] != a):
                    at_change += 1
                else:
                    inside += 1
    rep["errors_at_label_change"], rep["errors_inside_run"] = at_change, inside
    try:
        from sklearn.metrics import f1_score
        g, p = flatten(gold, pred, ids)
        rep["sklearn_macro_f1"] = float(f1_score(g, p, average="macro", zero_division=0))
    except Exception:
        pass
    return rep

def format_table(sc):
    lines = [f"{'label':26s} {'P':>6s} {'R':>6s} {'F1':>6s} {'gold':>6s} {'pred':>6s}"]
    for l, v in sc["per_label"].items():
        lines.append(f"{l:26s} {v['p']:6.3f} {v['r']:6.3f} {v['f1']:6.3f} {v['support']:6d} {v['predicted']:6d}")
    lines.append(f"{'MACRO':26s} {sc['macro_p']:6.3f} {sc['macro_r']:6.3f} {sc['macro_f1']:6.3f}   "
                 f"(vocab-macro-F1 {sc['macro_f1_vocab']:.3f}, accuracy {sc['accuracy']:.3f}, {sc['n_sentences']} sentences)")
    return "\n".join(lines)
