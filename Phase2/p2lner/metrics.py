from collections import Counter, defaultdict

def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f

def strict_scores(gold, pred, labels):
    tp, fp, fn = Counter(), Counter(), Counter()
    for did in gold:
        g = set(map(tuple, gold[did]))
        p = set(map(tuple, pred.get(did, [])))
        for x in g & p: tp[x[2]] += 1
        for x in p - g: fp[x[2]] += 1
        for x in g - p: fn[x[2]] += 1
    per = {}
    for l in labels:
        if tp[l] + fp[l] + fn[l] == 0:
            continue
        p, r, f = _prf(tp[l], fp[l], fn[l])
        per[l] = {"p": p, "r": r, "f1": f, "tp": tp[l], "fp": fp[l], "fn": fn[l], "support": tp[l] + fn[l]}
    n = max(len(per), 1)
    T, F, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
    mp, mr, mf = _prf(T, F, N)
    return {"macro_f1": sum(v["f1"] for v in per.values()) / n,
            "macro_p": sum(v["p"] for v in per.values()) / n,
            "macro_r": sum(v["r"] for v in per.values()) / n,
            "micro_f1": mf, "micro_p": mp, "micro_r": mr, "per_label": per}

def error_analysis(gold, pred, texts=None, n_examples=5):
    cats = Counter()
    confusion = Counter()
    boundary_kind = Counter()
    by_label = defaultdict(Counter)
    examples = defaultdict(list)
    for did, g in gold.items():
        g = set(map(tuple, g)); p = set(map(tuple, pred.get(did, [])))
        P = sorted(p)
        for x in sorted(g - p):
            s, e, l = x
            ov = [y for y in P if y[0] < e and y[1] > s]
            same_span = [y for y in ov if y[0] == s and y[1] == e]
            if same_span:
                c = "type"; confusion[(l, same_span[0][2])] += 1
            elif any(y[2] == l for y in ov):
                c = "boundary"
                y = next(y for y in ov if y[2] == l)
                boundary_kind[("start " + ("early" if y[0] < s else "late" if y[0] > s else "ok")) + " / " +
                              ("end " + ("early" if y[1] < e else "late" if y[1] > e else "ok"))] += 1
            elif ov:
                c = "both"
            else:
                c = "missed"
            cats[c] += 1; by_label[l][c] += 1
            if texts is not None and len(examples[c]) < n_examples:
                t = texts[did]
                pr = "; ".join(f"[{t[y[0]:y[1]]}]{y[2]}" for y in ov[:2])
                examples[c].append(f"{l}: [{t[s:e]}] | pred: {pr or '-'}")
        G = sorted(g)
        for y in sorted(p - g):
            if not any(x[0] < y[1] and x[1] > y[0] for x in G):
                cats["spurious"] += 1; by_label[y[2]]["spurious"] += 1
                if texts is not None and len(examples["spurious"]) < n_examples:
                    t = texts[did]
                    examples["spurious"].append(f"{y[2]}: [{t[y[0]:y[1]]}]")
    return {"categories": dict(cats), "type_confusion": {f"{a}->{b}": n for (a, b), n in confusion.most_common(15)},
            "boundary_kind": dict(boundary_kind), "by_label": {k: dict(v) for k, v in by_label.items()},
            "examples": dict(examples)}

def format_table(scores):
    lines = [f"{'label':10s} {'P':>6s} {'R':>6s} {'F1':>6s} {'support':>8s}"]
    for l, v in scores["per_label"].items():
        lines.append(f"{l:10s} {v['p']:6.3f} {v['r']:6.3f} {v['f1']:6.3f} {v['support']:8d}")
    lines.append(f"{'MACRO':10s} {scores['macro_p']:6.3f} {scores['macro_r']:6.3f} {scores['macro_f1']:6.3f}")
    lines.append(f"{'micro':10s} {scores['micro_p']:6.3f} {scores['micro_r']:6.3f} {scores['micro_f1']:6.3f}")
    return "\n".join(lines)
