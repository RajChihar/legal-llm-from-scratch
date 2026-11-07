import numpy as np

GRID = np.round(np.arange(0.02, 0.951, 0.01), 4)

def prf_counts(G, P):
    G = np.asarray(G, dtype=bool); P = np.asarray(P, dtype=bool)
    tp = (G & P).sum(0).astype(np.int64)
    fp = (~G & P).sum(0).astype(np.int64)
    fn = (G & ~P).sum(0).astype(np.int64)
    return tp, fp, fn

def _f1(tp, fp, fn):
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)

def multilabel_report(G, P, space=None):
    G = np.asarray(G, dtype=bool); P = np.asarray(P, dtype=bool)
    assert G.shape == P.shape, (G.shape, P.shape)
    tp, fp, fn = prf_counts(G, P)
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    f1 = _f1(tp, fp, fn)
    present = (tp + fn) > 0
    TP, FP, FN = int(tp.sum()), int(fp.sum()), int(fn.sum())
    out = {"macro_f1": float(f1.mean()) if len(f1) else 0.0,
           "macro_p": float(prec.mean()) if len(f1) else 0.0,
           "macro_r": float(rec.mean()) if len(f1) else 0.0,
           "macro_f1_present": float(f1[present].mean()) if present.any() else 0.0,
           "micro_p": TP / (TP + FP) if TP + FP else 0.0,
           "micro_r": TP / (TP + FN) if TP + FN else 0.0,
           "micro_f1": 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0.0,
           "subset_acc": float((G == P).all(1).mean()) if len(G) else 0.0,
           "n_docs": int(G.shape[0]), "n_labels": int(G.shape[1]), "labels_present": int(present.sum()),
           "gold_per_doc": float(G.sum(1).mean()) if len(G) else 0.0,
           "pred_per_doc": float(P.sum(1).mean()) if len(G) else 0.0,
           "empty_pred_docs": int((P.sum(1) == 0).sum())}
    ids = list(range(G.shape[1])) if space is None else list(space)
    out["per_label"] = [{"id": int(i), "p": float(a), "r": float(b), "f1": float(c), "gold": int(t + n), "pred": int(t + f)}
                        for i, a, b, c, t, f, n in zip(ids, prec, rec, f1, tp, fp, fn)]
    return out

def apply_thresholds(prob, thr, min_labels=1):
    P = prob >= np.asarray(thr, dtype=np.float64)
    if min_labels:
        empty = ~P.any(1)
        if empty.any():
            P[np.where(empty)[0], prob[empty].argmax(1)] = True
    return P

def best_global_threshold(G, prob, min_labels=1, grid=GRID):
    best = (-1.0, 0.5)
    for t in grid:
        f = multilabel_report(G, apply_thresholds(prob, t, min_labels))["macro_f1"]
        if f > best[0] + 1e-12 or (abs(f - best[0]) <= 1e-12 and abs(t - 0.5) < abs(best[1] - 0.5)):
            best = (f, float(t))
    return best[1], best[0]

def best_label_thresholds(G, prob, fallback, min_support=3, grid=GRID):
    G = np.asarray(G, dtype=bool)
    L = G.shape[1]
    F = np.zeros((len(grid), L))
    for gi, t in enumerate(grid):
        tp, fp, fn = prf_counts(G, prob >= t)
        F[gi] = _f1(tp, fp, fn)
    best = F.max(0)
    dist = np.abs(grid[:, None] - fallback)
    cand = np.where(F >= best[None] - 1e-12, -dist, -np.inf)
    thr = grid[cand.argmax(0)].astype(np.float64)
    thr[G.sum(0) < min_support] = fallback
    return thr

def fit_thresholds(G, prob, strategy, min_labels=1, min_support=3):
    if strategy == "fixed":
        return 0.5
    t, _ = best_global_threshold(G, prob, min_labels)
    if strategy == "global":
        return t
    if strategy == "per_label":
        return best_label_thresholds(G, prob, t, min_support)
    raise ValueError(strategy)

def crossfit_strategies(G, prob, strategies=("fixed", "global", "per_label"), min_labels=1, min_support=3, seed=0):
    n = len(G)
    idx = np.random.default_rng(seed).permutation(n)
    halves = (idx[: n // 2], idx[n // 2:])
    out = {}
    for s in strategies:
        fs = []
        for a, b in (halves, halves[::-1]):
            thr = fit_thresholds(G[a], prob[a], s, min_labels, min_support)
            fs.append(multilabel_report(G[b], apply_thresholds(prob[b], thr, min_labels))["macro_f1"])
        out[s] = float(np.mean(fs))
    return out

def format_report(rep, title=""):
    lines = [title] if title else []
    lines.append(f"macro-F1 {rep['macro_f1']:.4f} (P {rep['macro_p']:.4f} R {rep['macro_r']:.4f}) | macro-F1 over the "
                 f"{rep['labels_present']} gold-present labels {rep['macro_f1_present']:.4f} | micro-F1 {rep['micro_f1']:.4f} "
                 f"(P {rep['micro_p']:.4f} R {rep['micro_r']:.4f}) | subset acc {rep['subset_acc']:.4f} | "
                 f"statutes/doc gold {rep['gold_per_doc']:.2f} pred {rep['pred_per_doc']:.2f} | {rep['n_docs']} docs x "
                 f"{rep['n_labels']} labels")
    return "\n".join(lines)

def frequency_buckets(per_label, train_freq, edges=(0, 20, 100, 500, 10**9)):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in per_label if lo <= train_freq.get(r["id"], 0) < hi]
        if sel:
            rows.append({"train_docs": f"[{lo}, {hi if hi < 10**9 else 'inf'})", "labels": len(sel),
                         "macro_f1": round(float(np.mean([r["f1"] for r in sel])), 4),
                         "macro_p": round(float(np.mean([r["p"] for r in sel])), 4),
                         "macro_r": round(float(np.mean([r["r"] for r in sel])), 4),
                         "dev_gold": int(sum(r["gold"] for r in sel))})
    return rows
