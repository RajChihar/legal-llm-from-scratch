"""Retrieval metrics (Codabench: micro-F1@7 primary; micro-precision@7 and micro-recall@7 secondary).

Every query returns exactly k = 7 candidates; the gold set is complete (neither truncated nor padded):
    micro-P@k  = sum_q |top_k(q) & gold(q)| / (k * n_queries)
    micro-R@k  = sum_q |top_k(q) & gold(q)| / sum_q |gold(q)|
    micro-F1@k = 2 P R / (P + R)
Analysis-only measures: recall@K for larger K (is the gold in reach of a re-ranker?), MRR, mean average precision,
per-query F1@7.
"""
import numpy as np


def rank(scores, q_ids, c_ids, k=None):
    """scores [n_q, n_c] -> per query the candidate indices sorted by decreasing score (the query's own id is never
    returned: a case is not its own prior case). k: keep only the top k."""
    s = np.array(scores, dtype=np.float64, copy=True)
    pos = {c: j for j, c in enumerate(c_ids)}
    for i, q in enumerate(q_ids):
        j = pos.get(q)
        if j is not None:
            s[i, j] = -np.inf
    if k is None or k >= s.shape[1]:
        order = np.argsort(-s, axis=1, kind="stable")
    else:
        part = np.argpartition(-s, k - 1, axis=1)[:, :k]
        order = np.take_along_axis(part, np.argsort(-np.take_along_axis(s, part, 1), axis=1, kind="stable"), 1)
    return order


def micro_prf(pred, gold, k=7):
    """pred: {q: [cand keys] (ranked)}, gold: {q: [cand keys]} -> (P, R, F1, per-query F1)."""
    tp = n_pred = n_gold = 0
    per = {}
    for q, g in gold.items():
        top = list(pred.get(q, []))[:k]
        h = len(set(top) & set(g))
        tp += h; n_pred += k; n_gold += len(g)
        p_q = h / k
        r_q = h / len(g) if g else 0.0
        per[q] = 2 * p_q * r_q / (p_q + r_q) if p_q + r_q else 0.0
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), per


def report(scores, q_ids, c_ids, gold, k=7, ks=(7, 20, 50, 100, 200)):
    """Full report from a score matrix: micro P/R/F1@k, recall@K, MRR, MAP@100, per-query F1 and gold ranks."""
    order = rank(scores, q_ids, c_ids)
    pred = {q: [c_ids[j] for j in order[i, :max(ks + (k,))]] for i, q in enumerate(q_ids)}
    p, r, f, per = micro_prf(pred, gold, k)
    out = {"micro_p@7": p, "micro_r@7": r, "micro_f1@7": f, "n_queries": len(gold)}
    n_gold = sum(len(g) for g in gold.values())
    for K in ks:
        out[f"recall@{K}"] = sum(len(set(pred[q][:K]) & set(g)) for q, g in gold.items()) / max(1, n_gold)
    pos = {c: j for j, c in enumerate(c_ids)}
    mrr, ap, gold_ranks = [], [], {}
    rank_of = np.empty_like(order)
    for i in range(order.shape[0]):
        rank_of[i, order[i]] = np.arange(order.shape[1])
    for i, q in enumerate(q_ids):
        g = gold.get(q)
        if not g:
            continue
        rk = sorted(int(rank_of[i, pos[c]]) + 1 for c in g)
        gold_ranks[q] = rk
        mrr.append(1.0 / rk[0])
        ap.append(sum((j + 1) / r_ for j, r_ in enumerate(rk) if r_ <= 100) / len(rk))
    out["mrr"] = float(np.mean(mrr)) if mrr else 0.0
    out["map@100"] = float(np.mean(ap)) if ap else 0.0
    return out, per, gold_ranks


def top_k_lists(scores, q_ids, c_ids, k=7):
    order = rank(scores, q_ids, c_ids, k)
    return [[c_ids[j] for j in row] for row in order]


def zscore(s):
    """Per-query standardisation of a score matrix (used to fuse scores of different scales)."""
    s = np.asarray(s, dtype=np.float64)
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)
