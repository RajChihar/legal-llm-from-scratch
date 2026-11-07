import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .bm25 import bm25_scores
from .metrics import rank, report, zscore
from .tokenizer import BPETokenizer
from .train import (DEFAULTS, bm25_cached, load_data, load_run, load_scores, merge, resolve_ckpt, save_json,
                    save_scores, score_split, set_by_path, tok_path, write_submission)

def bm25_grid(data, split="dev", grid=None):
    grid = grid or [dict(k1=1.2, b=0.75, ngrams=(1,), query_tf="sat"), dict(k1=1.2, b=0.75, ngrams=(1,), query_tf="binary"),
                    dict(k1=1.2, b=0.75, ngrams=(1,), query_tf="count"), dict(k1=2.0, b=0.75, ngrams=(1,), query_tf="sat"),
                    dict(k1=1.2, b=0.4, ngrams=(1,), query_tf="sat"), dict(k1=1.2, b=0.75, ngrams=(1, 2), query_tf="sat")]
    d = data[split]
    rows = []
    for g in grid:
        rep, _, _ = report(bm25_scores(d, **g), d["q_ids"], d["c_ids"], d["gold"])
        rows.append({**{k: str(v) for k, v in g.items()}, **{k: round(rep[k], 4) for k in
                     ("micro_f1@7", "micro_p@7", "micro_r@7", "recall@20", "recall@100", "mrr")},
                     "params": {**g, "ngrams": list(g["ngrams"])}})
    return rows

_YEAR = re.compile(r"\b(1[89]\d{2}|20[0-4]\d)\b")

def case_year(text):
    ys = [int(y) for y in _YEAR.findall(text)]
    return max(ys) if ys else None

def future_mask(split, tol=0):
    qy = np.array([case_year(t) or 9999 for t in split["q_text"]], dtype=np.float64)
    cy = np.array([case_year(t) or 0 for t in split["c_text"]], dtype=np.float64)
    return cy[None, :] > qy[:, None] + tol

def year_analysis(data, split="dev"):
    d = data[split]
    fm = future_mask(d)
    pos = {c: j for j, c in enumerate(d["c_ids"])}
    g = [(i, pos[c]) for i, q in enumerate(d["q_ids"]) for c in d["gold"][q]]
    return {"share_gold_candidates_later_than_query": round(float(np.mean([fm[i, j] for i, j in g])), 4) if g else None,
            "share_pool_later_than_query": round(float(fm.mean()), 4),
            "queries_without_year": int(sum(case_year(t) is None for t in d["q_text"])),
            "candidates_without_year": int(sum(case_year(t) is None for t in d["c_text"]))}

def fuse(parts, weights, fmask=None, penalty=1000.0):
    s = sum(w * zscore(p) for p, w in zip(parts, weights))
    return s - penalty * fmask if fmask is not None else s

def fusion(run_dirs, data, cfg, out_dir, bm25_weights=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7), year_filter=(False, True),
           log=print):
    out_dir = Path(out_dir)
    dev, te = data["dev"], data["test"]
    dd = [load_scores(Path(r) / "dev_scores.npz", dev) for r in run_dirs]
    td = [load_scores(Path(r) / "test_scores.npz", te) for r in run_dirs]
    bd, bt = bm25_cached(data, "dev", cfg), bm25_cached(data, "test", cfg)
    fmd, fmt = future_mask(dev), future_mask(te)
    grid = []
    for w in bm25_weights:
        for yf in year_filter:
            parts, ws = dd + [bd], [(1 - w) / len(dd)] * len(dd) + [w]
            rep, _, _ = report(fuse(parts, ws, fmd if yf else None), dev["q_ids"], dev["c_ids"], dev["gold"])
            grid.append({"bm25_weight": w, "year_filter": yf, "micro_f1@7": round(rep["micro_f1@7"], 4),
                         "recall@100": round(rep["recall@100"], 4)})
    best = max(grid, key=lambda r: (r["micro_f1@7"], -r["bm25_weight"], not r["year_filter"]))
    w, yf = best["bm25_weight"], best["year_filter"]
    ws = [(1 - w) / len(td)] * len(td) + [w]
    s_dev = fuse(dd + [bd], ws, fmd if yf else None)
    s_te = fuse(td + [bt], ws, fmt if yf else None)
    rep, per, ranks = report(s_dev, dev["q_ids"], dev["c_ids"], dev["gold"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_scores(out_dir / "dev_scores.npz", dev, s_dev)
    save_scores(out_dir / "test_scores.npz", te, s_te)
    zp = write_submission(s_te, te, out_dir / "submission")
    s = {"run": out_dir.name, "members": [Path(r).name for r in run_dirs], "bm25_weight": w, "year_filter": yf,
         "grid": grid, "dev": rep, "submission": str(zp)}
    save_json(s, out_dir / "summary.json")
    save_json({"report": rep, "per_query_f1": per, "gold_ranks": ranks}, out_dir / "dev_report.json")
    log(f"[fusion] {s['members']} | BM25 weight {w} | year filter {yf} | dev micro-F1@7 {rep['micro_f1@7']:.4f}")
    return s

def view_ablation(run_dir, data, views, device="cuda", log=print):
    model, cfg = load_run(run_dir, device)
    pad = BPETokenizer.load(tok_path(cfg)).pad_id
    dev = data["dev"]
    rows = []
    for name, ov in views.items():
        s = score_split(model, dev, cfg, device, ov, pad)
        rep, _, _ = report(s, dev["q_ids"], dev["c_ids"], dev["gold"])
        rows.append({"view": name, **{k: round(rep[k], 4) for k in ("micro_f1@7", "recall@20", "recall@100", "mrr")}})
        log(f"[views] {name}: micro-F1@7 {rep['micro_f1@7']:.4f} | recall@100 {rep['recall@100']:.4f}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows

def error_analysis(run_dir, data, cfg, k=7):
    dev = data["dev"]
    s = load_scores(Path(run_dir) / "dev_scores.npz", dev)
    rep, per, ranks = report(s, dev["q_ids"], dev["c_ids"], dev["gold"])
    order = rank(s, dev["q_ids"], dev["c_ids"], k)
    b = bm25_cached(data, "dev", cfg)
    border = rank(b, dev["q_ids"], dev["c_ids"])
    brank = np.empty_like(border)
    for i in range(border.shape[0]):
        brank[i, border[i]] = np.arange(border.shape[1])
    pos = {c: j for j, c in enumerate(dev["c_ids"])}
    fp_r, miss_r, hit_r = [], [], []
    hubs = Counter()
    for i, q in enumerate(dev["q_ids"]):
        g = {pos[c] for c in dev["gold"][q]}
        top = set(order[i].tolist())
        hubs.update(dev["c_ids"][j] for j in top)
        fp_r += [brank[i, j] + 1 for j in top - g]
        miss_r += [brank[i, j] + 1 for j in g - top]
        hit_r += [brank[i, j] + 1 for j in g & top]
    ng = np.array([len(dev["gold"][q]) for q in dev["q_ids"]])
    ql = np.array([len(t) for t in dev["q_tokens"]])
    f = np.array([per[q] for q in dev["q_ids"]])

    def by(v, edges):
        rows = []
        for a, c in zip(edges, edges[1:]):
            m = (v >= a) & (v < c)
            if m.any():
                rows.append({"range": f"{a}-{c if c < 10 ** 9 else 'inf'}", "queries": int(m.sum()),
                             "mean_f1@7": round(float(f[m].mean()), 4)})
        return rows
    allr = np.array([r for v in ranks.values() for r in v])
    return {"f1_by_n_relevant": by(ng, (0, 1, 3, 6, 10, 20, 10 ** 9)),
            "f1_by_query_tokens": by(ql, (0, 2000, 5000, 10000, 20000, 10 ** 9)),
            "queries_with_no_hit": int((f == 0).sum()), "queries": len(f),
            "gold_rank_quantiles": {p: int(np.percentile(allr, p)) for p in (10, 25, 50, 75, 90)} if len(allr) else {},
            "share_gold_beyond_rank_100": round(float((allr > 100).mean()), 4) if len(allr) else None,
            "median_bm25_rank": {"false_positives": float(np.median(fp_r)) if fp_r else None,
                                 "missed_gold": float(np.median(miss_r)) if miss_r else None,
                                 "retrieved_gold": float(np.median(hit_r)) if hit_r else None},
            "hub_candidates(top7 of most queries)": hubs.most_common(8),
            "hub_share_of_top7": round(sum(c for _, c in hubs.most_common(10)) / max(1, k * len(f)), 4)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--eda", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(a.config)) if a.config else {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        set_by_path(cfg, k, v)
    from .train import run
    if a.eda:
        c = merge(DEFAULTS, cfg)
        c["backbone"]["ckpt"] = resolve_ckpt(c["backbone"]["ckpt"], [c["ckpt_root"]] if c.get("ckpt_root") else [])
        data = load_data(c)
        D.describe_raw(c["data_dir"])
        print(json.dumps(D.stats(data), indent=1))
        print(json.dumps(bm25_grid(data), indent=1))
        return
    run(cfg)

if __name__ == "__main__":
    main()
