import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .context import select_source
from .metrics import analysis, bertscore, rouge_fallback, rouge_scores
from .tokenizer import BPETokenizer
from .train import (DEFAULTS, load_data, load_run, merge, save_json, score_generations, set_by_path, summarise,
                    write_submission)

def coverage_analysis(data, tok, budgets=(512, 1024, 1536, 3072), strategies=("head", "tail", "headtail", "uniform",
                                                                                 "salient"), max_docs=200, lead=256):
    dev = data["dev"]
    gap = tok.encode(" ...")
    n = min(max_docs, len(dev["ids"]))
    rows = []
    full = [set(D.content_words(dev["documents"][i])) for i in range(n)]
    refs = [set(D.content_words(dev["summaries"][i])) for i in range(n)]
    cov_full = np.mean([len(r & f) / max(1, len(r)) for r, f in zip(refs, full)])
    for L in budgets:
        row = {"budget_tokens": L, "docs_longer_than_budget": round(float(np.mean([len(dev["tokens"][i]) > L
                                                                                    for i in range(n)])), 3)}
        for st in strategies:
            c = []
            for i in range(n):
                sel = select_source(dev["tokens"][i], L, st, dev["sent_end"][i], dev["sent_score"][i], 0.5, 4, lead, gap)
                w = set(D.content_words(tok.decode(sel)))
                c.append(len(refs[i] & w) / max(1, len(refs[i])))
            row[st] = round(float(np.mean(c)), 4)
        rows.append(row)
    return {"rows": rows, "whole_document": round(float(cov_full), 4), "docs": n}

def _pair_sim(a, b):
    s = rouge_fallback(a, b, lcs=False)
    return (s["rouge1"][2] + s["rouge2"][2]) / 2

def mbr_select(cands):
    if len(cands) == 1:
        return 0, [1.0]
    m = np.zeros((len(cands), len(cands)))
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            m[i, j] = m[j, i] = _pair_sim(cands[i], cands[j])
    mean = m.sum(1) / (len(cands) - 1)
    return int(mean.argmax()), mean.tolist()

def mbr_ensemble(run_dirs, out_dir, data):
    run_dirs = [Path(r) for r in run_dirs]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dev, te = data["dev"], data["test"]
    gens = {r.name: json.load(open(r / "dev_generations.json")) for r in run_dirs}
    common = set.intersection(*[{g["id"] for g in v} for v in gens.values()])
    pos = {i: k for k, i in enumerate(dev["ids"])}
    ids = [g["id"] for g in gens[run_dirs[0].name] if g["id"] in common]
    by = {n: {g["id"]: g["generated"] for g in v} for n, v in gens.items()}
    chosen, picks = [], Counter()
    for i in ids:
        names = list(by)
        k, _ = mbr_select([by[n][i] for n in names])
        chosen.append(by[names[k]][i]); picks[names[k]] += 1
    idx = [pos[i] for i in ids]
    scores, per_doc = score_generations(chosen, dev, idx)
    save_json([{"id": dev["ids"][j], "reference": dev["summaries"][j], "generated": t} | {k: v[q] for k, v in per_doc.items()}
               for q, (j, t) in enumerate(zip(idx, chosen))], out_dir / "dev_generations.json")
    tg = {r.name: {g["id"]: g["generated"] for g in json.load(open(r / "test_generations.json"))} for r in run_dirs}
    tsel, tpicks = [], Counter()
    for i in te["ids"]:
        names = list(tg)
        k, _ = mbr_select([tg[n][i] for n in names])
        tsel.append(tg[names[k]][i]); tpicks[names[k]] += 1
    zp = write_submission(te["raw_ids"], tsel, out_dir / "submission")
    s = {"run": out_dir.name, "members": [r.name for r in run_dirs], "dev": scores, "dev_picks": dict(picks),
         "test_picks": dict(tpicks), "submission": str(zp)}
    save_json(s, out_dir / "summary.json")
    return s

def length_sweep(run_dir, data, word_limits=(100, 200, 300, 400, 600, 800), bert=None):
    gens = json.load(open(Path(run_dir) / "dev_generations.json"))
    refs = [g["reference"] for g in gens]
    rows = []
    for L in list(word_limits) + [None]:
        cut = [" ".join(g["generated"].split()[:L]) if L else g["generated"] for g in gens]
        r = rouge_scores(cut, refs)
        row = {"max_words": L or "full", "mean_words": round(float(np.mean([len(c.split()) for c in cut])), 1),
               "rouge1_f": round(r["rouge1_f"], 4), "rouge1_r": round(r["rouge1_r"], 4), "rougeL_f": round(r["rougeL_f"], 4)}
        if bert is not None:
            b = bertscore(cut, refs, **bert)
            if b:
                row["bertscore_f"] = round(b["bertscore_f"], 4)
        rows.append(row)
    return rows

def generation_variants(run_dir, data, variants, n_docs=100, device="cuda", bert=None, log=print):
    model, cfg, tok = load_run(run_dir, device)
    dev = data["dev"]
    idx = list(range(min(n_docs, len(dev["ids"]))))
    spec = cfg["resolved"]
    rows = []
    for name, v in variants.items():
        texts, ntok, speed = summarise(model, dev, idx, cfg, spec, tok, device, v.get("gen"), v.get("context"))
        texts = [t if t.strip() else "." for t in texts]
        sc, _ = score_generations(texts, dev, idx)
        row = {"variant": name, "rouge1_f": round(sc["rouge1_f"], 4), "rouge2_f": round(sc["rouge2_f"], 4),
               "rougeL_f": round(sc["rougeL_f"], 4), "words": round(sc["words"], 1), "rep_3gram": round(sc["rep_3gram"], 4),
               "novel_3gram": round(sc["novel_3gram"], 4), "fact_precision": round(sc["fact_precision"], 4),
               "fact_recall": round(sc["fact_recall"], 4), "docs_per_s": speed["docs_per_sec"]}
        if bert is not None:
            b = bertscore(texts, [dev["summaries"][i] for i in idx], **bert)
            if b:
                row["bertscore_f"] = round(b["bertscore_f"], 4)
        rows.append(row)
        log(f"[variants] {name}: " + " | ".join(f"{k} {v}" for k, v in row.items() if k != "variant"))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows

def regenerate(run_dir, data, out_dir, gen=None, post=None, device="cuda", log=print):
    from .train import fill_empty
    model, cfg, tok = load_run(run_dir, device)
    cfg = merge(cfg, {"gen": gen or {}, "post": post or {}})
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    spec = cfg["resolved"]
    dev, te = data["dev"], data["test"]
    gidx = list(range(min(cfg["eval"]["gen_docs"], len(dev["ids"]))))
    texts, ntok, speed = summarise(model, dev, gidx, cfg, spec, tok, device, log=log)
    texts, _ = fill_empty(texts, dev, gidx)
    scores, per_doc = score_generations(texts, dev, gidx)
    save_json([{"id": dev["ids"][i], "reference": dev["summaries"][i], "generated": t, "new_tokens": n} |
               {k: v[j] for k, v in per_doc.items()} for j, (i, t, n) in enumerate(zip(gidx, texts, ntok))],
              out_dir / "dev_generations.json")
    tidx = list(range(len(te["ids"])))
    tt, tn, _ = summarise(model, te, tidx, cfg, spec, tok, device, log=log)
    tt, n_empty = fill_empty(tt, te, tidx)
    save_json([{"id": te["ids"][i], "generated": t, "new_tokens": n} for i, t, n in zip(tidx, tt, tn)],
              out_dir / "test_generations.json")
    zp = write_submission(te["raw_ids"], tt, out_dir / "submission")
    s = {"run": out_dir.name, "from_run": str(run_dir), "gen": cfg["gen"], "post": cfg["post"], "dev": scores,
         "generation": speed, "submission": str(zp), "test_empty_replaced": n_empty}
    save_json(cfg, out_dir / "config.json")
    save_json(s, out_dir / "summary.json")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return s

def bertscore_run(run_dir, bert_kw):
    run_dir = Path(run_dir)
    out = run_dir / "dev_bertscore.json"
    if out.exists():
        return json.load(open(out))
    gens = json.load(open(run_dir / "dev_generations.json"))
    b = bertscore([g["generated"] for g in gens], [g["reference"] for g in gens], **bert_kw)
    if b is None:
        return None
    for g, f in zip(gens, b["per_doc_f"]):
        g["bertscore_f"] = f
    save_json(gens, run_dir / "dev_generations.json")
    res = {k: v for k, v in b.items() if k != "per_doc_f"}
    save_json(res, out)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=2")
    ap.add_argument("--eda", action="store_true")
    ap.add_argument("--mbr", nargs="*", default=None)
    ap.add_argument("--out", default="runs/ENS_MBR")
    a = ap.parse_args()
    cfg = json.load(open(a.config)) if a.config else {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        set_by_path(cfg, k, v)
    from .train import run, resolve_ckpt
    if a.eda or a.mbr:
        c = merge(DEFAULTS, cfg)
        c["backbone"]["ckpt"] = resolve_ckpt(c["backbone"]["ckpt"], [c["ckpt_root"]] if c.get("ckpt_root") else [])
        data = load_data(c)
        if a.eda:
            D.describe_raw(c["data_dir"])
            print(json.dumps(D.stats(data), indent=1))
            tok = BPETokenizer.load(str(Path(c["backbone"]["ckpt"]).parent / "tokenizer.json"))
            print(json.dumps(coverage_analysis(data, tok), indent=1))
        if a.mbr:
            print(json.dumps(mbr_ensemble(a.mbr, a.out, data)["dev"], indent=1))
        return
    run(cfg)

if __name__ == "__main__":
    main()
