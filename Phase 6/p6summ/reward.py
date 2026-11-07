import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import data as D
from .context import source_for
from .metrics import rouge_scores
from .model import Summarizer, self_attention
from .peft import apply_method, count_params, is_new_param
from .tokenizer import BPETokenizer
from .train import (amp_dtype, fill_empty, load_run, merge, save_json, score_generations, seed_all, summarise,
                    write_submission)

RM_DEFAULTS = {
    "backbone": {"ckpt": "", "keep_layers": None},
    "method": {"name": "lorma", "variant": "plus", "r": 8, "alpha": 16, "targets": ["attn", "ffn"]},
    "src_len": 640,
    "sum_len": 352,
    "context": {"strategy": "headtail", "head_frac": 0.5, "k": 4, "lead": 256},
    "sep_text": "\n\nSUMMARY:\n",
    "pairs": {"docs": 400, "samples": 2, "temperature": 0.8, "synthetic": True, "margin": 0.02,
              "max_pairs_per_doc": 12, "val_frac": 0.1, "seed": 0},
    "train": {"epochs": 2, "batch_pairs": 8, "lr_new": 5e-4, "lr_backbone": 2e-5, "weight_decay": 0.01,
              "warmup_frac": 0.05, "grad_clip": 1.0, "reg": 1e-3, "seed": 1337, "max_minutes": 40, "eval_batch": 32},
}

def quality(texts, ref):
    r = rouge_scores(texts, [ref] * len(texts))["per_doc"]
    return [(a + b + c) / 3 for a, b, c in zip(r["rouge1_f"], r["rouge2_f"], r["rougeL_f"])]

class RewardModel(nn.Module):
    def __init__(self, ckpt, method, keep_layers=None, total_len=1024, dropout=0.1):
        super().__init__()
        assert method["name"] not in ("prefix", "prompt", "qlora"), "use lora / lorma / adapters / full for the RM"
        self.lm = Summarizer(ckpt, keep_layers, grad_ckpt=True)
        assert self.lm.arch in ("A", "C", "D"), "the reward model uses a decoder-stack model (A, C or D)"
        self.lm.set_lengths(total_len, "auto", total_len)
        apply_method(self.lm, method)
        del self.lm.lm_head
        self.lm.lm_head = None
        d = self.lm.tok_emb.weight.size(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 1))
        nn.init.zeros_(self.head[1].weight); nn.init.zeros_(self.head[1].bias)

    def _block(self, blk, x, mask):
        x = x + self_attention(blk.attn, blk.norm1(x), self.lm.rope, 0, mask)[0]
        return x + blk.ffn(blk.norm2(x))

    def forward(self, ids, seg, keep, sum_mask):
        lm = self.lm
        x = lm.drop(lm._embed(ids, seg if lm.arch == "C" else None))
        mask = keep[:, None, None, :]
        if lm.arch == "D":
            i = torch.arange(ids.size(1), device=ids.device)
            mask = mask & ((i[:, None] - i[None, :]).abs() < lm.window)[None, None]
        for blk in lm.blocks:
            if self.training:
                x = checkpoint(self._block, blk, x, mask, use_reentrant=False)
            else:
                x = self._block(blk, x, mask)
        h = lm.norm_f(x).float()
        m = sum_mask.float()[..., None]
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(pooled.to(self.head[1].weight.dtype)).squeeze(-1).float()

def rm_state(model):
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep or k.startswith("head.")}

def load_rm(rm_dir, device="cuda"):
    rm_dir = Path(rm_dir)
    cfg = json.load(open(rm_dir / "rm_config.json"))
    model = RewardModel(cfg["backbone"]["ckpt"], cfg["method"], cfg["backbone"].get("keep_layers"),
                        cfg["src_len"] + len(cfg["sep_ids"]) + cfg["sum_len"])
    _, unexpected = model.load_state_dict(torch.load(rm_dir / "rm.pt", map_location="cpu"), strict=False)
    assert not unexpected, unexpected
    tok = BPETokenizer.load(str(Path(cfg["backbone"]["ckpt"]).parent / "tokenizer.json"))
    return model.to(device).eval(), cfg, tok

class Encoder:
    def __init__(self, cfg, tok):
        self.cfg, self.tok = cfg, tok
        self.ccfg = merge(RM_DEFAULTS["context"], cfg.get("context", {}))
        self.sep = cfg["sep_ids"]
        self.gap = tok.encode(" ...")
        self._sum, self._src = {}, {}

    def src(self, split_name, split, i):
        k = (split_name, i)
        if k not in self._src:
            self._src[k] = [int(t) for t in source_for(split, i, self.ccfg, self.cfg["src_len"], self.gap)]
        return self._src[k]

    def summ(self, text):
        if text not in self._sum:
            self._sum[text] = self.tok.encode(text)[:self.cfg["sum_len"]] or [self.tok.encode(".")[0]]
        return self._sum[text]

    def batch(self, items, device):
        seqs = []
        for name, split, i, text in items:
            a, b = self.src(name, split, i) + self.sep, self.summ(text)
            seqs.append((a, b))
        T = max(len(a) + len(b) for a, b in seqs)
        B = len(seqs)
        ids = torch.full((B, T), self.tok.pad_id, dtype=torch.long)
        seg = torch.zeros((B, T), dtype=torch.long)
        keep = torch.zeros((B, T), dtype=torch.bool)
        sm = torch.zeros((B, T), dtype=torch.bool)
        for r, (a, b) in enumerate(seqs):
            n = len(a) + len(b)
            ids[r, :n] = torch.tensor(a + b)
            seg[r, len(a):n] = 1
            keep[r, :n] = True
            sm[r, len(a):n] = True
        return [t.to(device, non_blocking=True) for t in (ids, seg, keep, sm)]

@torch.no_grad()
def score(model, enc, items, device, bs=32):
    model.eval()
    dt = amp_dtype(device)
    order = sorted(range(len(items)), key=lambda k: len(enc.summ(items[k][3])))
    out = np.zeros(len(items), dtype=np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
            r = model(*enc.batch([items[k] for k in idx], device))
        out[idx] = r.float().cpu().numpy()
    return out

def _sentences(t):
    return [s.strip() for s in D.split_sentences(t) if s.strip()]

def synthetic_candidates(split, i, rng):
    ref = split["summaries"][i]
    sents = _sentences(ref)
    words = ref.split()
    out = []
    j = rng.randrange(len(split["ids"]) - 1)
    out.append(("other_doc", split["summaries"][j + (j >= i)]))
    if len(sents) >= 3 and len(set(sents)) > 1:
        sh = sents[:]
        while sh == sents:
            rng.shuffle(sh)
        out.append(("shuffled", " ".join(sh)))
    if len(words) >= 20:
        out.append(("truncated", " ".join(words[:max(8, int(0.35 * len(words)))])))
    if len(sents) >= 2:
        half = sents[:max(1, len(sents) // 2)]
        out.append(("repeated", " ".join(half + half)))
    out.append(("lead", " ".join(split["documents"][i].split()[:max(30, len(words))])))
    return out

def build_preferences(data, cfg, policy_dirs, device, log=print):
    pc = cfg["pairs"]
    tr = data["train"]
    rng = random.Random(pc["seed"])
    docs = sorted(rng.sample(range(len(tr["ids"])), min(pc["docs"], len(tr["ids"]))))
    cands = {i: [{"text": tr["summaries"][i], "kind": "reference"}] for i in docs}
    if pc.get("synthetic", True):
        for i in docs:
            cands[i] += [{"text": t, "kind": k} for k, t in synthetic_candidates(tr, i, rng)]
    for rd in policy_dirs:
        t0 = time.time()
        model, rcfg, rtok = load_run(rd, device)
        spec = rcfg["resolved"]
        variants = [("greedy", {"method": "greedy"})] + [
            (f"sample{s}", {"method": "sample", "temperature": pc["temperature"], "seed": 1000 + s})
            for s in range(pc["samples"])]
        for vname, g in variants:
            texts, _, _ = summarise(model, tr, docs, rcfg, spec, rtok, device, gcfg=g)
            texts, _ = fill_empty(texts, tr, docs)
            for i, t in zip(docs, texts):
                cands[i].append({"text": t, "kind": f"policy:{Path(rd).name}:{vname}"})
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        log(f"[rm] policy {Path(rd).name}: {len(variants)} summaries x {len(docs)} training documents in "
            f"{time.time() - t0:.0f}s")
    pairs = []
    for i in docs:
        cs = cands[i]
        for c, q in zip(cs, quality([c["text"] for c in cs], tr["summaries"][i])):
            c["q"] = float(q)
        pol = [k for k, c in enumerate(cs) if c["kind"].startswith("policy")]
        pp = [(w, l) for w in pol for l in pol if cs[w]["q"] - cs[l]["q"] > pc["margin"]]
        other = [(w, l) for w in range(len(cs)) for l in range(len(cs))
                 if (w not in pol or l not in pol) and cs[w]["q"] - cs[l]["q"] > pc["margin"]]
        rng.shuffle(pp); rng.shuffle(other)
        m = pc["max_pairs_per_doc"]
        chosen = pp[:m // 2 + (m // 2 - min(len(other), m // 2))]
        chosen += other[:m - len(chosen)]
        pairs += [(i, w, l) for w, l in chosen]
    kinds = Counter(c["kind"].split(":")[0] for i in docs for c in cands[i])
    log(f"[rm] {len(docs)} documents | candidates {dict(kinds)} | {len(pairs)} preference pairs")
    return docs, cands, pairs

def train_reward_model(cfg, data, policy_dirs, rm_dir, device=None, log=print):
    cfg = merge(RM_DEFAULTS, cfg)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(cfg["train"]["seed"])
    rm_dir = Path(rm_dir); rm_dir.mkdir(parents=True, exist_ok=True)
    tok = BPETokenizer.load(str(Path(cfg["backbone"]["ckpt"]).parent / "tokenizer.json"))
    cfg["sep_ids"] = tok.encode(cfg["sep_text"])
    t_start = time.time()
    docs, cands, pairs = build_preferences(data, cfg, policy_dirs, device, log)
    assert pairs, "no preference pairs (all candidates have the same quality?)"
    rng = random.Random(cfg["pairs"]["seed"])
    val_docs = set(rng.sample(docs, max(1, int(round(cfg["pairs"]["val_frac"] * len(docs))))))
    tr_pairs = [p for p in pairs if p[0] not in val_docs]
    va_pairs = [p for p in pairs if p[0] in val_docs]
    save_json({"n_docs": len(docs), "val_docs": sorted(val_docs),
               "candidates": {str(i): [{k: c[k] for k in ("kind", "q")} for c in cands[i]] for i in docs}},
              rm_dir / "preference_data.json")

    model = RewardModel(cfg["backbone"]["ckpt"], cfg["method"], cfg["backbone"].get("keep_layers"),
                        cfg["src_len"] + len(cfg["sep_ids"]) + cfg["sum_len"]).to(device)
    pcount = count_params(model)
    log(f"[rm] backbone {Path(cfg['backbone']['ckpt']).parent.name} ({model.lm.arch}) | method {cfg['method']['name']} | "
        f"params {pcount['total'] / 1e6:.1f}M, trainable {pcount['trainable'] / 1e6:.3f}M | "
        f"pairs train {len(tr_pairs)} / val {len(va_pairs)}")
    enc = Encoder(cfg, tok)
    tc = cfg["train"]
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        new = is_new_param(n) or n.startswith("head.")
        wd = tc["weight_decay"] if p.dim() >= 2 else 0.0
        groups.setdefault((new, wd), {"params": [], "lr": tc["lr_new"] if new else tc["lr_backbone"],
                                      "weight_decay": wd})["params"].append(p)
    opt = torch.optim.AdamW(list(groups.values()))
    base = [g["lr"] for g in opt.param_groups]
    bs = tc["batch_pairs"]
    total = max(1, math.ceil(len(tr_pairs) / bs) * tc["epochs"])
    warm = max(1, int(tc["warmup_frac"] * total))
    dt = amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(dt == torch.float16))
    trainable = [p for p in model.parameters() if p.requires_grad]
    tr_split = data["train"]

    def items(pp):
        out = []
        for i, w, l in pp:
            out.append(("train", tr_split, i, cands[i][w]["text"]))
            out.append(("train", tr_split, i, cands[i][l]["text"]))
        return out

    def val_acc():
        if not va_pairs:
            return None
        r = score(model, enc, items(va_pairs), device, tc["eval_batch"])
        return float(np.mean(r[0::2] > r[1::2]))

    hist = [{"epoch": 0, "val_pair_acc": val_acc()}]
    log(f"[rm] epoch 0 (untrained head) | val pair accuracy {hist[0]['val_pair_acc']}")
    best, step, deadline = -1.0, 0, t_start + 60 * tc["max_minutes"] if tc.get("max_minutes") else None
    torch.save(rm_state(model), rm_dir / "rm.pt")
    for ep in range(1, tc["epochs"] + 1):
        model.train()
        rng.shuffle(tr_pairs)
        tot, nb, acc = 0.0, 0, 0.0
        for s in range(0, len(tr_pairs), bs):
            batch = enc.batch(items(tr_pairs[s:s + bs]), device)
            with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
                r = model(*batch)
            rw, rl = r[0::2], r[1::2]
            loss = -F.logsigmoid(rw - rl).mean() + tc["reg"] * (r ** 2).mean()
            lr_mult = (step + 1) / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * lr_mult
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, tc["grad_clip"])
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            tot += float(loss); acc += float((rw > rl).float().mean()); nb += 1; step += 1
            if deadline and time.time() > deadline:
                break
        va = val_acc()
        rec = {"epoch": ep, "train_loss": tot / max(1, nb), "train_pair_acc": acc / max(1, nb), "val_pair_acc": va,
               "minutes": round((time.time() - t_start) / 60, 1)}
        hist.append(rec)
        log("[rm] " + " | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
        if va is None or va > best:
            best = va if va is not None else best
            torch.save(rm_state(model), rm_dir / "rm.pt")
        if deadline and time.time() > deadline:
            log("[rm] time limit reached"); break
    save_json(cfg, rm_dir / "rm_config.json")
    res = {"history": hist, "best_val_pair_acc": best, "params": pcount, "pairs_train": len(tr_pairs),
           "pairs_val": len(va_pairs), "docs": len(docs), "minutes": round((time.time() - t_start) / 60, 1),
           "policy_runs": [Path(p).name for p in policy_dirs]}
    save_json(res, rm_dir / "rm_metrics.json")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return res

def _kendall(a, b):
    n, s, c = len(a), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            x, y = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if x and y:
                s += x * y; c += 1
    return s / c if c else np.nan

def _z(v):
    v = np.asarray(v, dtype=np.float64)
    return (v - v.mean()) / (v.std() + 1e-8)

def extra_samples(run_dir, split, idx, n, temperature, device, seed=2000):
    if n <= 0:
        return [[] for _ in idx]
    model, rcfg, rtok = load_run(run_dir, device)
    out = [[] for _ in idx]
    for s in range(n):
        texts, _, _ = summarise(model, split, idx, rcfg, rcfg["resolved"], rtok, device,
                                gcfg={"method": "sample", "temperature": temperature, "seed": seed + s})
        texts, _ = fill_empty(texts, split, idx)
        for k, t in enumerate(texts):
            out[k].append(t)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out

def rerank_runs(rm_dir, data, run_dirs, out_dir, extra_from=None, n_extra=0, temperature=0.8, mbr_weight=0.5,
                device=None, log=print):
    from .run import mbr_select
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, tok = load_rm(rm_dir, device)
    enc = Encoder(cfg, tok)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [Path(r) for r in run_dirs]
    names = [r.name for r in run_dirs]
    dev, te = data["dev"], data["test"]
    dpos = {d: k for k, d in enumerate(dev["ids"])}
    dgen = {r.name: {g["id"]: g["generated"] for g in json.load(open(r / "dev_generations.json"))} for r in run_dirs}
    tgen = {r.name: {g["id"]: g["generated"] for g in json.load(open(r / "test_generations.json"))} for r in run_dirs}
    dev_ids = [d for d in dev["ids"] if all(d in dgen[n] for n in names)]
    didx = [dpos[d] for d in dev_ids]
    tidx = list(range(len(te["ids"])))
    xd = extra_samples(extra_from, dev, didx, n_extra, temperature, device) if extra_from else [[] for _ in didx]
    xt = extra_samples(extra_from, te, tidx, n_extra, temperature, device, seed=3000) if extra_from else [[] for _ in tidx]
    src_names = names + [f"{Path(extra_from).name}:sample{s}" for s in range(n_extra if extra_from else 0)]

    def pick(split_name, split, idx, cand_lists):
        items = [(split_name, split, i, t) for i, cl in zip(idx, cand_lists) for t in cl]
        r = score(model, enc, items, device, cfg["train"]["eval_batch"])
        out, k = [], 0
        for cl in cand_lists:
            rr = r[k:k + len(cl)]; k += len(cl)
            _, cons = mbr_select(cl)
            out.append({"reward": rr.tolist(), "consensus": cons,
                        "rm": int(np.argmax(rr)), "mbr": int(np.argmax(cons)),
                        "rm_mbr": int(np.argmax(_z(rr) + mbr_weight * _z(cons)))})
        return out

    t0 = time.time()
    dc = [[dgen[n][d] for n in names] + x for d, x in zip(dev_ids, xd)]
    dsel = pick("dev", dev, didx, dc)
    q = [quality(cl, dev["summaries"][i]) for cl, i in zip(dc, didx)]
    rules = {f"single:{n}": [k] * len(dc) for k, n in enumerate(src_names)}
    rules.update({"rm": [s["rm"] for s in dsel], "mbr": [s["mbr"] for s in dsel], "rm+mbr": [s["rm_mbr"] for s in dsel],
                  "oracle(ROUGE)": [int(np.argmax(x)) for x in q]})
    rr0 = random.Random(0)
    rules["random"] = [rr0.randrange(len(c)) for c in dc]
    report = {}
    for rule, ch in rules.items():
        texts = [cl[c] for cl, c in zip(dc, ch)]
        sc, _ = score_generations(texts, dev, didx)
        report[rule] = {"rouge1_f": round(sc["rouge1_f"], 4), "rouge2_f": round(sc["rouge2_f"], 4),
                        "rougeL_f": round(sc["rougeL_f"], 4), "q": round(float(np.mean([x[c] for x, c in zip(q, ch)])), 4),
                        "words": round(sc["words"], 1)}
    taus = [_kendall(s["reward"], x) for s, x in zip(dsel, q)]
    picks = Counter(src_names[s["rm"]] for s in dsel)
    report_meta = {"candidates_per_doc": len(src_names), "candidate_sources": src_names, "dev_docs": len(dc),
                   "kendall_tau_reward_vs_rouge": round(float(np.nanmean(taus)), 4),
                   "rm_top1_is_oracle": round(float(np.mean([s["rm"] == int(np.argmax(x)) for s, x in zip(dsel, q)])), 4),
                   "rm_picks": dict(picks), "seconds": round(time.time() - t0, 1)}
    save_json({"rules": report, **report_meta}, out_dir / "rerank_report.json")
    rm_texts = [cl[s["rm"]] for cl, s in zip(dc, dsel)]
    sc, per_doc = score_generations(rm_texts, dev, didx)
    save_json([{"id": dev["ids"][i], "reference": dev["summaries"][i], "generated": t, "picked": src_names[s["rm"]],
                "rewards": s["reward"]} | {k: v[j] for k, v in per_doc.items()}
               for j, (i, t, s) in enumerate(zip(didx, rm_texts, dsel))], out_dir / "dev_generations.json")
    tc_ = [[tgen[n][te["ids"][i]] for n in names] + x for i, x in zip(tidx, xt)]
    tsel = pick("test", te, tidx, tc_)
    ttexts = [cl[s["rm"]] for cl, s in zip(tc_, tsel)]
    save_json([{"id": te["ids"][i], "generated": t, "picked": src_names[s["rm"]]} for i, t, s in zip(tidx, ttexts, tsel)],
              out_dir / "test_generations.json")
    zp = write_submission(te["raw_ids"], ttexts, out_dir / "submission")
    summary = {"run": out_dir.name, "reward_model": str(rm_dir), "members": names, "extra_from": str(extra_from) if extra_from else None,
               "n_extra": n_extra, "dev": {k: sc[k] for k in ("rouge1_f", "rouge2_f", "rougeL_f", "words")},
               "test_picks": dict(Counter(src_names[s["rm"]] for s in tsel)), "submission": str(zp), **report_meta}
    save_json(summary, out_dir / "summary.json")
    log(f"[rerank] {len(dc)} dev docs x {len(src_names)} candidates | Kendall tau {report_meta['kendall_tau_reward_vs_rouge']} | "
        f"RM ROUGE-1 {report['rm']['rouge1_f']} vs MBR {report['mbr']['rouge1_f']} vs oracle {report['oracle(ROUGE)']['rouge1_f']}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return summary, report
