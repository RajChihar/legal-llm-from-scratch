"""Contrastive training, pool encoding, evaluation, submission and run management for prior case retrieval.

Training instances = (train query, one of its relevant candidates). A batch holds B instances with distinct queries;
its candidate set = their positives + `neg.n_hard` hard negatives per query. Hard negatives are drawn from the top
`neg.top` non-relevant candidates of a miner ranking over the train pool:
    bm25      lexical BM25 ranking (fixed)
    teacher   the KD teacher's ranking (fixed)
    model     the model's own ranking, re-mined every `neg.refresh_every` epochs (ANCE-style; first epoch: BM25)
    none      no hard negatives (in-batch negatives only)
`neg.in_batch` = False restricts every query to its own positive + own hard negatives. Other relevant candidates of
the same query are always masked (never treated as negatives); a query is never its own candidate.
Chunk dropout: during training every document is represented by `chunks.train_k` randomly chosen chunks of its K
selected chunks (cheaper steps, and the mean-pooled representation becomes robust to which chunks are seen);
evaluation always uses all K.
An "epoch" = one pass over the training instances (or `train.inst_per_epoch` of them). Dev micro-F1@7 (ranking the
whole dev pool) is measured before training and after every epoch; the best epoch is kept; `train.max_minutes`
stops a run early (evaluating first).
"""
import hashlib
import json
import math
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .bm25 import bm25_scores
from .metrics import rank, report, top_k_lists
from .model import build_model, kd_loss, ranking_loss, similarity, trainable_state
from .peft import count_params, is_new_param
from .tokenizer import BPETokenizer

# ------------------------------------------------------------------------------------------- config
DEFAULTS = {
    "run_name": "debug",
    "out_dir": "runs",
    "data_dir": "",
    "cache_dir": "",          # token + BM25 cache (default: <out_dir>/../p7cache)
    "ckpt_root": "",
    "backbone": {"ckpt": "", "attention": "bidirectional", "keep_layers": None, "c_segment": 0, "grad_ckpt": True},
    "method": {"name": "full"},
    "chunks": {"len": 512, "k": 4, "strategy": "uniform", "overlap": 0.0, "train_k": 2},
    "rep": {"pool": "mean", "agg": "mean", "proj": 256, "sim": "cos", "tau": 0.05, "learn_tau": False},
    "loss": {"name": "infonce"},
    "neg": {"miner": "bm25", "n_hard": 3, "top": 100, "skip_top": 0, "in_batch": True, "refresh_every": 1},
    "kd": {"teacher_run": "", "kd_alpha": 0.7, "tau_t": 0.05},
    "bm25": {"k1": 1.2, "b": 0.75, "ngrams": [1], "query_tf": "sat"},
    "train": {"epochs": 4, "inst_per_epoch": None, "batch_queries": 8, "grad_accum": 1, "lr_backbone": 2e-5,
              "lr_new": 1e-3, "weight_decay": 0.01, "warmup_frac": 0.06, "patience": 2, "grad_clip": 1.0,
              "seed": 1337, "llrd": 1.0, "max_minutes": None, "eval_docs_bs": 16, "chunk_bs": 32, "log_every": 0.1},
    "predict_test": True,
    "max_queries": None,      # debug: first n queries per split
    "max_cands": None,        # debug: gold candidates + n others per pool
    "n_proc": 4,
}


def merge(base, over):
    out = json.loads(json.dumps(base))
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def set_by_path(cfg, dotted, value):
    try:
        value = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        pass
    d = cfg
    ks = dotted.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = value


CKPT_FOLDERS = {"A": "A_llama_decoder", "A2": "A_llama_decoder_v2", "B": "B_encoder_decoder",
                "C": "C_hybrid_prefix_mlm", "D": "D_efficient_swa_mqa"}


def resolve_ckpt(spec, roots=()):
    """'@C' -> path of C_hybrid_prefix_mlm/model_final.pt found under `roots` or /kaggle/input; paths pass through."""
    if not spec.startswith("@"):
        return spec
    folder = CKPT_FOLDERS[spec[1:]]
    for r in [*roots, "/kaggle/input", "/kaggle/working"]:
        r = Path(r)
        if not r.exists():
            continue
        direct = r / folder / "model_final.pt"
        if direct.exists():
            return str(direct)
        for p in r.rglob("model_final.pt"):
            if p.parent.name == folder:
                return str(p)
    raise FileNotFoundError(f"{folder}/model_final.pt not found under {roots} or /kaggle/input")


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def tok_path(cfg):
    return str(Path(cfg["backbone"]["ckpt"]).parent / "tokenizer.json")


def cache_dir(cfg):
    return Path(cfg.get("cache_dir") or Path(cfg.get("out_dir", "runs")).resolve().parent / "p7cache")


def load_data(cfg, log=print):
    return D.load_cached(cfg["data_dir"], tok_path(cfg), cache_dir(cfg), cfg.get("n_proc", 4), cfg.get("max_queries"),
                         cfg.get("max_cands"), log)


def bm25_cached(data, split, cfg):
    """BM25 scores [n_queries, n_pool] of a split, cached in cache_dir (shared by all runs)."""
    bc = cfg["bm25"]
    d = data[split]
    key = hashlib.sha1(json.dumps([bc, d["q_ids"], d["c_ids"]]).encode()).hexdigest()[:12]
    p = cache_dir(cfg) / f"bm25_{split}_{key}.npy"
    if p.exists():
        return np.load(p)
    s = bm25_scores(d, bc["k1"], bc["b"], tuple(bc["ngrams"]), bc["query_tf"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(tmp, s); os.replace(tmp, p)
    return s


def amp_dtype(device):
    if device != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2, default=str)


# ------------------------------------------------------------------------------------------- chunks
def select_chunks(n, L, K, strategy="uniform", overlap=0.0):
    """Token spans [(start, end)] (in document order) that represent a case of n tokens (Phase-5 function)."""
    if n <= L:
        return [(0, n)]
    if n <= K * L:                                       # everything fits: consecutive chunks, nothing dropped
        m = math.ceil(n / L)
        return [(i * L, min(n, (i + 1) * L)) for i in range(m)]
    stride = max(1, int(round(L * (1 - overlap))))
    if strategy == "head":
        return [(s, s + L) for s in range(0, stride * K, stride)]
    if strategy == "tail":
        return [(n - L - s, n - s) for s in range(stride * (K - 1), -1, -stride)]
    if strategy == "headtail":
        if K == 1:
            return [(n - L, n)]
        return [(0, L)] + [(n - L - s, n - s) for s in range(stride * (K - 2), -1, -stride)]
    if strategy == "uniform":
        if K == 1:
            s = (n - L) // 2
            return [(s, s + L)]
        starts = np.linspace(0, n - L, K).round().astype(int)
        return [(int(s), int(s) + L) for s in starts]
    if strategy == "middle":
        s0 = (n - K * stride - (L - stride)) // 2
        return [(s0 + i * stride, s0 + i * stride + L) for i in range(K)]
    raise ValueError(f"unknown chunk strategy {strategy}")


def collate_docs(tokens, cc, pad_id, rng=None):
    """List of token arrays -> {"ids", "keep" [N,T], "doc", "slot" [N], "n_docs"}. rng: chunk dropout (train_k)."""
    spans = []
    for t in tokens:
        sp = select_chunks(len(t), cc["len"], cc["k"], cc["strategy"], cc.get("overlap", 0.0))
        tk = cc.get("train_k")
        if rng is not None and tk and len(sp) > tk:
            sp = [sp[i] for i in sorted(rng.sample(range(len(sp)), tk))]
        spans.append(sp)
    N = sum(len(s) for s in spans)
    T = max(e - s for sp in spans for s, e in sp)
    ids = torch.full((N, T), pad_id, dtype=torch.long)
    keep = torch.zeros((N, T), dtype=torch.bool)
    doc = torch.zeros(N, dtype=torch.long)
    slot = torch.zeros(N, dtype=torch.long)
    r = 0
    for b, (tk, sp) in enumerate(zip(tokens, spans)):
        for k, (s, e) in enumerate(sp):
            ids[r, :e - s] = torch.from_numpy(tk[s:e].astype(np.int64))
            keep[r, :e - s] = True
            doc[r], slot[r] = b, k
            r += 1
    return {"ids": ids, "keep": keep, "doc": doc, "slot": slot, "n_docs": len(tokens)}


def to_dev(b, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


# ------------------------------------------------------------------------------------------- encoding / eval
@torch.no_grad()
def encode_docs(model, tokens, cfg, device, chunk_override=None, pad_id=0):
    """All documents -> rep dict on `device`: doc [n,dim], chunks [n,K,dim], mask [n,K] (input order)."""
    model.eval()
    dt = amp_dtype(device)
    cc = {**cfg["chunks"], **(chunk_override or {})}
    order = np.argsort([len(t) for t in tokens], kind="stable")
    bs = cfg["train"].get("eval_docs_bs", 16)
    K = cc["k"]
    n = len(tokens)
    out = {"doc": torch.zeros(n, model.dim, device=device), "chunks": torch.zeros(n, K, model.dim, device=device),
           "mask": torch.zeros(n, K, dtype=torch.bool, device=device)}
    for s in range(0, n, bs):
        idx = order[s:s + bs]
        b = to_dev(collate_docs([tokens[i] for i in idx], cc, pad_id), device)
        with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
            r = model.encode(b, cfg["train"].get("chunk_bs", 32))
        it = torch.as_tensor(idx, device=device)
        k = r["chunks"].size(1)
        out["doc"][it] = r["doc"].float()
        out["chunks"][it, :k] = r["chunks"].float()
        out["mask"][it, :k] = r["mask"]
    return out


@torch.no_grad()
def score_split(model, split, cfg, device, chunk_override=None, pad_id=0, return_reps=False):
    """Similarity of every query of a split with every candidate of its pool -> np [n_q, n_c]."""
    q = encode_docs(model, split["q_tokens"], cfg, device, chunk_override, pad_id)
    c = encode_docs(model, split["c_tokens"], cfg, device, chunk_override, pad_id)
    s = similarity(q, c, model.sim).cpu().numpy()
    return (s, q, c) if return_reps else s


def evaluate(model, split, cfg, device, pad_id=0):
    s = score_split(model, split, cfg, device, pad_id=pad_id)
    rep, per, ranks = report(s, split["q_ids"], split["c_ids"], split["gold"])
    return rep, s, per, ranks


# ------------------------------------------------------------------------------------------- submission
def check_prediction(obj, test_split, k=7):
    """Codabench rules: {query id: [k candidate ids]} for every test query, ids unique, from the test pool,
    ranked (order is kept as given)."""
    assert isinstance(obj, dict), "prediction.json must be a JSON object {query_id: [candidate ids]}"
    want = {str(r) for r in test_split["q_raw"]}
    assert set(obj) == want, f"{len(want - set(obj))} test queries missing, {len(set(obj) - want)} unknown ids"
    pool = {str(r) for r in test_split["c_raw"]}
    for q, lst in obj.items():
        assert isinstance(lst, list) and len(lst) == k, f"{q}: {len(lst) if isinstance(lst, list) else lst} predictions"
        assert all(isinstance(x, str) for x in lst), f"{q}: candidate ids must be strings"
        assert len(set(lst)) == k, f"{q}: duplicate candidates"
        assert set(lst) <= pool, f"{q}: candidates outside the test pool {set(lst) - pool}"
        assert D.canon(q) not in {D.canon(x) for x in lst}, f"{q}: a query cannot be its own prior case"


def write_submission(scores, test_split, out_dir, k=7):
    """prediction.json = {query id (as in test.json): [7 candidate ids (as in test_candidates.json)]} at the zip root."""
    import zipfile
    raw_c = {c: str(r) for c, r in zip(test_split["c_ids"], test_split["c_raw"])}
    tops = top_k_lists(scores, test_split["q_ids"], test_split["c_ids"], k)
    obj = {str(qr): [raw_c[c] for c in top] for qr, top in zip(test_split["q_raw"], tops)}
    check_prediction(obj, test_split, k)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pj = out_dir / "prediction.json"
    json.dump(obj, open(pj, "w", encoding="utf-8"), indent=1)
    zp = out_dir / "submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"]
        check_prediction(json.loads(z.read("prediction.json")), test_split, k)
    return zp


# ------------------------------------------------------------------------------------------- run io
def load_run(run_dir, device="cuda", weights="best.pt"):
    run_dir = Path(run_dir)
    cfg = json.load(open(run_dir / "config.json"))
    model, _ = build_model(cfg)
    sd = torch.load(run_dir / weights, map_location="cpu")
    _, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    return model.to(device), cfg


def save_scores(path, split, s):
    np.savez_compressed(path, q_ids=np.array(split["q_ids"], dtype=object), c_ids=np.array(split["c_ids"], dtype=object),
                        scores=s.astype(np.float32))


def load_scores(path, split=None):
    z = np.load(path, allow_pickle=True)
    if split is not None:
        assert list(z["q_ids"]) == list(split["q_ids"]) and list(z["c_ids"]) == list(split["c_ids"]), \
            f"{path}: ids differ from the cached split"
    return z["scores"]


# ------------------------------------------------------------------------------------------- training
def make_optimizer(model, tc, device):
    n_layers = len(model.backbone.blocks)
    llrd = tc.get("llrd", 1.0)
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_new_param(n) or n.startswith("head."):
            lr, depth = tc["lr_new"], "new"
        else:
            if ".blocks." in n:
                depth = int(n.split(".blocks.")[1].split(".")[0]) + 1
            elif "tok_emb" in n or "seg_emb" in n:
                depth = 0
            else:
                depth = n_layers + 1
            lr = tc["lr_backbone"] * llrd ** (n_layers + 1 - depth)
        wd = tc["weight_decay"] if p.dim() >= 2 else 0.0
        groups.setdefault((depth, wd), {"params": [], "lr": lr, "weight_decay": wd})["params"].append(p)
    return torch.optim.AdamW(list(groups.values()), fused=(device == "cuda"))


def mine_negatives(scores, split, top=100, skip_top=0):
    """-> per query: array of pool indices of its top-ranked non-relevant candidates (self excluded)."""
    order = rank(scores, split["q_ids"], split["c_ids"], min(scores.shape[1], top + skip_top + 64))
    out = []
    for i, q in enumerate(split["q_ids"]):
        g = set(split["gold"][q])
        cand = [j for j in order[i] if split["c_ids"][j] not in g and split["c_ids"][j] != q]
        out.append(np.array(cand[skip_top:skip_top + top], dtype=np.int64))
    return out


def instances(split):
    pos = {c: j for j, c in enumerate(split["c_ids"])}
    return [(i, pos[c]) for i, q in enumerate(split["q_ids"]) for c in split["gold"][q]]


def query_batches(inst, B, rng):
    """Shuffled instances grouped into batches of B with distinct queries."""
    inst = list(inst)
    rng.shuffle(inst)
    batches, cur, used, rest = [], [], set(), []
    for x in inst:
        if x[0] in used:
            rest.append(x); continue
        cur.append(x); used.add(x[0])
        if len(cur) == B:
            batches.append(cur); cur, used = [], set()
            if rest:                                         # retry the postponed instances first
                inst_rest, rest = rest, []
                for y in inst_rest:
                    if y[0] in used:
                        rest.append(y); continue
                    cur.append(y); used.add(y[0])
                    if len(cur) == B:
                        batches.append(cur); cur, used = [], set()
    tail = cur + rest
    for k in range(0, len(tail), B):                          # leftovers (may repeat a query: masked correctly)
        batches.append(tail[k:k + B])
    return batches


def build_batch(batch, split, negs, n_hard, in_batch, rng):
    """-> query idx [B], candidate pool idx list, target [B], allowed / positives / own_negs [B, Nc] bool."""
    qidx = [q for q, _ in batch]
    cand, col = [], {}

    def add(j):
        if j not in col:
            col[j] = len(cand); cand.append(j)
        return col[j]
    tcol = [add(p) for _, p in batch]
    own = []
    for (qi, _), ng in zip(batch, negs):
        pick = rng.sample(list(ng), min(n_hard, len(ng))) if n_hard and len(ng) else []
        own.append([add(int(j)) for j in pick])
    B, Nc = len(batch), len(cand)
    positives = torch.zeros(B, Nc, dtype=torch.bool)
    own_negs = torch.zeros(B, Nc, dtype=torch.bool)
    selfc = torch.zeros(B, Nc, dtype=torch.bool)
    c_ids = split["c_ids"]
    for b, qi in enumerate(qidx):
        g = set(split["gold"][split["q_ids"][qi]])
        for j, cj in enumerate(cand):
            positives[b, j] = c_ids[cj] in g
            selfc[b, j] = c_ids[cj] == split["q_ids"][qi]
        own_negs[b, own[b]] = True
    target = torch.tensor(tcol)
    is_t = torch.zeros(B, Nc, dtype=torch.bool)
    is_t[torch.arange(B), target] = True
    allowed = (~positives | is_t) & ~selfc
    if not in_batch:
        allowed &= is_t | own_negs
    own_negs &= ~positives
    return qidx, cand, target, allowed, positives & ~selfc, own_negs


class Teacher:
    """Frozen KD teacher (a finished run): its representations of the train queries and the train pool, computed
    once with the teacher's own chunking and cached in the teacher's run folder."""
    def __init__(self, run_dir, tr, device, log=print):
        run_dir = Path(run_dir)
        cache = run_dir / "teacher_train_reps.pt"
        self.cfg = json.load(open(run_dir / "config.json"))
        self.sim = self.cfg["rep"].get("sim", "cos")
        key = hashlib.sha1(json.dumps([tr["q_ids"], tr["c_ids"]]).encode()).hexdigest()[:12]
        if cache.exists() and torch.load(cache, map_location="cpu").get("key") == key:
            obj = torch.load(cache, map_location="cpu")
        else:
            t0 = time.time()
            model, cfg = load_run(run_dir, device)
            q = encode_docs(model, tr["q_tokens"], cfg, device)
            c = encode_docs(model, tr["c_tokens"], cfg, device)
            obj = {"key": key, "q": {k: v.cpu() for k, v in q.items()}, "c": {k: v.cpu() for k, v in c.items()},
                   "params": count_params(model)["total"]}
            tmp = cache.with_suffix(f".{os.getpid()}.tmp")
            torch.save(obj, tmp); os.replace(tmp, cache)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            log(f"[kd] teacher representations of {len(tr['q_ids'])} queries + {len(tr['c_ids'])} candidates "
                f"in {time.time() - t0:.0f}s")
        self.q = {k: v.to(device) for k, v in obj["q"].items()}
        self.c = {k: v.to(device) for k, v in obj["c"].items()}
        self.params = obj["params"]

    def scores(self, qidx, cidx):
        sub = lambda r, i: {k: v[i] for k, v in r.items()}
        qi = torch.as_tensor(qidx, device=self.q["doc"].device)
        ci = torch.as_tensor(cidx, device=self.c["doc"].device)
        return similarity(sub(self.q, qi), sub(self.c, ci), self.sim)

    def all_scores(self):
        return similarity(self.q, self.c, self.sim).cpu().numpy()


def fit(model, data, cfg, device, log, run_dir, teacher=None, pad_id=0):
    tc, nc, kc = cfg["train"], cfg["neg"], cfg["kd"]
    tr, dev = data["train"], data["dev"]
    inst = instances(tr)
    ipe = min(len(inst), tc.get("inst_per_epoch") or len(inst))
    B = tc["batch_queries"]
    steps_per_epoch = math.ceil(math.ceil(ipe / B) / tc["grad_accum"])
    total = max(1, steps_per_epoch * tc["epochs"])
    warm = max(1, int(tc["warmup_frac"] * total))
    rng = random.Random(tc["seed"])
    miner = nc["miner"]
    if miner == "none" or not nc["n_hard"]:
        negs = [np.zeros(0, dtype=np.int64)] * len(tr["q_ids"])
    elif miner == "teacher":
        assert teacher is not None, "neg.miner = teacher needs kd.teacher_run"
        negs = mine_negatives(teacher.all_scores(), tr, nc["top"], nc["skip_top"])
    else:                                                    # bm25 (and the first epoch of "model")
        negs = mine_negatives(bm25_cached(data, "train", cfg), tr, nc["top"], nc["skip_top"])
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(model, tc, device) if trainable else None
    base_lrs = [g["lr"] for g in opt.param_groups] if opt else []
    dt = amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(dt == torch.float16))
    log(f"[train] {len(inst)} (query, relevant) instances from {len(tr['q_ids'])} queries | pool {len(tr['c_ids'])} | "
        f"{tc['epochs']} epochs x {ipe} | {steps_per_epoch} steps/epoch | batch {B} queries + {nc['n_hard']} hard "
        f"negatives each ({miner}) | in-batch {nc['in_batch']} | loss {cfg['loss']['name']} | amp {dt}")
    hist = []
    rep0, s0, _, _ = evaluate(model, dev, cfg, device, pad_id)
    hist.append({"epoch": 0, **{k: round(v, 4) for k, v in rep0.items() if isinstance(v, float)}})
    log(f"[epoch] epoch 0 (before fine-tuning) | dev micro-F1@7 {rep0['micro_f1@7']:.4f} | recall@100 {rep0['recall@100']:.4f}")
    best, best_epoch, bad, step = rep0["micro_f1@7"], 0, 0, 0
    torch.save(trainable_state(model), run_dir / "best.pt")
    t0, limit, stop = time.time(), tc.get("max_minutes"), False
    for ep in range(1, tc["epochs"] + 1):
        if not trainable:
            break
        if miner == "model" and ep > 1 and (ep - 1) % max(1, nc.get("refresh_every", 1)) == 0:
            tm = time.time()
            negs = mine_negatives(score_split(model, tr, cfg, device, pad_id=pad_id), tr, nc["top"], nc["skip_top"])
            log(f"[neg] re-mined hard negatives with the current model in {time.time() - tm:.0f}s")
        model.train()
        batches = query_batches(inst, B, rng)[:math.ceil(ipe / B)]
        every = max(1, int(len(batches) * tc.get("log_every", 0.1)))
        tot, parts, nb = 0.0, {}, 0
        for bi, batch in enumerate(batches):
            qidx, cand, target, allowed, positives, own_negs = build_batch(
                batch, tr, [negs[q] for q, _ in batch], nc["n_hard"], nc["in_batch"], rng)
            qb = to_dev(collate_docs([tr["q_tokens"][i] for i in qidx], cfg["chunks"], pad_id, rng), device)
            cb = to_dev(collate_docs([tr["c_tokens"][j] for j in cand], cfg["chunks"], pad_id, rng), device)
            target, allowed, positives, own_negs = (x.to(device) for x in (target, allowed, positives, own_negs))
            with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
                q = model.encode(qb, tc.get("chunk_bs"))
                c = model.encode(cb, tc.get("chunk_bs"))
            S = model.scores(q, c)
            loss = ranking_loss(S, target, allowed, positives, cfg["loss"]["name"], own_negs)
            pr = {"task": float(loss.detach())}
            if teacher is not None:
                T = teacher.scores(qidx, cand) / kc["tau_t"]
                kl = kd_loss(S, T, allowed)
                loss = (1 - kc["kd_alpha"]) * loss + kc["kd_alpha"] * kl
                pr["kd"] = float(kl.detach())
            scaler.scale(loss / tc["grad_accum"]).backward()
            tot += float(loss.detach()); nb += 1
            for k, v in pr.items():
                parts[k] = parts.get(k, 0.0) + v
            if nb % tc["grad_accum"] == 0 or bi == len(batches) - 1:
                lr_mult = (step + 1) / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))
                for g, b0 in zip(opt.param_groups, base_lrs):
                    g["lr"] = b0 * lr_mult
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, tc["grad_clip"])
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                step += 1
            if (bi + 1) % every == 0:
                el = (time.time() - t0) / 60
                done = (ep - 1) + (bi + 1) / len(batches)
                log(f"[progress] epoch {ep} {100 * (bi + 1) / len(batches):3.0f}% | loss {tot / nb:.4f} | "
                    + " ".join(f"{k} {v / nb:.4f}" for k, v in parts.items())
                    + f" | {el:.1f} min | ETA all epochs {el / done * (tc['epochs'] - done):.1f} min")
            if limit and (time.time() - t0) / 60 > limit:
                log(f"[train] time limit {limit} min reached in epoch {ep}: evaluating and stopping")
                stop = True
                break
        rep, _, _, _ = evaluate(model, dev, cfg, device, pad_id)
        rec = {"epoch": ep, "train_loss": round(tot / max(nb, 1), 4), "minutes": round((time.time() - t0) / 60, 2),
               **{f"train_{k}": round(v / max(nb, 1), 4) for k, v in parts.items()},
               **{k: round(v, 4) for k, v in rep.items() if isinstance(v, float)}}
        if rep["micro_f1@7"] > best:
            best, best_epoch, bad = rep["micro_f1@7"], ep, 0
            torch.save(trainable_state(model), run_dir / "best.pt"); rec["saved"] = True
        else:
            bad += 1
        hist.append(rec)
        log("[epoch] " + " | ".join(f"{k} {v}" for k, v in rec.items()))
        save_json(hist, run_dir / "history.json")
        if stop:
            break
        if bad >= tc["patience"]:
            log(f"[train] early stop: no dev improvement for {tc['patience']} epochs")
            break
    save_json(hist, run_dir / "history.json")
    return hist, best, best_epoch, len(inst)


def run(cfg, data=None, device=None, log=print):
    cfg = merge(DEFAULTS, cfg)
    roots = [cfg["ckpt_root"]] if cfg.get("ckpt_root") else []
    cfg["backbone"]["ckpt"] = resolve_ckpt(cfg["backbone"]["ckpt"], roots)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(cfg["train"]["seed"])
    run_dir = Path(cfg["out_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, run_dir / "config.json")
    data = data or load_data(cfg, log)
    tr, dev, te = data["train"], data["dev"], data["test"]
    pad_id = BPETokenizer.load(tok_path(cfg)).pad_id
    log(f"[data] train {len(tr['q_ids'])} queries / {len(tr['c_ids'])} candidates | dev {len(dev['q_ids'])} / "
        f"{len(dev['c_ids'])} | test {len(te['q_ids'])} / {len(te['c_ids'])}")
    teacher = None
    if cfg["kd"].get("teacher_run"):
        teacher = Teacher(cfg["kd"]["teacher_run"], tr, device, log)
        log(f"[kd] teacher {Path(cfg['kd']['teacher_run']).name} ({teacher.params / 1e6:.1f}M) | alpha "
            f"{cfg['kd']['kd_alpha']} | tau_t {cfg['kd']['tau_t']}")
    model, info = build_model(cfg)
    model.to(device)
    pc = count_params(model)
    info.update(params=pc)
    save_json(info, run_dir / "model_info.json")
    log(f"[model] arch {info['arch']} ({info['class']}) | layers {info['layers_used']}/{info['phase1_layers']} | "
        f"attention {info['attention']} | method {cfg['method']['name']} | chunks {cfg['chunks']['k']}x{cfg['chunks']['len']} "
        f"{cfg['chunks']['strategy']} (train {cfg['chunks'].get('train_k')}) | rep {cfg['rep']['pool']}/{cfg['rep']['agg']}"
        f"/{cfg['rep']['sim']} proj {cfg['rep'].get('proj')}")
    log(f"[model] params total {pc['total'] / 1e6:.2f}M | trainable {pc['trainable'] / 1e6:.3f}M ({pc['trainable_pct']}%)"
        + (f" | 4-bit NF4 weights {pc['nf4_weights'] / 1e6:.2f}M" if pc.get("nf4_weights") else ""))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    hist, best, best_epoch, n_inst = fit(model, data, cfg, device, log, run_dir, teacher, pad_id)
    train_min = (time.time() - t1) / 60
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
    del teacher
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device), strict=False)
    t2 = time.time()
    rep, s_dev, per, ranks = evaluate(model, dev, cfg, device, pad_id)
    enc_s = time.time() - t2
    save_scores(run_dir / "dev_scores.npz", dev, s_dev)
    save_json({"report": rep, "per_query_f1": per, "gold_ranks": ranks}, run_dir / "dev_report.json")
    log(f"[dev] micro-F1@7 {rep['micro_f1@7']:.4f} | P@7 {rep['micro_p@7']:.4f} | R@7 {rep['micro_r@7']:.4f} | "
        f"recall@100 {rep['recall@100']:.4f} | MRR {rep['mrr']:.4f}")
    summary = {"run": cfg["run_name"], "arch": info["arch"], "backbone": Path(cfg["backbone"]["ckpt"]).parent.name,
               "method": cfg["method"], "chunks": cfg["chunks"], "rep": cfg["rep"], "loss": cfg["loss"]["name"],
               "neg": cfg["neg"], "attention": info["attention"], "layers": info["layers_used"],
               "params_total": pc["total"], "params_trainable": pc["trainable"], "trainable_pct": pc["trainable_pct"],
               "nf4_weights": pc.get("nf4_weights", 0), "train_minutes": round(train_min, 1), "peak_train_mem_gb": peak,
               "best_epoch": best_epoch, "epochs_run": len(hist) - 1, "train_instances": n_inst,
               "kd_teacher": cfg["kd"].get("teacher_run") or None, "dev_before": hist[0], "dev": rep,
               "dev_encode_seconds": round(enc_s, 1),
               "docs_per_sec": round((len(dev["q_ids"]) + len(dev["c_ids"])) / max(enc_s, 1e-6), 1)}
    if cfg.get("predict_test", True):
        s_te = score_split(model, te, cfg, device, pad_id=pad_id)
        save_scores(run_dir / "test_scores.npz", te, s_te)
        zp = write_submission(s_te, te, run_dir / "submission")
        summary["submission"] = str(zp)
        log(f"[submit] {zp} | {len(te['q_ids'])} queries x 7 candidates")
    save_json(summary, run_dir / "summary.json")
    return summary, model
