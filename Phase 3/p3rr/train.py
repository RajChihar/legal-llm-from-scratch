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
import torch.nn.functional as F

from . import data as D
from .metrics import format_table, full_report
from .model import build_model, crf_log_marginals_np, trainable_state, viterbi_np
from .peft import count_params, is_new_param
from .tokenizer import BPETokenizer

DEFAULTS = {
    "run_name": "debug",
    "out_dir": "runs",
    "data_dir": "",
    "backbone": {"ckpt": "", "attention": "bidirectional", "keep_layers": None, "c_segment": 0, "grad_ckpt": False},
    "tokenizer": {"path": "", "fine_punct": False},
    "sentence": {"max_tokens": 128, "head_frac": 0.75},
    "window": {"mode": "context", "max_len": 512, "overlap": 0.5, "train_overlap": 0.25, "random_offset": True},
    "method": {"name": "full"},
    "head": {"type": "linear", "context": "gru", "hidden": 256, "dropout": 0.1, "use_pos": True, "pos_buckets": 20,
             "use_domain": True},
    "loss": {"class_weight": "sqrt_inv", "cw_power": 0.5, "cw_max": 10.0, "label_smoothing": 0.0},
    "kd": {"teacher_run": "", "kd_alpha": 0.5, "kd_T": 2.0},
    "train": {"epochs": 10, "batch_size": 8, "grad_accum": 1, "lr_backbone": 5e-5, "lr_new": 1e-3,
              "weight_decay": 0.01, "warmup_frac": 0.1, "patience": 3, "grad_clip": 1.0, "seed": 1337,
              "num_workers": 0, "llrd": 1.0},
    "predict_test": True,
    "refit_full": False,
    "max_docs": None,
    "cache_dir": "",
    "ckpt_root": "",
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

def resolve_paths(cfg):
    roots = [cfg["ckpt_root"]] if cfg.get("ckpt_root") else []
    cfg["backbone"]["ckpt"] = resolve_ckpt(cfg["backbone"]["ckpt"], roots)
    return cfg

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_tokenizer(cfg):
    p = cfg["tokenizer"].get("path") or str(Path(cfg["backbone"]["ckpt"]).parent / "tokenizer.json")
    return BPETokenizer.load(p, fine_punct=cfg["tokenizer"].get("fine_punct", False))

def cache_dir(cfg):
    return Path(cfg.get("cache_dir") or Path(cfg.get("out_dir", "runs")).resolve().parent / "p3cache")

def tokenize_docs(docs, tok, cdir):
    key = hashlib.sha1((tok.fingerprint() + json.dumps(sorted((k, d["sents"]) for k, d in docs.items()))).encode()).hexdigest()[:16]
    path = Path(cdir) / f"sents_{key}.pkl"
    if path.exists():
        try:
            return pickle.load(open(path, "rb"))
        except Exception:
            pass
    out = {}
    for did, d in docs.items():
        sids = []
        for s in d["sents"]:
            ids = tok.encode(" " + s.strip()) if s.strip() else []
            if not ids:
                ids = tok.encode(" .")
            sids.append(np.asarray(ids, dtype=np.int32))
        out[did] = sids
    try:
        Path(cdir).mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        pickle.dump(out, open(tmp, "wb"), protocol=pickle.HIGHEST_PROTOCOL); os.replace(tmp, path)
    except OSError:
        pass
    return out

def truncate(ids, max_tokens, head_frac):
    if len(ids) <= max_tokens:
        return ids
    h = int(round(max_tokens * head_frac))
    return np.concatenate([ids[:h], ids[len(ids) - (max_tokens - h):]])

def prepare(docs, tok, labels, cfg):
    lab = {l: i for i, l in enumerate(labels)}
    sc = cfg["sentence"]
    nb = cfg["head"].get("pos_buckets", 20)
    toks = tokenize_docs(docs, tok, cache_dir(cfg))
    out = {}
    for did, d in docs.items():
        sents = [truncate(x, sc["max_tokens"], sc["head_frac"]) for x in toks[did]]
        n = len(sents)
        out[did] = {"sents": sents, "lens": np.array([len(x) for x in sents]), "n": n,
                    "y": None if d["labels"] is None else np.array([lab[l] for l in d["labels"]], dtype=np.int64),
                    "pos": np.minimum((np.arange(n) * nb) // max(1, n), nb - 1),
                    "dom": D.DOMAINS.index(d["domain"])}
    return out

def sent_windows(lens, L, overlap, start=0):
    n = len(lens)
    out, i = [], start
    while i < n:
        j, tot = i, 0
        while j < n and (tot + lens[j] <= L or j == i):
            tot += lens[j]; j += 1
        out.append((i, j))
        if j >= n:
            break
        i = i + max(1, int((j - i) * (1 - overlap)))
    return out

def doc_windows(p, cfg, train=False, rng=None):
    w = cfg["window"]
    if w["mode"] == "sentence":
        return [(k, k + 1) for k in range(p["n"])]
    if not train:
        return sent_windows(p["lens"], w["max_len"], w["overlap"])
    ws = sent_windows(p["lens"], w["max_len"], w["train_overlap"])
    if w.get("random_offset") and rng is not None and len(ws) > 1:
        first = ws[0][1] - ws[0][0]
        off = rng.randrange(0, max(1, first // 2 + 1))
        if off:
            ws = [ws[0]] + sent_windows(p["lens"], w["max_len"], w["train_overlap"], start=off)
    return ws

def collate(items, prep, pad_id, teacher=None):
    B = len(items)
    T = max(int(prep[d]["lens"][i:j].sum()) for d, i, j in items)
    S = max(j - i for _, i, j in items)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    keep = torch.zeros((B, T), dtype=torch.bool)
    sidx = torch.full((B, T), -1, dtype=torch.long)
    y = torch.full((B, S), -100, dtype=torch.long)
    smask = torch.zeros((B, S), dtype=torch.bool)
    pos = torch.zeros((B, S), dtype=torch.long)
    dom = torch.zeros(B, dtype=torch.long)
    K = None if teacher is None else next(iter(teacher.values())).shape[1]
    tch = torch.zeros((B, S, K), dtype=torch.float32) if teacher is not None else None
    for b, (d, i, j) in enumerate(items):
        p = prep[d]
        t = 0
        for s in range(i, j):
            x = p["sents"][s]
            ids[b, t:t + len(x)] = torch.from_numpy(x.astype(np.int64))
            sidx[b, t:t + len(x)] = s - i
            t += len(x)
        keep[b, :t] = True
        smask[b, :j - i] = True
        pos[b, :j - i] = torch.from_numpy(p["pos"][i:j])
        dom[b] = p["dom"]
        if p["y"] is not None:
            y[b, :j - i] = torch.from_numpy(p["y"][i:j])
        if tch is not None:
            tch[b, :j - i] = torch.from_numpy(teacher[d][i:j])
    return ids, keep, sidx, y, smask, pos, dom, tch

def amp_dtype(device):
    if device != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

def to_dev(xs, device):
    return [x.to(device, non_blocking=True) if x is not None else None for x in xs]

@torch.no_grad()
def predict_scores(model, prep, cfg, batch_size=16, device="cuda"):
    model.eval()
    dt = amp_dtype(device)
    causal = model.backbone.attention == "causal"
    K = len(model.labels)
    acc = {d: np.zeros((p["n"], K), dtype=np.float32) for d, p in prep.items()}
    wsum = {d: np.zeros(p["n"], dtype=np.float32) for d, p in prep.items()}
    items = sorted(((d, i, j) for d, p in prep.items() for i, j in doc_windows(p, cfg)),
                   key=lambda x: int(prep[x[0]]["lens"][x[1]:x[2]].sum()))
    for k in range(0, len(items), batch_size):
        chunk = items[k:k + batch_size]
        ids, keep, sidx, _, smask, pos, dom, _ = to_dev(collate(chunk, prep, pad_id=0), device)
        with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
            em = model.emissions(ids, keep, sidx, smask, pos, dom)
        sc = (em if model.crf is not None else F.log_softmax(em, -1)).float().cpu().numpy()
        for b, (d, i, j) in enumerate(chunk):
            n = j - i
            r = np.arange(n, dtype=np.float32)
            w = (r + 1) if causal else np.minimum(r + 1, n - r)
            acc[d][i:j] += sc[b, :n] * w[:, None]
            wsum[d][i:j] += w
    return {d: acc[d] / np.maximum(wsum[d], 1e-6)[:, None] for d in acc}

def to_logprobs(model, scores):
    if model.crf is None:
        return scores
    tr, st, en = model.crf.params_np()
    return {d: crf_log_marginals_np(s, tr, st, en) for d, s in scores.items()}

def decode_all(model, scores):
    if model.crf is None:
        return {d: [model.labels[k] for k in s.argmax(1)] for d, s in scores.items()}
    tr, st, en = model.crf.params_np()
    return {d: [model.labels[k] for k in viterbi_np(s, tr, st, en)] for d, s in scores.items()}

def evaluate(model, docs, prep, cfg, device):
    sc = predict_scores(model, prep, cfg, cfg["train"]["batch_size"] * 2, device)
    pred = decode_all(model, sc)
    gold = {k: d["labels"] for k, d in docs.items()}
    rep = full_report(gold, pred, model.labels, {k: d["domain"] for k, d in docs.items()})
    return rep, pred, sc

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2, default=str)

def load_run(run_dir, device="cuda", weights="best.pt"):
    run_dir = Path(run_dir)
    cfg = json.load(open(run_dir / "config.json"))
    labels = json.load(open(run_dir / "labels.json"))
    model, _ = build_model(cfg, labels)
    sd = torch.load(run_dir / weights, map_location="cpu")
    _, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    return model.to(device), cfg, labels

def check_prediction(pred, test_docs, labels):
    assert set(pred) == set(test_docs), (f"prediction ids differ from test ids: missing {sorted(set(test_docs) - set(pred))[:5]}, "
                                         f"unknown {sorted(set(pred) - set(test_docs))[:5]}")
    lab = set(labels)
    for d, doc in test_docs.items():
        assert len(pred[d]) == len(doc["sents"]), f"{d}: {len(pred[d])} predictions for {len(doc['sents'])} sentences"
        bad = [l for l in pred[d] if l not in lab]
        assert not bad, f"{d}: labels not in label_vocab.json: {bad[:5]}"

def write_submission(pred, test_docs, labels, out_dir):
    import zipfile
    check_prediction(pred, test_docs, labels)
    obj = {d: list(pred[d]) for d in test_docs}
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pj = out_dir / "prediction.json"
    json.dump(obj, open(pj, "w", encoding="utf-8"))
    zp = out_dir / "submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"]
        check_prediction(json.loads(z.read("prediction.json")), test_docs, labels)
    return zp

def save_scores(scores, path):
    np.savez_compressed(path, **{f"doc{i}": s for i, s in enumerate(scores.values())}, ids=np.array(list(scores), dtype=object))

def load_scores(path):
    z = np.load(path, allow_pickle=True)
    return {d: z[f"doc{i}"] for i, d in enumerate(z["ids"])}

def run(cfg, data=None, device=None, log=print):
    cfg = resolve_paths(merge(DEFAULTS, cfg))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(cfg["train"]["seed"])
    run_dir = Path(cfg["out_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, run_dir / "config.json")

    data = data or D.load_phase3(cfg["data_dir"], cfg.get("max_docs"))
    labels = data["labels"]
    save_json(labels, run_dir / "labels.json")
    train_docs, dev_docs, test_docs = data["train"], data["dev"], data["test"]
    log(f"[data] train docs {len(train_docs)} | dev docs {len(dev_docs)} | test docs {len(test_docs)} | labels {len(labels)}")
    tok = get_tokenizer(cfg)
    t0 = time.time()
    prep_tr, prep_dev = prepare(train_docs, tok, labels, cfg), prepare(dev_docs, tok, labels, cfg)
    log(f"[data] tokenised in {time.time() - t0:.0f}s | train sentences {sum(p['n'] for p in prep_tr.values()):,} | "
        f"train tokens {sum(int(p['lens'].sum()) for p in prep_tr.values()):,}")

    teacher, t_info = None, None
    if cfg["kd"].get("teacher_run"):
        assert cfg["head"]["type"] == "linear", "KD student uses the linear head"
        teacher, t_info = teacher_scores(cfg, labels, {**prep_tr, **prep_dev}, device, run_dir, log)

    model, info, pc = new_model(cfg, labels, prep_tr, device)
    save_json(info, run_dir / "model_info.json")
    log(f"[model] arch {info['arch']} ({info['class']}) | layers {info['layers_used']}/{info['phase1_layers']} | "
        f"attention {info['attention']} | method {cfg['method']['name']} | head {cfg['head']['type']}/{cfg['head']['context']} "
        f"| window {cfg['window']['mode']}")
    log(f"[model] params total {pc['total'] / 1e6:.2f}M | trainable {pc['trainable'] / 1e6:.3f}M ({pc['trainable_pct']}%)")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_train = time.time()
    hist, best, best_epoch = fit(model, prep_tr, teacher, cfg, labels, tok.pad_id, device, log,
                                 dev=(dev_docs, prep_dev), save_path=run_dir / "best.pt", hist_path=run_dir / "history.json")
    train_min = (time.time() - t_train) / 60
    peak_mem = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device), strict=False)
    summary = {"run": cfg["run_name"], "arch": info["arch"], "backbone": Path(cfg["backbone"]["ckpt"]).parent.name,
               "method": cfg["method"], "head": cfg["head"]["type"], "context": cfg["head"]["context"],
               "window_mode": cfg["window"]["mode"], "attention": info["attention"], "layers": info["layers_used"],
               "params_total": pc["total"], "params_trainable": pc["trainable"], "trainable_pct": pc["trainable_pct"],
               "train_minutes": round(train_min, 1), "peak_train_mem_gb": peak_mem, "best_epoch": best_epoch,
               "kd_teacher": t_info}
    t0 = time.time()
    rep, pred, sc = evaluate(model, dev_docs, prep_dev, cfg, device)
    dt_inf = time.time() - t0
    save_json(rep, run_dir / "dev_report.json")
    save_json(pred, run_dir / "dev_prediction.json")
    save_scores(to_logprobs(model, sc), run_dir / "dev_logprobs.npz")
    n_sent = sum(p["n"] for p in prep_dev.values())
    summary.update(dev={k: rep["all"][k] for k in ("macro_f1", "macro_p", "macro_r", "macro_f1_vocab", "accuracy")},
                   dev_CL_macro_f1=rep.get("CL", {}).get("macro_f1"), dev_IT_macro_f1=rep.get("IT", {}).get("macro_f1"),
                   infer_sentences_per_sec=round(n_sent / max(dt_inf, 1e-6), 1), top_confusions=rep["top_confusions"])
    log("\n" + format_table(rep["all"]))
    log(f"[dev] MACRO-F1 all {rep['all']['macro_f1']:.4f} | CL {summary['dev_CL_macro_f1']} | IT {summary['dev_IT_macro_f1']}"
        + (f" | sklearn {rep['sklearn_macro_f1']:.4f}" if "sklearn_macro_f1" in rep else ""))
    log(f"[dev] top confusions: {rep['top_confusions'][:6]}")

    prep_te = prepare(test_docs, tok, labels, cfg)
    if cfg.get("predict_test", True):
        zp = predict_test(model, test_docs, prep_te, cfg, labels, device, run_dir / "submission")
        summary.update(submission=str(zp))
        log(f"[submit] {zp} | {len(test_docs)} docs, {sum(p['n'] for p in prep_te.values())} sentences")

    if cfg.get("refit_full"):
        del model
        torch.cuda.empty_cache() if device == "cuda" else None
        prep_all = {**prep_tr, **prep_dev}
        rcfg = merge(cfg, {"train": {"epochs": max(1, best_epoch)}})
        seed_all(cfg["train"]["seed"])
        model, _, _ = new_model(rcfg, labels, prep_all, device)
        log(f"[refit] training on train+dev ({len(prep_all)} docs) for {rcfg['train']['epochs']} epochs (best dev epoch)")
        t0 = time.time()
        fit(model, prep_all, teacher, rcfg, labels, tok.pad_id, device, log, dev=None,
            save_path=run_dir / "refit.pt", hist_path=run_dir / "history_refit.json")
        zp = predict_test(model, test_docs, prep_te, rcfg, labels, device, run_dir / "submission_refit", tag="test_refit")
        summary.update(submission_refit=str(zp), refit_minutes=round((time.time() - t0) / 60, 1))
        log(f"[refit] {zp}")
    save_json(summary, run_dir / "summary.json")
    return summary, model

def predict_test(model, test_docs, prep_te, cfg, labels, device, out_dir, tag="test"):
    sc = predict_scores(model, prep_te, cfg, cfg["train"]["batch_size"] * 2, device)
    save_scores(to_logprobs(model, sc), Path(out_dir).parent / f"{tag}_logprobs.npz")
    return write_submission(decode_all(model, sc), test_docs, labels, out_dir)

def new_model(cfg, labels, prep_tr, device):
    model, info = build_model(cfg, labels)
    n_prompt = 0 if model.backbone.soft_prompt is None else model.backbone.soft_prompt.size(0)
    L = cfg["sentence"]["max_tokens"] if cfg["window"]["mode"] == "sentence" else cfg["window"]["max_len"]
    assert cfg["sentence"]["max_tokens"] <= L, "sentence.max_tokens must be <= window.max_len"
    assert L + n_prompt <= info["max_len"], f"max_len {L} (+{n_prompt} prompt) > backbone positions {info['max_len']}"
    if cfg["loss"]["class_weight"] == "sqrt_inv" and model.crf is None:
        cnt = np.bincount(np.concatenate([p["y"] for p in prep_tr.values()]), minlength=len(labels)).astype(np.float64)
        ref = np.median(cnt[cnt > 0])
        w = np.clip((ref / np.maximum(cnt, 1)) ** cfg["loss"]["cw_power"], 1.0 / cfg["loss"]["cw_max"], cfg["loss"]["cw_max"])
        w = w / (w * cnt).sum() * cnt.sum()
        model.class_w.copy_(torch.tensor(w, dtype=torch.float32))
    model.to(device)
    pc = count_params(model)
    info.update(params=pc, class_weights=[round(float(x), 3) for x in model.class_w.cpu()])
    return model, info, pc

def teacher_scores(cfg, labels, prep, device, run_dir, log):
    trun = Path(cfg["kd"]["teacher_run"])
    tcfg = json.load(open(trun / "config.json"))
    assert json.load(open(trun / "labels.json")) == labels, "teacher was trained with a different label list"
    cache = trun / "teacher_logprobs.pkl"
    scores = pickle.load(open(cache, "rb")) if cache.exists() else {}
    tsum = json.load(open(trun / "summary.json")) if (trun / "summary.json").exists() else {}
    todo = [d for d in prep if d not in scores]
    if todo:
        tm, tcfg, _ = load_run(trun, device)
        tok = get_tokenizer(tcfg)
        data = D.load_phase3(tcfg["data_dir"], cfg.get("max_docs"))
        docs = {**data["train"], **data["dev"]}
        tprep = prepare({d: docs[d] for d in todo}, tok, labels, tcfg)
        sc = to_logprobs(tm, predict_scores(tm, tprep, tcfg, 16, device))
        scores.update(sc)
        tmp = cache.with_suffix(f".{os.getpid()}.tmp")
        pickle.dump(scores, open(tmp, "wb")); os.replace(tmp, cache)
        del tm
        torch.cuda.empty_cache() if device == "cuda" else None
    info = {"teacher_run": str(trun), "teacher_backbone": Path(tcfg["backbone"]["ckpt"]).parent.name,
            "teacher_dev_macro_f1": tsum.get("dev", {}).get("macro_f1"), "teacher_params": tsum.get("params_total")}
    save_json(info, run_dir / "teacher.json")
    log(f"[kd] teacher {trun.name} ({info['teacher_backbone']}) | dev macro-F1 {info['teacher_dev_macro_f1']} | "
        f"scores for {len(prep)} docs ({len(todo)} computed, rest cached)")
    return {d: scores[d] for d in prep}, info

def make_optimizer(model, tc, device):
    n_layers = len(model.backbone.blocks)
    llrd = tc.get("llrd", 1.0)
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_new_param(n) or n.startswith("head.") or n.startswith("crf."):
            lr, depth = tc["lr_new"], "new"
        else:
            if ".blocks." in n:
                depth = int(n.split(".blocks.")[1].split(".")[0]) + 1
            elif "tok_emb" in n or "seg_emb" in n:
                depth = 0
            else:
                depth = n_layers + 1
            lr = tc["lr_backbone"] * llrd ** (n_layers + 1 - depth)
        wd = tc["weight_decay"] if p.dim() >= 2 and not n.startswith("crf.") else 0.0
        groups.setdefault((depth, wd), {"params": [], "lr": lr, "weight_decay": wd})["params"].append(p)
    return torch.optim.AdamW(list(groups.values()), fused=(device == "cuda"))

def epoch_batches(prep, cfg, bs, rng):
    items = []
    for d, p in prep.items():
        items += [(d, i, j) for i, j in doc_windows(p, cfg, train=True, rng=rng)]
    rng.shuffle(items)
    batches = []
    chunk = bs * 50
    for c in range(0, len(items), chunk):
        part = sorted(items[c:c + chunk], key=lambda x: int(prep[x[0]]["lens"][x[1]:x[2]].sum()))
        batches += [part[i:i + bs] for i in range(0, len(part), bs)]
    rng.shuffle(batches)
    return batches

def fit(model, prep_tr, teacher, cfg, labels, pad_id, device, log, dev=None, save_path=None, hist_path=None):
    tc = cfg["train"]
    opt = make_optimizer(model, tc, device)
    base_lrs = [g["lr"] for g in opt.param_groups]
    n_win = sum(len(doc_windows(p, cfg, train=True)) for p in prep_tr.values())
    steps_per_epoch = math.ceil(math.ceil(n_win / tc["batch_size"]) / tc["grad_accum"])
    total = steps_per_epoch * tc["epochs"]
    warm = max(1, int(tc["warmup_frac"] * total))
    dt = amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(dt == torch.float16))
    log(f"[train] ~{n_win} windows/epoch | {steps_per_epoch} steps/epoch | {total} steps | amp {dt}")
    dev_docs, prep_dev = dev if dev else ({}, {})
    trainable = [p for p in model.parameters() if p.requires_grad]
    hist, best, best_epoch, bad_epochs, step = [], -1.0, 0, 0, 0
    rng = random.Random(tc["seed"])
    kcfg = {**cfg["loss"], **cfg["kd"]}
    t0 = time.time()
    for ep in range(1, tc["epochs"] + 1):
        model.train()
        tot_loss, nb = 0.0, 0
        for batch in epoch_batches(prep_tr, cfg, tc["batch_size"], rng):
            ids, keep, sidx, y, smask, pos, dom, tch = to_dev(collate(batch, prep_tr, pad_id, teacher), device)
            with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
                em = model.emissions(ids, keep, sidx, smask, pos, dom)
            loss = model.loss(em, y, smask, kcfg, tch) / tc["grad_accum"]
            scaler.scale(loss).backward()
            tot_loss += loss.detach() * tc["grad_accum"]; nb += 1
            if nb % tc["grad_accum"] == 0:
                lr_mult = (step + 1) / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))
                for g, b in zip(opt.param_groups, base_lrs):
                    g["lr"] = b * lr_mult
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, tc["grad_clip"])
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                step += 1
        rec = {"epoch": ep, "train_loss": float(tot_loss) / max(nb, 1), "minutes": round((time.time() - t0) / 60, 2)}
        if dev_docs:
            rep, _, _ = evaluate(model, dev_docs, prep_dev, cfg, device)
            rec.update(dev_macro_f1=rep["all"]["macro_f1"], dev_macro_p=rep["all"]["macro_p"],
                       dev_macro_r=rep["all"]["macro_r"], dev_acc=rep["all"]["accuracy"],
                       dev_CL=rep.get("CL", {}).get("macro_f1", 0.0), dev_IT=rep.get("IT", {}).get("macro_f1", 0.0))
            if rep["all"]["macro_f1"] > best:
                best, best_epoch, bad_epochs = rep["all"]["macro_f1"], ep, 0
                torch.save(trainable_state(model), save_path); rec["saved"] = True
            else:
                bad_epochs += 1
        hist.append(rec)
        log("[epoch] " + " | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
        if hist_path:
            save_json(hist, hist_path)
        if dev_docs and bad_epochs >= tc["patience"]:
            log(f"[train] early stop: no dev improvement for {tc['patience']} epochs")
            break
    if not dev_docs:
        best_epoch = len(hist)
        torch.save(trainable_state(model), save_path)
    return hist, best, best_epoch
