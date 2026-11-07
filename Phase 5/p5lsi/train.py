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
from .metrics import (apply_thresholds, best_global_threshold, crossfit_strategies, fit_thresholds, format_report,
                      multilabel_report)
from .model import build_model, trainable_state
from .peft import count_params, is_new_param
from .tokenizer import BPETokenizer

DEFAULTS = {
    "run_name": "debug",
    "out_dir": "runs",
    "data_dir": "",
    "cache_dir": "",
    "ckpt_root": "",
    "backbone": {"ckpt": "", "attention": "bidirectional", "keep_layers": None, "c_segment": 0, "grad_ckpt": False},
    "chunks": {"len": 512, "k": 4, "strategy": "headtail", "overlap": 0.0},
    "method": {"name": "full"},
    "head": {"chunk_pool": "attn", "agg": "attn", "use_pos": True, "pos_buckets": 8, "dropout": 0.1,
             "hidden": 256, "att_hidden": 128},
    "loss": {"name": "bce", "pos_weight": "none", "pos_weight_max": 30.0, "gamma_pos": 0.0, "gamma_neg": 4.0,
             "asl_clip": 0.05},
    "kd": {"teacher_run": "", "kd_alpha": 0.5, "kd_T": 2.0},
    "train": {"epochs": 4, "docs_per_epoch": 10000, "batch_size": 8, "grad_accum": 1, "lr_backbone": 3e-5,
              "lr_new": 1e-3, "weight_decay": 0.01, "warmup_frac": 0.06, "patience": 2, "grad_clip": 1.0,
              "seed": 1337, "llrd": 1.0, "max_minutes": None, "eval_batch_size": 16, "chunk_bs": 32,
              "log_every": 0.2, "sampler": "uniform", "rare_power": 0.5, "eval_docs": 3000},
    "calibrate": {"strategy": "auto", "min_support": 3, "min_labels": 1},
    "predict_test": True,
    "max_docs": None,
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

def tok_path(cfg):
    return str(Path(cfg["backbone"]["ckpt"]).parent / "tokenizer.json")

def cache_dir(cfg):
    return Path(cfg.get("cache_dir") or Path(cfg.get("out_dir", "runs")).resolve().parent / "p5cache")

def load_data(cfg, log=print):
    data = D.load_cached(cfg["data_dir"], tok_path(cfg), cache_dir(cfg), cfg.get("max_docs"), cfg.get("n_proc", 4), log)
    data["space"] = D.label_space(data)
    for split in ("train", "dev"):
        data[split]["Y"] = D.multi_hot(data[split]["labels"], data["space"])
    return data

def subset(split, idx):
    out = {}
    for k, v in split.items():
        if k in ("ids", "raw_ids", "tokens", "labels") and v is not None:
            out[k] = [v[i] for i in idx]
        elif k in ("Y", "n_chars") and v is not None:
            out[k] = v[idx]
        else:
            out[k] = v
    return out

def select_chunks(n, L, K, strategy="headtail", overlap=0.0):
    if n <= L:
        return [(0, n)]
    if n <= K * L:
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

def chunk_pos_bucket(s, e, n, nb):
    return min(nb - 1, int(nb * ((s + e) / 2) / max(1, n)))

def collate(doc_tokens, ccfg, nb, pad_id, chunk_override=None):
    cc = {**ccfg, **(chunk_override or {})}
    spans = [select_chunks(len(t), cc["len"], cc["k"], cc["strategy"], cc.get("overlap", 0.0)) for t in doc_tokens]
    N = sum(len(s) for s in spans)
    T = max(e - s for sp in spans for s, e in sp)
    ids = torch.full((N, T), pad_id, dtype=torch.long)
    keep = torch.zeros((N, T), dtype=torch.bool)
    doc = torch.zeros(N, dtype=torch.long); slot = torch.zeros(N, dtype=torch.long); cpos = torch.zeros(N, dtype=torch.long)
    r = 0
    for b, (tk, sp) in enumerate(zip(doc_tokens, spans)):
        for k, (s, e) in enumerate(sp):
            ids[r, :e - s] = torch.from_numpy(tk[s:e].astype(np.int64))
            keep[r, :e - s] = True
            doc[r], slot[r], cpos[r] = b, k, chunk_pos_bucket(s, e, len(tk), nb)
            r += 1
    return ids, keep, doc, slot, cpos

def amp_dtype(device):
    if device != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

def to_dev(xs, device):
    return [x.to(device, non_blocking=True) for x in xs]

@torch.no_grad()
def predict_logits(model, tokens, cfg, device="cuda", chunk_override=None, return_weights=False):
    model.eval()
    dt = amp_dtype(device)
    tc = cfg["train"]
    nb = cfg["head"].get("pos_buckets", 8)
    order = np.argsort([len(t) for t in tokens], kind="stable")
    out = np.zeros((len(tokens), model.n_labels), dtype=np.float32)
    weights = [None] * len(tokens)
    bs = tc.get("eval_batch_size", 16)
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        ids, keep, doc, slot, cpos = collate([tokens[i] for i in idx], cfg["chunks"], nb, pad_id=0, chunk_override=chunk_override)
        ids, keep, doc_d, slot_d, cpos_d = to_dev((ids, keep, doc, slot, cpos), device)
        with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
            lg, w = model(ids, keep, doc_d, slot_d, cpos_d, len(idx), chunk_bs=tc.get("chunk_bs", 32))
        out[idx] = lg.float().cpu().numpy()
        if return_weights:
            for b, i in enumerate(idx):
                sel = doc == b
                ww = None if w is None else w[b, :int(sel.sum())].float().cpu().numpy()
                weights[i] = (ww, cpos[sel].numpy())
    return (out, weights) if return_weights else out

def probs_from_logits(lg):
    return torch.sigmoid(torch.from_numpy(np.asarray(lg)).float()).numpy()

def evaluate(model, split, cfg, device, chunk_override=None):
    lg = predict_logits(model, split["tokens"], cfg, device, chunk_override)
    prob = probs_from_logits(lg)
    ml = cfg["calibrate"].get("min_labels", 1)
    t, _ = best_global_threshold(split["Y"], prob, ml)
    return (multilabel_report(split["Y"], apply_thresholds(prob, t, ml)), t,
            multilabel_report(split["Y"], apply_thresholds(prob, 0.5, ml)), lg, prob)

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2, default=str)

def load_run(run_dir, device="cuda", weights="best.pt"):
    run_dir = Path(run_dir)
    cfg = json.load(open(run_dir / "config.json"))
    space = json.load(open(run_dir / "label_space.json"))
    model, _ = build_model(cfg, len(space))
    sd = torch.load(run_dir / weights, map_location="cpu")
    _, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    return model.to(device), cfg

def check_prediction(obj, raw_ids, space):
    assert isinstance(obj, list), "prediction.json must be a JSON list"
    got = [r["id"] for r in obj]
    assert len(got) == len(raw_ids), f"{len(got)} predictions for {len(raw_ids)} test documents"
    assert len({json.dumps(g) for g in got}) == len(got), "duplicate ids in the submission"
    assert {json.dumps(g) for g in got} == {json.dumps(g) for g in raw_ids}, "submission ids differ from test.json ids"
    valid = set(int(x) for x in space)
    for r in obj:
        assert set(r) == {"id", "pred"}, f"unexpected keys {sorted(r)}"
        p = r["pred"]
        assert isinstance(p, list), f"pred must be a list of statute ids, got {p!r}"
        assert all(type(x) is int for x in p), f"pred must contain integers only, got {p!r}"
        assert len(set(p)) == len(p), f"duplicate statute ids in {p!r}"
        assert set(p) <= valid, f"statute ids outside the label space: {sorted(set(p) - valid)}"

def preds_to_ids(P, space):
    sp = [int(x) for x in space]
    return [[sp[k] for k in np.flatnonzero(row)] for row in np.asarray(P, dtype=bool)]

def write_submission(raw_ids, pred_lists, out_dir, space):
    import zipfile
    obj = [{"id": rid, "pred": [int(x) for x in p]} for rid, p in zip(raw_ids, pred_lists)]
    check_prediction(obj, raw_ids, space)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pj = out_dir / "prediction.json"
    json.dump(obj, open(pj, "w", encoding="utf-8"))
    zp = out_dir / "submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"]
        check_prediction(json.loads(z.read("prediction.json")), raw_ids, space)
    return zp

def save_probs(path, ids, logits):
    np.savez_compressed(path, ids=np.array(ids, dtype=object), logits=logits.astype(np.float32))

def load_probs(path):
    z = np.load(path, allow_pickle=True)
    return list(z["ids"]), z["logits"]

def calibrate(Y, prob, ccfg, seed=0):
    ml, ms = ccfg.get("min_labels", 1), ccfg.get("min_support", 3)
    cf = crossfit_strategies(Y, prob, min_labels=ml, min_support=ms, seed=seed)
    strat = ccfg.get("strategy", "auto")
    if strat == "auto":
        strat = max(cf, key=lambda k: (cf[k], -["fixed", "global", "per_label"].index(k)))
    return fit_thresholds(Y, prob, strat, ml, ms), strat, cf

def thr_to_json(thr, space):
    if np.ndim(thr) == 0:
        return float(thr)
    return {str(int(s)): round(float(t), 4) for s, t in zip(space, thr)}

def thr_from_json(obj, space):
    if isinstance(obj, (int, float)):
        return float(obj)
    return np.array([obj[str(int(s))] for s in space], dtype=np.float64)

def run(cfg, data=None, device=None, log=print):
    cfg = resolve_paths(merge(DEFAULTS, cfg))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(cfg["train"]["seed"])
    run_dir = Path(cfg["out_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, run_dir / "config.json")

    data = data or load_data(cfg, log)
    tr, dev, te, space = data["train"], data["dev"], data["test"], data["space"]
    save_json([int(x) for x in space], run_dir / "label_space.json")
    log(f"[data] train {len(tr['ids'])} | dev {len(dev['ids'])} | test {len(te['ids'])} documents | "
        f"{len(space)} statutes (ids {space[0]}..{space[-1]}) | statutes/doc train {tr['Y'].sum(1).mean():.2f}")
    pad_id = BPETokenizer.load(tok_path(cfg)).pad_id

    teacher, t_info = None, None
    schedule = doc_schedule(len(tr["ids"]), cfg["train"], tr["Y"])
    if cfg["kd"].get("teacher_run"):
        need = sorted(set(int(i) for ep in schedule for i in ep))
        teacher, t_info = teacher_logits(cfg, tr, need, device, run_dir, space, log)

    model, info, pc = new_model(cfg, tr, len(space), device)
    save_json(info, run_dir / "model_info.json")
    log(f"[model] arch {info['arch']} ({info['class']}) | layers {info['layers_used']}/{info['phase1_layers']} | "
        f"attention {info['attention']} | method {cfg['method']['name']} | chunks {cfg['chunks']['k']}x{cfg['chunks']['len']} "
        f"{cfg['chunks']['strategy']} | head {cfg['head']['chunk_pool']}/{cfg['head']['agg']} | loss {cfg['loss']['name']}")
    log(f"[model] params total {pc['total'] / 1e6:.2f}M | trainable {pc['trainable'] / 1e6:.3f}M ({pc['trainable_pct']}%)"
        + (f" | 4-bit NF4 weights {pc['nf4_weights'] / 1e6:.2f}M" if pc.get("nf4_weights") else ""))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_train = time.time()
    hist, best, best_epoch = fit(model, tr, dev, schedule, teacher, cfg, pad_id, device, log,
                                 save_path=run_dir / "best.pt", hist_path=run_dir / "history.json")
    train_min = (time.time() - t_train) / 60
    peak_mem = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device), strict=False)
    t0 = time.time()
    lg_dev = predict_logits(model, dev["tokens"], cfg, device)
    dt_inf = time.time() - t0
    p_dev = probs_from_logits(lg_dev)
    ccfg, ml = cfg["calibrate"], cfg["calibrate"].get("min_labels", 1)
    thr, strat, cf = calibrate(dev["Y"], p_dev, ccfg, cfg["train"]["seed"])
    rep05 = multilabel_report(dev["Y"], apply_thresholds(p_dev, 0.5, ml), space)
    P_dev = apply_thresholds(p_dev, thr, ml)
    rep = multilabel_report(dev["Y"], P_dev, space)
    save_probs(run_dir / "dev_logits.npz", dev["ids"], lg_dev)
    save_json({"at_0.5": rep05, "calibrated": rep, "strategy": strat, "crossfit_macro_f1": cf,
               "thresholds": thr_to_json(thr, space)}, run_dir / "dev_report.json")
    save_json([{"id": i, "gold": g, "pred": p, "n_tokens": int(len(t))}
               for i, g, p, t in zip(dev["ids"], dev["labels"], preds_to_ids(P_dev, space), dev["tokens"])],
              run_dir / "dev_predictions.json")
    log("\n" + format_report(rep05, "dev, threshold 0.5"))
    log(format_report(rep, f"dev, thresholds '{strat}' (fitted on all of dev)"))
    log(f"[dev] cross-fitted macro-F1 (thresholds fitted on the other half of dev): "
        + " | ".join(f"{k} {v:.4f}" for k, v in cf.items()) + f" -> strategy '{strat}'")
    log(f"[dev] MACRO-F1 {rep['macro_f1']:.4f} (in-sample, {strat}) | {cf[strat]:.4f} (cross-fitted) | "
        f"micro-F1 {rep['micro_f1']:.4f} | subset acc {rep['subset_acc']:.4f}")
    keys = ("macro_f1", "macro_p", "macro_r", "macro_f1_present", "micro_f1", "subset_acc", "pred_per_doc")
    summary = {"run": cfg["run_name"], "arch": info["arch"], "backbone": Path(cfg["backbone"]["ckpt"]).parent.name,
               "method": cfg["method"], "chunks": cfg["chunks"], "head": cfg["head"], "loss": cfg["loss"],
               "sampler": cfg["train"].get("sampler", "uniform"), "attention": info["attention"],
               "layers": info["layers_used"], "params_total": pc["total"], "params_trainable": pc["trainable"],
               "trainable_pct": pc["trainable_pct"], "nf4_weights": pc.get("nf4_weights", 0),
               "train_minutes": round(train_min, 1), "peak_train_mem_gb": peak_mem,
               "best_epoch": best_epoch, "epochs_run": len(hist), "kd_teacher": t_info, "n_labels": len(space),
               "threshold_strategy": strat, "crossfit_macro_f1": cf,
               "dev_at_0.5": {k: rep05[k] for k in keys}, "dev_calibrated": {k: rep[k] for k in keys},
               "infer_docs_per_sec": round(len(dev["ids"]) / max(dt_inf, 1e-6), 2)}

    if cfg.get("predict_test", True):
        lg_te = predict_logits(model, te["tokens"], cfg, device)
        save_probs(run_dir / "test_logits.npz", te["ids"], lg_te)
        P_te = apply_thresholds(probs_from_logits(lg_te), thr, ml)
        zp = write_submission(te["raw_ids"], preds_to_ids(P_te, space), run_dir / "submission", space)
        summary.update(submission=str(zp), test_pred_per_doc=round(float(P_te.sum(1).mean()), 3))
        log(f"[submit] {zp} | {len(P_te)} documents | statutes/doc {P_te.sum(1).mean():.2f} (thresholds '{strat}')")
    save_json(summary, run_dir / "summary.json")
    return summary, model

def rare_weights(Y, power=0.5):
    freq = np.maximum(Y.sum(0).astype(np.float64), 1)
    w = np.where(Y > 0, freq[None, :] ** -power, 0).max(1)
    w[w == 0] = w[w > 0].min() if (w > 0).any() else 1.0
    return w / w.sum()

def doc_schedule(n, tc, Y=None):
    rng = np.random.default_rng(tc["seed"])
    dpe = min(n, tc.get("docs_per_epoch") or n)
    if tc.get("sampler", "uniform") == "rare":
        assert Y is not None
        w = rare_weights(Y, tc.get("rare_power", 0.5))
        return [rng.choice(n, dpe, replace=False, p=w) for _ in range(tc["epochs"])]
    stream, out = np.array([], dtype=np.int64), []
    for _ in range(tc["epochs"]):
        while len(stream) < dpe:
            stream = np.concatenate([stream, rng.permutation(n)])
        out.append(stream[:dpe]); stream = stream[dpe:]
    return out

def new_model(cfg, tr, n_labels, device):
    model, info = build_model(cfg, n_labels)
    assert cfg["chunks"]["len"] <= info["max_len"], f"chunk length {cfg['chunks']['len']} > backbone positions {info['max_len']}"
    pos = tr["Y"].sum(0).astype(np.float64)
    model.head.init_bias(pos / max(1, len(tr["Y"])))
    if cfg["loss"].get("pos_weight", "none") == "sqrt":
        pw = np.clip(np.sqrt((len(tr["Y"]) - pos) / np.maximum(pos, 1)), 1.0, cfg["loss"].get("pos_weight_max", 30.0))
        model.pos_weight.copy_(torch.tensor(pw, dtype=torch.float32))
    model.to(device)
    pc = count_params(model)
    info.update(params=pc, n_labels=n_labels)
    return model, info, pc

def teacher_logits(cfg, tr, need, device, run_dir, space, log):
    trun = Path(cfg["kd"]["teacher_run"])
    tcfg = json.load(open(trun / "config.json"))
    assert json.load(open(trun / "label_space.json")) == [int(x) for x in space], "teacher and student label spaces differ"
    cache = trun / "teacher_train_logits.pkl"
    scores = pickle.load(open(cache, "rb")) if cache.exists() else {}
    tsum = json.load(open(trun / "summary.json")) if (trun / "summary.json").exists() else {}
    todo = [i for i in need if tr["ids"][i] not in scores]
    if todo:
        t0 = time.time()
        tm, tcfg = load_run(trun, device)
        assert tok_path(tcfg) and json.load(open(tok_path(tcfg)))["merges"] == json.load(open(tok_path(cfg)))["merges"], \
            "teacher and student must share the Phase-1 tokenizer (token cache is reused)"
        lg = predict_logits(tm, [tr["tokens"][i] for i in todo], tcfg, device)
        scores.update({tr["ids"][i]: lg[k] for k, i in enumerate(todo)})
        tmp = cache.with_suffix(f".{os.getpid()}.tmp")
        pickle.dump(scores, open(tmp, "wb")); os.replace(tmp, cache)
        del tm
        torch.cuda.empty_cache() if device == "cuda" else None
        log(f"[kd] teacher logits for {len(todo)} documents in {time.time() - t0:.0f}s")
    out = np.zeros((len(tr["ids"]), len(space)), dtype=np.float32)
    for i in need:
        out[i] = scores[tr["ids"][i]]
    tdev = json.load(open(trun / "dev_report.json")) if (trun / "dev_report.json").exists() else {}
    thr = thr_from_json(tdev["thresholds"], space) if "thresholds" in tdev else 0.5
    mf = multilabel_report(tr["Y"][need], apply_thresholds(probs_from_logits(out[need]), thr))["macro_f1"]
    info = {"teacher_run": str(trun), "teacher_backbone": Path(tcfg["backbone"]["ckpt"]).parent.name,
            "teacher_dev_macro_f1": tsum.get("dev_calibrated", {}).get("macro_f1"), "teacher_params": tsum.get("params_total"),
            "teacher_train_macro_f1_on_student_docs": round(mf, 4)}
    save_json(info, run_dir / "teacher.json")
    log(f"[kd] teacher {trun.name} ({info['teacher_backbone']}) | dev macro-F1 {info['teacher_dev_macro_f1']} | "
        f"train macro-F1 on the student's {len(need)} documents {mf:.3f}")
    return out, info

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

def epoch_batches(idx, n_chunks, bs, rng):
    items = list(map(int, idx))
    rng.shuffle(items)
    batches, block = [], bs * 50
    for c in range(0, len(items), block):
        part = sorted(items[c:c + block], key=lambda i: n_chunks[i])
        batches += [part[i:i + bs] for i in range(0, len(part), bs)]
    rng.shuffle(batches)
    return batches

def fit(model, tr, dev, schedule, teacher, cfg, pad_id, device, log, save_path, hist_path):
    tc, cc = cfg["train"], cfg["chunks"]
    nb = cfg["head"].get("pos_buckets", 8)
    n_chunks = [len(select_chunks(len(t), cc["len"], cc["k"], cc["strategy"], cc.get("overlap", 0.0))) for t in tr["tokens"]]
    opt = make_optimizer(model, tc, device)
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = math.ceil(math.ceil(len(schedule[0]) / tc["batch_size"]) / tc["grad_accum"])
    total = steps_per_epoch * len(schedule)
    warm = max(1, int(tc["warmup_frac"] * total))
    dt = amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(dt == torch.float16))
    log(f"[train] {len(schedule)} epochs x {len(schedule[0])} documents | ~{np.mean(n_chunks):.2f} chunks/doc | "
        f"{steps_per_epoch} steps/epoch | {total} steps | amp {dt} | time limit {tc.get('max_minutes')} min")
    trainable = [p for p in model.parameters() if p.requires_grad]
    hist, best, best_epoch, bad, step = [], -1.0, 0, 0, 0
    rng = random.Random(tc["seed"])
    kcfg = {**cfg["loss"], **cfg["kd"]}
    y_all = torch.from_numpy(tr["Y"])
    ne = tc.get("eval_docs")
    dev_sel = dev if not ne or ne >= len(dev["ids"]) else \
        subset(dev, np.sort(np.random.default_rng(tc["seed"]).choice(len(dev["ids"]), ne, replace=False)))
    t0 = time.time()
    limit = tc.get("max_minutes")
    stop = False
    for ep, idx in enumerate(schedule, 1):
        model.train()
        tot, nbat = 0.0, 0
        batches = epoch_batches(idx, n_chunks, tc["batch_size"], rng)
        every = max(1, int(len(batches) * tc.get("log_every", 0.2)))
        for bi, batch in enumerate(batches):
            ids, keep, doc, slot, cpos = to_dev(collate([tr["tokens"][i] for i in batch], cc, nb, pad_id), device)
            y = y_all[batch].to(device)
            tch = None if teacher is None else torch.from_numpy(teacher[batch]).to(device)
            with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
                lg, _ = model(ids, keep, doc, slot, cpos, len(batch))
            loss = model.loss(lg, y, kcfg, tch) / tc["grad_accum"]
            scaler.scale(loss).backward()
            tot += float(loss.detach()) * tc["grad_accum"]; nbat += 1
            if nbat % tc["grad_accum"] == 0 or bi == len(batches) - 1:
                lr_mult = (step + 1) / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))
                for g, b in zip(opt.param_groups, base_lrs):
                    g["lr"] = b * lr_mult
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, tc["grad_clip"])
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                step += 1
            if (bi + 1) % every == 0:
                el = (time.time() - t0) / 60
                done = (ep - 1) + (bi + 1) / len(batches)
                log(f"[progress] epoch {ep} {100 * (bi + 1) / len(batches):3.0f}% | loss {tot / nbat:.4f} | {el:.1f} min | "
                    f"ETA all epochs {el / done * (len(schedule) - done):.1f} min")
            if limit and (time.time() - t0) / 60 > limit:
                log(f"[train] time limit {limit} min reached in epoch {ep} ({100 * (bi + 1) / len(batches):.0f}%): evaluating and stopping")
                stop = True
                break
        rec = {"epoch": ep, "train_loss": tot / max(nbat, 1), "minutes": round((time.time() - t0) / 60, 2)}
        rep, t, rep05, _, _ = evaluate(model, dev_sel, cfg, device)
        rec.update(dev_macro_f1=rep["macro_f1"], thr=t, dev_micro_f1=rep["micro_f1"], dev_subset_acc=rep["subset_acc"],
                   dev_macro_f1_at_05=rep05["macro_f1"], dev_pred_per_doc=rep["pred_per_doc"], dev_docs=rep["n_docs"])
        if rep["macro_f1"] > best:
            best, best_epoch, bad = rep["macro_f1"], ep, 0
            torch.save(trainable_state(model), save_path); rec["saved"] = True
        else:
            bad += 1
        hist.append(rec)
        log("[epoch] " + " | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
        save_json(hist, hist_path)
        if stop:
            break
        if bad >= tc["patience"]:
            log(f"[train] early stop: no dev improvement for {tc['patience']} epochs")
            break
    return hist, best, best_epoch
