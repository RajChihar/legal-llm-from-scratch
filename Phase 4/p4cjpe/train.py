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
from .metrics import best_threshold, binary_report, format_report
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
    "loss": {"class_weight": "balanced", "label_smoothing": 0.0},
    "kd": {"teacher_run": "", "kd_alpha": 0.5, "kd_T": 2.0},
    "train": {"epochs": 3, "docs_per_epoch": 8000, "batch_size": 8, "grad_accum": 1, "lr_backbone": 3e-5,
              "lr_new": 1e-3, "weight_decay": 0.01, "warmup_frac": 0.06, "patience": 2, "grad_clip": 1.0,
              "seed": 1337, "llrd": 1.0, "max_minutes": None, "eval_batch_size": 16, "chunk_bs": 32,
              "log_every": 0.2},
    "calibrate": True,
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
    return Path(cfg.get("cache_dir") or Path(cfg.get("out_dir", "runs")).resolve().parent / "p4cache")

def load_data(cfg, log=print):
    return D.load_cached(cfg["data_dir"], tok_path(cfg), cache_dir(cfg), cfg.get("max_docs"), cfg.get("n_proc", 4), log)

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
    out = np.zeros((len(tokens), 2), dtype=np.float32)
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
    return torch.softmax(torch.from_numpy(lg).float(), -1)[:, 1].numpy()

def evaluate(model, split, cfg, device, chunk_override=None):
    lg = predict_logits(model, split["tokens"], cfg, device, chunk_override)
    p1 = probs_from_logits(lg)
    rep = binary_report(split["labels"], (lg[:, 1] > lg[:, 0]).astype(int))
    return rep, lg, p1

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2, default=str)

def load_run(run_dir, device="cuda", weights="best.pt"):
    run_dir = Path(run_dir)
    cfg = json.load(open(run_dir / "config.json"))
    model, _ = build_model(cfg)
    sd = torch.load(run_dir / weights, map_location="cpu")
    _, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    return model.to(device), cfg

def check_prediction(obj, raw_ids):
    assert isinstance(obj, list), "prediction.json must be a JSON list"
    got = [r["id"] for r in obj]
    assert len(got) == len(raw_ids), f"{len(got)} predictions for {len(raw_ids)} test cases"
    assert len({json.dumps(g) for g in got}) == len(got), "duplicate ids in the submission"
    assert {json.dumps(g) for g in got} == {json.dumps(g) for g in raw_ids}, "submission ids differ from test.json ids"
    for r in obj:
        assert set(r) == {"id", "pred"}, f"unexpected keys {sorted(r)}"
        assert type(r["pred"]) is int and r["pred"] in (0, 1), f"pred must be the integer 0 or 1, got {r['pred']!r}"

def write_submission(raw_ids, preds, out_dir):
    import zipfile
    obj = [{"id": rid, "pred": int(p)} for rid, p in zip(raw_ids, preds)]
    check_prediction(obj, raw_ids)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pj = out_dir / "prediction.json"
    json.dump(obj, open(pj, "w", encoding="utf-8"))
    zp = out_dir / "submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"]
        check_prediction(json.loads(z.read("prediction.json")), raw_ids)
    return zp

def save_probs(path, ids, logits):
    np.savez_compressed(path, ids=np.array(ids, dtype=object), logits=logits.astype(np.float32))

def load_probs(path):
    z = np.load(path, allow_pickle=True)
    return list(z["ids"]), z["logits"]

def run(cfg, data=None, device=None, log=print):
    cfg = resolve_paths(merge(DEFAULTS, cfg))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(cfg["train"]["seed"])
    run_dir = Path(cfg["out_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, run_dir / "config.json")

    data = data or load_data(cfg, log)
    tr, dev, te = data["train"], data["dev"], data["test"]
    log(f"[data] train {len(tr['ids'])} | dev {len(dev['ids'])} | test {len(te['ids'])} cases | "
        f"train label-1 share {tr['labels'].mean():.3f}")
    pad_id = BPETokenizer.load(tok_path(cfg)).pad_id

    teacher, t_info = None, None
    schedule = doc_schedule(len(tr["ids"]), cfg["train"])
    if cfg["kd"].get("teacher_run"):
        need = sorted(set(int(i) for ep in schedule for i in ep))
        teacher, t_info = teacher_logits(cfg, tr, need, device, run_dir, log)

    model, info, pc = new_model(cfg, tr, device)
    save_json(info, run_dir / "model_info.json")
    log(f"[model] arch {info['arch']} ({info['class']}) | layers {info['layers_used']}/{info['phase1_layers']} | "
        f"attention {info['attention']} | method {cfg['method']['name']} | chunks {cfg['chunks']['k']}x{cfg['chunks']['len']} "
        f"{cfg['chunks']['strategy']} | pool {cfg['head']['chunk_pool']}/{cfg['head']['agg']}")
    log(f"[model] params total {pc['total'] / 1e6:.2f}M | trainable {pc['trainable'] / 1e6:.3f}M ({pc['trainable_pct']}%) | "
        f"class weights {info['class_weights']}")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_train = time.time()
    hist, best, best_epoch = fit(model, tr, dev, schedule, teacher, cfg, pad_id, device, log,
                                 save_path=run_dir / "best.pt", hist_path=run_dir / "history.json")
    train_min = (time.time() - t_train) / 60
    peak_mem = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device), strict=False)
    t0 = time.time()
    rep, lg_dev, p_dev = evaluate(model, dev, cfg, device)
    dt_inf = time.time() - t0
    thr, f_thr = best_threshold(dev["labels"], p_dev) if cfg.get("calibrate", True) else (0.5, rep["macro_f1"])
    rep_thr = binary_report(dev["labels"], (p_dev >= thr).astype(int))
    save_probs(run_dir / "dev_logits.npz", dev["ids"], lg_dev)
    save_json({"argmax": rep, "calibrated": rep_thr, "threshold": thr}, run_dir / "dev_report.json")
    save_json([{"id": i, "gold": int(g), "prob1": round(float(p), 5), "pred": int(p >= thr), "n_tokens": int(len(t))}
               for i, g, p, t in zip(dev["ids"], dev["labels"], p_dev, dev["tokens"])], run_dir / "dev_predictions.json")
    log("\n" + format_report(rep, "dev, argmax (threshold 0.5)"))
    log(format_report(rep_thr, f"dev, calibrated threshold {thr:.2f}"))
    log(f"[dev] MACRO-F1 {rep['macro_f1']:.4f} (argmax) | {rep_thr['macro_f1']:.4f} (threshold {thr:.2f}) | "
        f"P {rep_thr['macro_p']:.4f} R {rep_thr['macro_r']:.4f} | acc {rep_thr['accuracy']:.4f}")
    summary = {"run": cfg["run_name"], "arch": info["arch"], "backbone": Path(cfg["backbone"]["ckpt"]).parent.name,
               "method": cfg["method"], "chunks": cfg["chunks"], "head": cfg["head"], "attention": info["attention"],
               "layers": info["layers_used"], "params_total": pc["total"], "params_trainable": pc["trainable"],
               "trainable_pct": pc["trainable_pct"], "train_minutes": round(train_min, 1), "peak_train_mem_gb": peak_mem,
               "best_epoch": best_epoch, "epochs_run": len(hist), "kd_teacher": t_info, "threshold": thr,
               "dev": {k: rep[k] for k in ("macro_f1", "macro_p", "macro_r", "accuracy")},
               "dev_calibrated": {k: rep_thr[k] for k in ("macro_f1", "macro_p", "macro_r", "accuracy")},
               "infer_docs_per_sec": round(len(dev["ids"]) / max(dt_inf, 1e-6), 2)}

    if cfg.get("predict_test", True):
        lg_te = predict_logits(model, te["tokens"], cfg, device)
        save_probs(run_dir / "test_logits.npz", te["ids"], lg_te)
        pred = (probs_from_logits(lg_te) >= thr).astype(int)
        zp = write_submission(te["raw_ids"], pred, run_dir / "submission")
        summary.update(submission=str(zp), test_share_pred_1=round(float(pred.mean()), 3))
        log(f"[submit] {zp} | {len(pred)} cases | predicted label-1 share {pred.mean():.3f} (threshold {thr:.2f})")
    save_json(summary, run_dir / "summary.json")
    return summary, model

def doc_schedule(n, tc):
    rng = np.random.default_rng(tc["seed"])
    dpe = min(n, tc.get("docs_per_epoch") or n)
    stream, out = np.array([], dtype=np.int64), []
    for _ in range(tc["epochs"]):
        while len(stream) < dpe:
            stream = np.concatenate([stream, rng.permutation(n)])
        out.append(stream[:dpe]); stream = stream[dpe:]
    return out

def new_model(cfg, tr, device):
    model, info = build_model(cfg)
    assert cfg["chunks"]["len"] <= info["max_len"], f"chunk length {cfg['chunks']['len']} > backbone positions {info['max_len']}"
    if cfg["loss"]["class_weight"] == "balanced":
        cnt = np.bincount(tr["labels"], minlength=2).astype(np.float64)
        w = cnt.sum() / (2 * np.maximum(cnt, 1))
        model.class_w.copy_(torch.tensor(w, dtype=torch.float32))
    model.to(device)
    pc = count_params(model)
    info.update(params=pc, class_weights=[round(float(x), 3) for x in model.class_w.cpu()])
    return model, info, pc

def teacher_logits(cfg, tr, need, device, run_dir, log):
    trun = Path(cfg["kd"]["teacher_run"])
    tcfg = json.load(open(trun / "config.json"))
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
        log(f"[kd] teacher logits for {len(todo)} cases in {time.time() - t0:.0f}s")
    out = np.zeros((len(tr["ids"]), 2), dtype=np.float32)
    for i in need:
        out[i] = scores[tr["ids"][i]]
    acc = float(((out[need, 1] > out[need, 0]).astype(int) == tr["labels"][need]).mean())
    info = {"teacher_run": str(trun), "teacher_backbone": Path(tcfg["backbone"]["ckpt"]).parent.name,
            "teacher_dev_macro_f1": tsum.get("dev", {}).get("macro_f1"), "teacher_params": tsum.get("params_total"),
            "teacher_train_acc_on_student_cases": round(acc, 4)}
    save_json(info, run_dir / "teacher.json")
    log(f"[kd] teacher {trun.name} ({info['teacher_backbone']}) | dev macro-F1 {info['teacher_dev_macro_f1']} | "
        f"train accuracy on the student's {len(need)} cases {acc:.3f}")
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
    log(f"[train] {len(schedule)} epochs x {len(schedule[0])} cases | ~{np.mean(n_chunks):.2f} chunks/case | "
        f"{steps_per_epoch} steps/epoch | {total} steps | amp {dt} | time limit {tc.get('max_minutes')} min")
    trainable = [p for p in model.parameters() if p.requires_grad]
    hist, best, best_epoch, bad, step = [], -1.0, 0, 0, 0
    rng = random.Random(tc["seed"])
    kcfg = {**cfg["loss"], **cfg["kd"]}
    y_all = torch.from_numpy(tr["labels"])
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
        rep, _, p1 = evaluate(model, dev, cfg, device)
        rec.update(dev_macro_f1=rep["macro_f1"], dev_macro_p=rep["macro_p"], dev_macro_r=rep["macro_r"],
                   dev_acc=rep["accuracy"], dev_pred1_share=float((p1 >= 0.5).mean()))
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
