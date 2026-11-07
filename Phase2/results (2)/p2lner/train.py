"""Training, sliding-window inference, evaluation and run management.

Long documents are cut into windows of `max_len` tokens with stride `stride` (overlap = max_len - stride).
At inference every token gets a weighted average of its scores over all windows that contain it:
    bidirectional: weight = distance to the nearer window edge (centre tokens have context on both sides)
    causal       : weight = position inside the window (later tokens have seen more left context)
"""
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from .metrics import error_analysis, format_table, strict_scores
from .model import build_model, trainable_state
from .peft import count_params, is_new_param
from .spans import TagSet, decode_doc, learn_trim_rules, reachability, spans_to_tags, trim_spans
from .tokenizer import BPETokenizer


# ------------------------------------------------------------------------------------------- config
DEFAULTS = {
    "run_name": "debug",
    "out_dir": "runs",
    "data_dir": "",
    "backbone": {"ckpt": "", "attention": "bidirectional", "keep_layers": None, "c_segment": 0, "grad_ckpt": False},
    "tokenizer": {"path": "", "fine_punct": True},
    "window": {"max_len": 512, "stride": 384, "random_offset": True},
    "method": {"name": "full"},
    "head": {"type": "linear", "dropout": 0.1, "hidden": 0},
    "loss": {"class_weight": "none", "cw_power": 0.5, "cw_max": 10.0, "label_smoothing": 0.0, "focal_gamma": 0},
    "kd": {"teacher_run": "", "kd_alpha": 0.5, "kd_T": 2.0},
    "decode": {"strategy": "bio_viterbi", "trim": True},
    "split": {"mode": "random", "dev_frac": 0.15, "seed": 1337},
    "train": {"epochs": 10, "batch_size": 8, "grad_accum": 1, "lr_backbone": 5e-5, "lr_new": 1e-3,
              "weight_decay": 0.01, "warmup_frac": 0.1, "patience": 3, "grad_clip": 1.0, "seed": 1337,
              "eval_every_epochs": 1, "num_workers": 0, "llrd": 1.0},
    "predict_test": True,
    "refit_full": False,      # after training, retrain on train+dev for best_epoch epochs -> submission_refit/
    "cache_dir": "",          # tokenisation cache (default: <out_dir>/../p2cache)
    "ckpt_root": "",          # extra folder searched for "@A"/"@B"/... checkpoints
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
    """--set train.lr_new=5e-4 style override (value parsed as JSON when possible)."""
    try:
        value = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        pass
    d = cfg
    ks = dotted.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = value


# Phase-1 run folders (as written by the Phase-1 notebooks); configs refer to them as "@A", "@C", ...
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


def resolve_paths(cfg):
    roots = [cfg["ckpt_root"]] if cfg.get("ckpt_root") else []
    cfg["backbone"]["ckpt"] = resolve_ckpt(cfg["backbone"]["ckpt"], roots)
    return cfg


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def get_tokenizer(cfg):
    p = cfg["tokenizer"].get("path") or str(Path(cfg["backbone"]["ckpt"]).parent / "tokenizer.json")
    tok = BPETokenizer.load(p, fine_punct=cfg["tokenizer"].get("fine_punct", True))
    cache = cfg.get("cache_dir") or str(Path(cfg.get("out_dir", "runs")).resolve().parent / "p2cache")
    try:
        return tok.enable_disk_cache(cache)
    except OSError:                                    # read-only location: run without the cache
        return tok


# ------------------------------------------------------------------------------------------- data prep
def prepare(docs, tok, tagset):
    """{id: {"text", "gold"?}} -> {id: {"enc", "tags"?}} (tags = BIO ids per token)."""
    out = {}
    for did, d in docs.items():
        enc = tok.encode_with_offsets(d["text"])
        item = {"enc": enc, "ids": np.asarray(enc.ids, dtype=np.int64)}
        if d.get("gold") is not None:
            item["tags"], _ = spans_to_tags(d["text"], enc, d["gold"], tagset)
        out[did] = item
    return out


def windows(n, L, stride, offset=0):
    if n <= L:
        return [(0, n)]
    starts = list(range(-offset, n - L, stride)) if offset else list(range(0, n - L, stride))
    starts = [max(0, s) for s in starts]
    starts.append(n - L)
    return sorted(set((s, s + L) for s in starts))


def collate(items, prep, pad_id, teacher=None, K=None):
    T = max(e - s for _, s, e in items)
    B = len(items)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    keep = torch.zeros((B, T), dtype=torch.bool)
    labels = torch.full((B, T), -100, dtype=torch.long)
    tch = torch.zeros((B, T, K), dtype=torch.float16) if teacher is not None else None
    for b, (did, s, e) in enumerate(items):
        n = e - s
        ids[b, :n] = torch.from_numpy(prep[did]["ids"][s:e])
        keep[b, :n] = True
        if "tags" in prep[did]:
            labels[b, :n] = torch.from_numpy(prep[did]["tags"][s:e])
        if tch is not None:
            tch[b, :n] = torch.from_numpy(teacher[did][s:e])
    return ids, keep, labels, tch


# ------------------------------------------------------------------------------------------- inference
def amp_dtype(device):
    if device != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


@torch.no_grad()
def predict_scores(model, prep, L, stride, batch_size=16, device="cuda"):
    """Per-document token scores [N,K] (log-probs for the linear head, raw emissions for the CRF)."""
    model.eval()
    dt = amp_dtype(device)
    causal = model.backbone.attention == "causal"
    K = model.tagset.n
    acc = {d: np.zeros((len(p["ids"]), K), dtype=np.float32) for d, p in prep.items()}
    wsum = {d: np.zeros(len(p["ids"]), dtype=np.float32) for d, p in prep.items()}
    items = sorted(((d, s, e) for d, p in prep.items() for s, e in windows(len(p["ids"]), L, stride)),
                   key=lambda x: x[2] - x[1])                  # equal lengths batch together -> little padding
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        ids, keep, _, _ = collate(chunk, prep, pad_id=0)       # padded positions are masked, any valid id works
        ids, keep = ids.to(device), keep.to(device)
        with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
            em = model.emissions(ids, keep)
        sc = (em if model.crf is not None else F.log_softmax(em, -1)).float().cpu().numpy()
        for b, (d, s, e) in enumerate(chunk):
            n = e - s
            pos = np.arange(n, dtype=np.float32)
            w = (pos + 1) if causal else np.minimum(pos + 1, n - pos)
            acc[d][s:e] += sc[b, :n] * w[:, None]
            wsum[d][s:e] += w
    return {d: acc[d] / np.maximum(wsum[d], 1e-6)[:, None] for d in acc}


def decode_all(model, docs, prep, scores, cfg, trim_rules):
    trans = start = end = None
    if model.crf is not None:
        tr, st, en = model.crf.full_trans()
        trans, start, end = (x.detach().float().cpu().numpy() for x in (tr, st, en))
    out = {}
    for did, d in docs.items():
        sp = decode_doc(d["text"], prep[did]["enc"], scores[did], model.tagset, cfg["decode"]["strategy"],
                        trans, start, end)
        if cfg["decode"].get("trim", True):
            sp = trim_spans(d["text"], sp, trim_rules)
        out[did] = sorted(set(sp))
    return out


def evaluate(model, docs, prep, cfg, trim_rules, device):
    sc = predict_scores(model, prep, cfg["window"]["max_len"], cfg["window"]["stride"],
                        cfg["train"]["batch_size"] * 2, device)
    pred = decode_all(model, docs, prep, sc, cfg, trim_rules)
    gold = {k: d["gold"] for k, d in docs.items()}
    return strict_scores(gold, pred, model.tagset.labels), pred


# ------------------------------------------------------------------------------------------- run io
def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2, default=str)


def load_run(run_dir, device="cuda"):
    """Rebuild a trained model from a run folder (Phase-1 checkpoint + saved trainable weights)."""
    run_dir = Path(run_dir)
    cfg = json.load(open(run_dir / "config.json"))
    labels = json.load(open(run_dir / "labels.json"))
    tagset = TagSet(labels)
    model, _ = build_model(cfg, tagset)
    sd = torch.load(run_dir / "best.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    trim_rules = json.load(open(run_dir / "trim_rules.json"))
    return model.to(device), cfg, tagset, trim_rules


def write_submission(pred, test_ids, labels, out_dir):
    """prediction.json at the zip root; checks IDs/labels/offsets before zipping."""
    import zipfile
    assert set(pred) == set(test_ids), "prediction IDs must match the official test IDs exactly"
    lab = set(labels)
    obj = {}
    for d in test_ids:
        rows = []
        for s, e, l in pred[d]:
            assert l in lab and 0 <= s < e, (d, s, e, l)
            rows.append([int(s), int(e), l])
        obj[d] = rows
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pj = out_dir / "prediction.json"
    json.dump(obj, open(pj, "w", encoding="utf-8"))
    zp = out_dir / "submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"]
    return zp


# ------------------------------------------------------------------------------------------- main loop
def run(cfg, data=None, device=None, log=print):
    cfg = resolve_paths(merge(DEFAULTS, cfg))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(cfg["train"]["seed"])
    run_dir = Path(cfg["out_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, run_dir / "config.json")

    # ---- data
    data = data or D.load_phase2(cfg["data_dir"])
    labels = data["labels"]
    tagset = TagSet(labels)
    save_json(labels, run_dir / "labels.json")
    train_docs, dev_docs = D.make_split(data, **cfg["split"])
    log(f"[data] train docs {len(train_docs)} | dev docs {len(dev_docs)} | test docs {len(data['test'])}")
    tok = get_tokenizer(cfg)
    trim_rules = learn_trim_rules(train_docs) if cfg["decode"].get("trim", True) else None
    save_json(trim_rules, run_dir / "trim_rules.json")
    reach, _, skipped = reachability(train_docs, tok, tagset)
    log(f"[data] BIO round-trip ceiling on train: macro-F1 {reach['macro_f1']:.4f} (overlaps dropped: {skipped})")
    t0 = time.time()
    prep_tr, prep_dev = prepare(train_docs, tok, tagset), prepare(dev_docs, tok, tagset)
    tok.save_disk_cache()
    log(f"[data] tokenised in {time.time() - t0:.0f}s | train tokens {sum(len(p['ids']) for p in prep_tr.values()):,}")

    # ---- teacher (knowledge distillation)
    teacher = None
    if cfg["kd"].get("teacher_run"):
        assert cfg["head"]["type"] == "linear", "KD is implemented for the linear head"
        teacher, t_eval = teacher_scores(cfg, tagset, {**prep_tr, **prep_dev}, dev_docs, prep_dev, trim_rules, device, run_dir, log)

    # ---- model
    model, info, pc = new_model(cfg, tagset, prep_tr, device)
    save_json(info, run_dir / "model_info.json")
    log(f"[model] arch {info['arch']} ({info['class']}) | layers {info['layers_used']}/{info['phase1_layers']} | "
        f"attention {info['attention']} | method {cfg['method']['name']} | head {cfg['head']['type']}")
    log(f"[model] params total {pc['total'] / 1e6:.2f}M | trainable {pc['trainable'] / 1e6:.3f}M ({pc['trainable_pct']}%)")
    L, stride, tc = cfg["window"]["max_len"], cfg["window"]["stride"], cfg["train"]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_train = time.time()
    hist, best, best_epoch = fit(model, prep_tr, teacher, cfg, tagset, tok.pad_id, device, log,
                                 dev=(dev_docs, prep_dev, trim_rules), save_path=run_dir / "best.pt",
                                 hist_path=run_dir / "history.json")
    train_min = (time.time() - t_train) / 60
    peak_mem = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None

    # ---- final evaluation with the best weights
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device), strict=False)
    summary = {"run": cfg["run_name"], "arch": info["arch"], "method": cfg["method"], "head": cfg["head"]["type"],
               "attention": info["attention"], "params_total": pc["total"], "params_trainable": pc["trainable"],
               "trainable_pct": pc["trainable_pct"], "train_minutes": round(train_min, 1),
               "peak_train_mem_gb": peak_mem, "best_dev_macro_f1": best if dev_docs else None,
               "best_epoch": best_epoch, "bio_ceiling_macro_f1": reach["macro_f1"]}
    if dev_docs:
        t0 = time.time()
        sc, pred = evaluate(model, dev_docs, prep_dev, cfg, trim_rules, device)
        dt_inf = time.time() - t0
        texts = {k: d["text"] for k, d in dev_docs.items()}
        err = error_analysis({k: d["gold"] for k, d in dev_docs.items()}, pred, texts)
        save_json(sc, run_dir / "dev_scores.json"); save_json(err, run_dir / "dev_errors.json")
        save_json({k: [list(x) for x in v] for k, v in pred.items()}, run_dir / "dev_prediction.json")
        ntok = sum(len(p["ids"]) for p in prep_dev.values())
        summary.update(dev=dict((k, sc[k]) for k in ("macro_f1", "macro_p", "macro_r", "micro_f1")),
                       infer_tokens_per_sec=round(ntok / dt_inf), error_categories=err["categories"])
        log("\n" + format_table(sc))
        log(f"[errors] {err['categories']}")
    test_docs = {k: {"text": v} for k, v in data["test"].items()}
    t_prep = prepare(test_docs, tok, tagset) if cfg.get("predict_test", True) or cfg.get("refit_full") else None
    tok.save_disk_cache()
    if cfg.get("predict_test", True):
        zp, n_ent = predict_test(model, test_docs, t_prep, cfg, trim_rules, labels, device, run_dir / "submission")
        summary.update(submission=str(zp), test_entities=n_ent)
        log(f"[submit] {zp} | {n_ent} entities over {len(test_docs)} docs")

    # ---- optional refit on train + dev for the selected number of epochs (final submission)
    if cfg.get("refit_full") and dev_docs:
        del model
        torch.cuda.empty_cache() if device == "cuda" else None
        prep_all = {**prep_tr, **prep_dev}
        all_docs = {**train_docs, **dev_docs}
        trim_all = learn_trim_rules(all_docs) if cfg["decode"].get("trim", True) else None
        rcfg = merge(cfg, {"train": {"epochs": max(1, best_epoch)}})
        seed_all(tc["seed"])
        model, _, _ = new_model(rcfg, tagset, prep_all, device)
        log(f"[refit] training on all {len(all_docs)} documents for {rcfg['train']['epochs']} epochs (best dev epoch)")
        t0 = time.time()
        fit(model, prep_all, teacher, rcfg, tagset, tok.pad_id, device, log, dev=None,
            save_path=run_dir / "refit.pt", hist_path=run_dir / "history_refit.json")
        save_json(trim_all, run_dir / "trim_rules_refit.json")
        zp, n_ent = predict_test(model, test_docs, t_prep, rcfg, trim_all, labels, device, run_dir / "submission_refit")
        summary.update(submission_refit=str(zp), refit_minutes=round((time.time() - t0) / 60, 1))
        log(f"[refit] {zp} | {n_ent} entities")
    save_json(summary, run_dir / "summary.json")
    return summary, model


def predict_test(model, test_docs, t_prep, cfg, trim_rules, labels, device, out_dir):
    tsc = predict_scores(model, t_prep, cfg["window"]["max_len"], cfg["window"]["stride"], cfg["train"]["batch_size"] * 2, device)
    tpred = decode_all(model, test_docs, t_prep, tsc, cfg, trim_rules)
    return write_submission(tpred, list(test_docs), labels, out_dir), sum(len(v) for v in tpred.values())


def new_model(cfg, tagset, prep_tr, device):
    model, info = build_model(cfg, tagset)
    n_prompt = 0 if model.backbone.soft_prompt is None else model.backbone.soft_prompt.size(0)
    L = cfg["window"]["max_len"]
    assert L + n_prompt <= info["max_len"], f"max_len {L} (+{n_prompt} prompt) > backbone positions {info['max_len']}"
    if cfg["loss"]["class_weight"] == "sqrt_inv":
        cnt = np.bincount(np.concatenate([p["tags"] for p in prep_tr.values()]), minlength=tagset.n).astype(np.float64)
        ref = np.median(cnt[cnt > 0])
        w = np.clip((ref / np.maximum(cnt, 1)) ** cfg["loss"]["cw_power"], 1.0 / cfg["loss"]["cw_max"], cfg["loss"]["cw_max"])
        model.class_w.copy_(torch.tensor(w / w.mean(), dtype=torch.float32))
    model.to(device)
    pc = count_params(model)
    info.update(params=pc)
    return model, info, pc


def teacher_scores(cfg, tagset, prep, dev_docs, prep_dev, trim_rules, device, run_dir, log):
    """Teacher log-probs for every training token, cached in the teacher's run folder so that several
    students (and the refit) reuse them instead of re-running the teacher."""
    import pickle
    trun = Path(cfg["kd"]["teacher_run"])
    tcfg = json.load(open(trun / "config.json"))
    assert json.load(open(trun / "labels.json")) == tagset.labels
    assert tcfg["tokenizer"].get("fine_punct", True) == cfg["tokenizer"].get("fine_punct", True), "teacher/student tokenisation differ"
    cache = trun / f"teacher_scores_fp{int(cfg['tokenizer'].get('fine_punct', True))}.pkl"
    scores = pickle.load(open(cache, "rb")) if cache.exists() else {}
    todo = {d: p for d, p in prep.items() if d not in scores}
    t_eval = json.load(open(trun / "summary.json")).get("dev", {}).get("macro_f1") if (trun / "summary.json").exists() else None
    if todo:
        tm, tcfg, _, _ = load_run(trun, device)
        new = predict_scores(tm, todo, tcfg["window"]["max_len"], tcfg["window"]["stride"], 16, device)
        scores.update({d: s.astype(np.float16) for d, s in new.items()})
        tmp = cache.with_suffix(f".{os.getpid()}.tmp")
        pickle.dump(scores, open(tmp, "wb")); os.replace(tmp, cache)
        save_json({"teacher_run": str(trun), "teacher_dev_macro_f1": t_eval, "teacher_params": count_params(tm)},
                  run_dir / "teacher.json")
        del tm
        torch.cuda.empty_cache() if device == "cuda" else None
    log(f"[kd] teacher {trun} | dev macro-F1 {t_eval} | scores for {len(prep)} docs ({len(todo)} computed, rest cached)")
    return scores, t_eval


def make_optimizer(model, tc, device):
    """AdamW with separate LRs for new parameters (head, PEFT) and pretrained backbone parameters.
    train.llrd < 1 applies layer-wise LR decay to the backbone: block i of n gets lr * llrd^(n - i),
    embeddings lr * llrd^(n+1) (lower layers hold more general features and change less)."""
    n_layers = len(model.backbone.blocks)
    llrd = tc.get("llrd", 1.0)
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        new = is_new_param(n) or n.startswith("head.") or n.startswith("crf.")
        if new:
            lr, depth = tc["lr_new"], "new"
        else:
            if ".blocks." in n:
                depth = int(n.split(".blocks.")[1].split(".")[0]) + 1
            elif "tok_emb" in n or "seg_emb" in n:
                depth = 0
            else:
                depth = n_layers + 1                                # final norm
            lr = tc["lr_backbone"] * llrd ** (n_layers + 1 - depth)
        wd = tc["weight_decay"] if p.dim() >= 2 else 0.0
        groups.setdefault((depth, wd), {"params": [], "lr": lr, "weight_decay": wd})["params"].append(p)
    return torch.optim.AdamW(list(groups.values()), fused=(device == "cuda"))


def epoch_batches(prep, L, stride, bs, rng, random_offset):
    """Shuffled windows, length-bucketed: windows of equal length share a batch, so full-length batches carry
    no padding (lets SDPA skip the mask entirely for bidirectional models)."""
    items = []
    for d, p in prep.items():
        off = rng.randrange(stride) if random_offset else 0
        items += [(d, s, e) for s, e in windows(len(p["ids"]), L, stride, off)]
    rng.shuffle(items)
    batches = []
    chunk = bs * 50
    for c in range(0, len(items), chunk):
        part = sorted(items[c:c + chunk], key=lambda x: x[2] - x[1])
        batches += [part[i:i + bs] for i in range(0, len(part), bs)]
    rng.shuffle(batches)
    return batches


def fit(model, prep_tr, teacher, cfg, tagset, pad_id, device, log, dev=None, save_path=None, hist_path=None):
    """Training loop. With `dev`, evaluates every eval_every_epochs, keeps the best weights and early-stops;
    without it, trains the fixed number of epochs and saves the last weights."""
    tc = cfg["train"]
    L, stride = cfg["window"]["max_len"], cfg["window"]["stride"]
    opt = make_optimizer(model, tc, device)
    base_lrs = [g["lr"] for g in opt.param_groups]
    n_win = sum(len(windows(len(p["ids"]), L, stride)) for p in prep_tr.values())
    steps_per_epoch = math.ceil(math.ceil(n_win / tc["batch_size"]) / tc["grad_accum"])
    total = steps_per_epoch * tc["epochs"]
    warm = max(1, int(tc["warmup_frac"] * total))
    dt = amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(dt == torch.float16))
    log(f"[train] {n_win} windows/epoch | {steps_per_epoch} steps/epoch | {total} steps | amp {dt}")
    dev_docs, prep_dev, trim_rules = dev if dev else ({}, {}, None)
    trainable = [p for p in model.parameters() if p.requires_grad]
    hist, best, best_epoch, bad_epochs, step = [], -1.0, 0, 0, 0
    rng = random.Random(tc["seed"])
    t0 = time.time()
    for ep in range(1, tc["epochs"] + 1):
        model.train()
        tot_loss, nb = 0.0, 0
        for batch in epoch_batches(prep_tr, L, stride, tc["batch_size"], rng, cfg["window"].get("random_offset")):
            ids, keep, lab, tch = collate(batch, prep_tr, pad_id, teacher, tagset.n)
            ids, keep, lab = ids.to(device, non_blocking=True), keep.to(device, non_blocking=True), lab.to(device, non_blocking=True)
            tch = tch.to(device, non_blocking=True) if tch is not None else None
            with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
                em = model.emissions(ids, keep)
            loss = model.loss(em, lab, keep, {**cfg["loss"], **cfg["kd"]}, tch) / tc["grad_accum"]
            scaler.scale(loss).backward()
            tot_loss += loss.detach() * tc["grad_accum"]; nb += 1   # no per-step GPU sync
            if nb % tc["grad_accum"] == 0:
                # linear warm-up, then linear decay to 0
                lr_mult = (step + 1) / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))
                for g, b in zip(opt.param_groups, base_lrs):
                    g["lr"] = b * lr_mult
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, tc["grad_clip"])
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                step += 1
        rec = {"epoch": ep, "train_loss": float(tot_loss) / max(nb, 1), "minutes": round((time.time() - t0) / 60, 2)}
        if dev_docs and ep % tc["eval_every_epochs"] == 0:
            sc, _ = evaluate(model, dev_docs, prep_dev, cfg, trim_rules, device)
            rec.update(dev_macro_f1=sc["macro_f1"], dev_macro_p=sc["macro_p"], dev_macro_r=sc["macro_r"], dev_micro_f1=sc["micro_f1"])
            if sc["macro_f1"] > best:
                best, best_epoch, bad_epochs = sc["macro_f1"], ep, 0
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
