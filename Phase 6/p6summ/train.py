import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .context import fid_chunks, source_for
from .generate import GEN_DEFAULTS, generate
from .metrics import analysis, rouge_scores
from .model import Summarizer, seq_loss, trainable_state
from .peft import AdaLoRAController, apply_method, count_params, is_new_param
from .tokenizer import BPETokenizer

DEFAULTS = {
    "run_name": "debug",
    "out_dir": "runs",
    "data_dir": "",
    "cache_dir": "",
    "ckpt_root": "",
    "data": {"dev_size": 300, "dev_frac": 0.05, "split_seed": 0},
    "backbone": {"ckpt": "", "keep_layers": None, "grad_ckpt": True, "random_init": False, "rope_scale": "auto"},
    "method": {"name": "full"},
    "context": {"src_len": 1400, "strategy": "headtail", "head_frac": 0.5, "k": 4, "lead": 256, "gap": True},
    "target": {"tgt_len": "auto", "tgt_quantile": 0.9, "tgt_cap": 640},
    "sep_text": "\n\nSUMMARY:\n",
    "kd": {"teacher_run": "", "kd_alpha": 0.5, "kd_T": 2.0},
    "dapt": {"steps": 0, "len": 1024, "batch_size": 4, "grad_accum": 2},
    "adalora": {"target_r": 4, "init_warmup": 0.1, "final_warmup": 0.2, "delta_t": 10, "orth_reg": 0.1},
    "train": {"epochs": 3, "docs_per_epoch": None, "batch_size": 4, "grad_accum": 4, "lr_backbone": 5e-5,
              "lr_new": 5e-4, "weight_decay": 0.01, "warmup_frac": 0.05, "patience": 2, "grad_clip": 1.0,
              "seed": 1337, "max_minutes": None, "eval_batch_size": 8, "log_every": 0.1, "gen_eval_docs": 16},
    "gen": dict(GEN_DEFAULTS, max_new="auto", min_new="auto"),
    "post": {"dedup_sentences": True, "max_words": None},
    "eval": {"gen_docs": 200},
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

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def tok_path(cfg):
    return str(Path(cfg["backbone"]["ckpt"]).parent / "tokenizer.json")

def cache_dir(cfg):
    return Path(cfg.get("cache_dir") or Path(cfg.get("out_dir", "runs")).resolve().parent / "p6cache")

def load_data(cfg, log=print):
    dc = cfg["data"]
    return D.load_cached(cfg["data_dir"], tok_path(cfg), cache_dir(cfg), cfg.get("max_docs"), cfg.get("n_proc", 4),
                         dc["dev_size"], dc["dev_frac"], dc["split_seed"], log)

def amp_dtype(device):
    if device != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2, default=str)

def resolve_spec(cfg, data, tok):
    if cfg.get("resolved"):
        return cfg["resolved"]
    t = cfg["target"]
    if t["tgt_len"] == "auto":
        lens = np.array([len(x) for x in data["train"]["sum_tokens"]]) + 1
        tgt = int(min(t["tgt_cap"], math.ceil(np.quantile(lens, t["tgt_quantile"]) / 16) * 16))
    else:
        tgt = int(t["tgt_len"])
    sep = tok.encode(cfg["sep_text"])
    g = cfg["gen"]
    max_new = tgt if g["max_new"] == "auto" else int(g["max_new"])
    if g["min_new"] == "auto":
        lens = np.array([len(x) for x in data["train"]["sum_tokens"]])
        min_new = int(min(max_new // 2, np.quantile(lens, 0.1)))
    else:
        min_new = int(g["min_new"])
    return {"tgt_len": tgt, "src_len": int(cfg["context"]["src_len"]), "sep_ids": sep, "max_new": max_new,
            "min_new": min_new, "gap_ids": tok.encode(" ...") if cfg["context"].get("gap", True) else []}

def build_model(cfg, spec, tok):
    bc = cfg["backbone"]
    model = Summarizer(bc["ckpt"], bc.get("keep_layers"), bc.get("grad_ckpt", True), bc.get("random_init", False))
    model.bos_id = tok.bos_id
    apply_method(model, cfg["method"])
    if model.arch == "B":
        train_len = spec["tgt_len"] + 1
        table = max(spec["max_new"], spec["tgt_len"]) + 2
    else:
        train_len = spec["src_len"] + len(spec["sep_ids"]) + spec["tgt_len"]
        table = spec["src_len"] + len(spec["sep_ids"]) + max(spec["max_new"], spec["tgt_len"]) + 2
    model.rope_info = model.set_lengths(train_len, bc.get("rope_scale", "auto"), table)
    info = {"arch": model.arch, "class": model.cls_name, "model_kw": model.kw, "phase1_layers": model.phase1_layers,
            "layers_used": model.n_layers(), "rope": model.rope_info, "train_seq_len": train_len,
            "random_init": bool(bc.get("random_init"))}
    if model.arch == "B":
        info["fid_chunks"] = math.ceil(spec["src_len"] / model.enc_len)
    return model, info

def load_run(run_dir, device="cuda", weights="best.pt"):
    run_dir = Path(run_dir)
    cfg = json.load(open(run_dir / "config.json"))
    tok = BPETokenizer.load(tok_path(cfg))
    model, _ = build_model(cfg, cfg["resolved"], tok)
    sd = torch.load(run_dir / weights, map_location="cpu")
    _, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    return model.to(device), cfg, tok

def make_target(sum_tokens, tgt_len, eot):
    s = [int(x) for x in sum_tokens]
    return s + [eot] if len(s) <= tgt_len - 1 else s[:tgt_len]

def example(split, i, cfg, spec, eot, with_target=True):
    ex = {"src": source_for(split, i, cfg["context"], spec["src_len"], spec["gap_ids"]).astype(np.int64)}
    if with_target:
        ex["tgt"] = make_target(split["sum_tokens"][i], spec["tgt_len"], eot)
    return ex

def collate(exs, arch, spec, tok, enc_len=None):
    pad = tok.pad_id
    if arch == "B":
        chunks = [(b, k, c) for b, ex in enumerate(exs) for k, c in enumerate(fid_chunks(ex["src"], enc_len))]
        S = max(len(c) for _, _, c in chunks)
        src = torch.full((len(chunks), S), pad, dtype=torch.long)
        keep = torch.zeros(len(chunks), S, dtype=torch.bool)
        for r, (_, _, c) in enumerate(chunks):
            src[r, :len(c)] = torch.from_numpy(c)
            keep[r, :len(c)] = True
        T = max(len(ex["tgt"]) for ex in exs)
        dec_in = torch.full((len(exs), T), pad, dtype=torch.long)
        labels = torch.full((len(exs), T), -100, dtype=torch.long)
        for b, ex in enumerate(exs):
            t = ex["tgt"]
            dec_in[b, :len(t)] = torch.tensor([tok.bos_id] + t[:-1])
            labels[b, :len(t)] = torch.tensor(t)
        return {"src": src, "keep": keep, "doc": torch.tensor([b for b, _, _ in chunks]),
                "slot": torch.tensor([k for _, k, _ in chunks]), "n_docs": len(exs),
                "k_max": max(k for _, k, _ in chunks) + 1, "dec_in": dec_in, "labels": labels}
    seqs, labs, pls = [], [], []
    for ex in exs:
        if "lm" in ex:
            w = list(map(int, ex["lm"]))
            seqs.append(w[:-1]); labs.append(w[1:]); pls.append(0)
            continue
        prompt = list(map(int, ex["src"])) + spec["sep_ids"]
        seq = prompt + ex["tgt"]
        seqs.append(seq[:-1])
        labs.append([-100] * (len(prompt) - 1) + ex["tgt"])
        pls.append(len(prompt))
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    labels = torch.full((len(seqs), T), -100, dtype=torch.long)
    for b, (s, l) in enumerate(zip(seqs, labs)):
        assert len(s) == len(l)
        ids[b, :len(s)] = torch.tensor(s)
        labels[b, :len(l)] = torch.tensor(l)
    return {"ids": ids, "labels": labels, "prefix_len": torch.tensor(pls, dtype=torch.long)}

def to_dev(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

def gen_input(ex, arch, spec, enc_len=None):
    if arch == "B":
        return [torch.from_numpy(c) for c in fid_chunks(ex["src"], enc_len)]
    return torch.tensor(list(map(int, ex["src"])) + spec["sep_ids"], dtype=torch.long)

@torch.no_grad()
def dev_loss(model, split, idx, cfg, spec, tok, device, bs):
    model.eval()
    dt = amp_dtype(device)
    eot = tok.special_to_id["<|endoftext|>"]
    tot, n = 0.0, 0
    order = sorted(idx, key=lambda i: len(split["tokens"][i]))
    for k in range(0, len(order), bs):
        exs = [example(split, i, cfg, spec, eot) for i in order[k:k + bs]]
        b = to_dev(collate(exs, model.arch, spec, tok, getattr(model, "enc_len", None)), device)
        with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
            lg, lab = model(b)
        tot += float(torch.nn.functional.cross_entropy(lg.float(), lab, reduction="sum"))
        n += int(lab.numel())
    return tot / max(1, n)

def postprocess(text, pcfg):
    t = " ".join(text.replace("�", " ").split())
    if pcfg.get("dedup_sentences", True):
        seen, keep = set(), []
        for s in D.split_sentences(t):
            key = " ".join(s.lower().split())
            if len(key.split()) >= 4 and key in seen:
                continue
            seen.add(key); keep.append(s)
        t = "".join(keep).strip()
    if pcfg.get("max_words"):
        t = " ".join(t.split()[:int(pcfg["max_words"])])
    return t

def summarise(model, split, idx, cfg, spec, tok, device, gcfg=None, ccfg=None, log=None):
    c = dict(cfg) if ccfg is None else merge(cfg, {"context": ccfg})
    eot = tok.special_to_id["<|endoftext|>"]
    inputs = [gen_input(example(split, i, c, spec, eot, with_target=False), model.arch, spec,
                        getattr(model, "enc_len", None)) for i in idx]
    g = merge(cfg["gen"], gcfg or {})
    g["max_new"] = spec["max_new"] if g["max_new"] == "auto" else int(g["max_new"])
    g["min_new"] = spec["min_new"] if g["min_new"] == "auto" else int(g["min_new"])
    toks, speed = generate(model, inputs, g, tok, device, log, amp_dtype(device))
    texts = [postprocess(tok.decode(t), cfg["post"]) for t in toks]
    return texts, [len(t) for t in toks], speed

def score_generations(texts, split, idx):
    refs = [split["summaries"][i] for i in idx]
    srcs = [split["documents"][i] for i in idx]
    r = rouge_scores(texts, refs)
    a = analysis(texts, refs, srcs)
    return {k: v for k, v in r.items() if k != "per_doc"} | a, r["per_doc"]

def check_prediction(obj, raw_ids):
    assert isinstance(obj, list), "prediction.json must be a JSON list"
    got = [r["id"] for r in obj]
    assert len(got) == len(raw_ids), f"{len(got)} predictions for {len(raw_ids)} test documents"
    assert len({json.dumps(g) for g in got}) == len(got), "duplicate ids in the submission"
    assert {json.dumps(g) for g in got} == {json.dumps(g) for g in raw_ids}, "submission ids differ from test.json ids"
    for r in obj:
        assert set(r) == {"id", "summary"}, f"unexpected keys {sorted(r)}"
        assert isinstance(r["summary"], str) and r["summary"].strip(), f"empty summary for {r['id']!r}"

def write_submission(raw_ids, texts, out_dir):
    import zipfile
    obj = [{"id": rid, "summary": t} for rid, t in zip(raw_ids, texts)]
    check_prediction(obj, raw_ids)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pj = out_dir / "prediction.json"
    json.dump(obj, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
    zp = out_dir / "submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"]
        check_prediction(json.loads(z.read("prediction.json")), raw_ids)
    return zp

def fill_empty(texts, split, idx, n_words=60):
    out, n = [], 0
    for t, i in zip(texts, idx):
        if not t.strip():
            t, n = " ".join(split["documents"][i].split()[:n_words]) or ".", n + 1
        out.append(t)
    return out, n

def make_optimizer(model, tc, device, lr_mult=1.0):
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        new = is_new_param(n)
        lr = (tc["lr_new"] if new else tc["lr_backbone"]) * lr_mult
        wd = tc["weight_decay"] if p.dim() >= 2 else 0.0
        groups.setdefault((new, wd), {"params": [], "lr": lr, "weight_decay": wd})["params"].append(p)
    return torch.optim.AdamW(list(groups.values()), fused=(device == "cuda"))

def length_batches(idx, lens, bs, rng):
    items = list(map(int, idx))
    rng.shuffle(items)
    batches, block = [], bs * 50
    for c in range(0, len(items), block):
        part = sorted(items[c:c + block], key=lambda i: lens[i])
        batches += [part[i:i + bs] for i in range(0, len(part), bs)]
    rng.shuffle(batches)
    return batches

class Teacher:
    def __init__(self, run_dir, device, student_spec):
        self.model, self.cfg, self.tok = load_run(run_dir, device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.spec = self.cfg["resolved"]
        assert self.spec["tgt_len"] == student_spec["tgt_len"], \
            f"teacher tgt_len {self.spec['tgt_len']} != student {student_spec['tgt_len']}: same summary targets needed"
        self.eot = self.tok.special_to_id["<|endoftext|>"]

    @torch.no_grad()
    def logits(self, split, batch_idx, device, student_labels):
        exs = [example(split, i, self.cfg, self.spec, self.eot) for i in batch_idx]
        b = to_dev(collate(exs, self.model.arch, self.spec, self.tok, getattr(self.model, "enc_len", None)), device)
        dt = amp_dtype(device)
        with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
            lg, lab = self.model(b)
        assert torch.equal(lab, student_labels), "teacher and student summary tokens are not aligned"
        return lg

def dapt(model, tr, cfg, spec, tok, device, log, deadline):
    dc, tc = cfg["dapt"], cfg["train"]
    assert model.arch != "B", "DAPT is implemented for the decoder-style models (A, C, D)"
    L = min(dc["len"], spec["src_len"] + len(spec["sep_ids"]) + spec["tgt_len"])
    rng = np.random.default_rng(tc["seed"] + 1)
    long_docs = [i for i, t in enumerate(tr["tokens"]) if len(t) > L + 1]
    opt = make_optimizer(model, tc, device)
    base = [g["lr"] for g in opt.param_groups]
    dt = amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(dt == torch.float16))
    trainable = [p for p in model.parameters() if p.requires_grad]
    model.train()
    t0, run_loss = time.time(), []
    for step in range(dc["steps"]):
        for _ in range(dc["grad_accum"]):
            exs = []
            for i in rng.choice(long_docs, dc["batch_size"]):
                s = int(rng.integers(0, len(tr["tokens"][i]) - L - 1))
                exs.append({"lm": tr["tokens"][i][s:s + L + 1]})
            b = to_dev(collate(exs, model.arch, spec, tok), device)
            with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
                lg, lab = model(b)
            loss = torch.nn.functional.cross_entropy(lg.float(), lab) / dc["grad_accum"]
            scaler.scale(loss).backward()
            run_loss.append(float(loss.detach()) * dc["grad_accum"])
        mult = min(1.0, (step + 1) / max(1, dc["steps"] // 20)) * max(0.0, 1 - step / dc["steps"])
        for g, b0 in zip(opt.param_groups, base):
            g["lr"] = b0 * mult
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, tc["grad_clip"])
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if (step + 1) % max(1, dc["steps"] // 10) == 0:
            log(f"[dapt] step {step + 1}/{dc['steps']} | LM loss {np.mean(run_loss[-50:]):.4f} | {(time.time() - t0) / 60:.1f} min")
        if deadline and time.time() > deadline:
            log("[dapt] time budget reached"); break
    return {"steps": step + 1, "len": L, "first_loss": float(np.mean(run_loss[:20])), "last_loss": float(np.mean(run_loss[-50:])),
            "minutes": round((time.time() - t0) / 60, 1)}

def fit(model, tr, dev, cfg, spec, tok, device, log, run_dir, teacher=None):
    tc = cfg["train"]
    eot = tok.special_to_id["<|endoftext|>"]
    enc_len = getattr(model, "enc_len", None)
    n = len(tr["ids"])
    dpe = min(n, tc.get("docs_per_epoch") or n)
    rng_np = np.random.default_rng(tc["seed"])
    schedule, stream = [], np.array([], dtype=np.int64)
    for _ in range(tc["epochs"]):
        while len(stream) < dpe:
            stream = np.concatenate([stream, rng_np.permutation(n)])
        schedule.append(stream[:dpe]); stream = stream[dpe:]
    lens = [min(len(t), spec["src_len"]) + len(s) for t, s in zip(tr["tokens"], tr["sum_tokens"])]
    opt = make_optimizer(model, tc, device)
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = math.ceil(math.ceil(dpe / tc["batch_size"]) / tc["grad_accum"])
    total = steps_per_epoch * len(schedule)
    warm = max(1, int(tc["warmup_frac"] * total))
    ada = None
    if cfg["method"]["name"] == "adalora":
        a = cfg["adalora"]
        ada = AdaLoRAController(model, total, a["target_r"], a["init_warmup"], a["final_warmup"], a["delta_t"],
                                orth_reg=a["orth_reg"])
        log(f"[adalora] {len(ada.mods)} modules | rank budget {ada.b0} -> {ada.bT} over {total} steps")
    dt = amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(dt == torch.float16))
    trainable = [p for p in model.parameters() if p.requires_grad]
    dev_idx = list(range(len(dev["ids"])))
    gen_idx = dev_idx[:tc.get("gen_eval_docs", 0)]
    log(f"[train] {len(schedule)} epochs x {dpe} documents | {steps_per_epoch} steps/epoch | {total} steps | "
        f"batch {tc['batch_size']} x accum {tc['grad_accum']} | amp {dt} | time limit {tc.get('max_minutes')} min")
    hist = []
    l0 = dev_loss(model, dev, dev_idx, cfg, spec, tok, device, tc["eval_batch_size"])
    hist.append({"epoch": 0, "dev_loss": round(l0, 4), "dev_ppl": round(math.exp(min(l0, 30)), 2)})
    log(f"[epoch] epoch 0 (before fine-tuning) | dev_loss {l0:.4f} | dev_ppl {math.exp(min(l0, 30)):.2f}")
    best, best_epoch, bad, step = float("inf"), 0, 0, 0
    rng = random.Random(tc["seed"])
    kd = cfg["kd"]
    t0 = time.time()
    limit = tc.get("max_minutes")
    stop = False
    for ep, idx in enumerate(schedule, 1):
        model.train()
        tot, parts, nbat, ntok = 0.0, {}, 0, 0
        batches = length_batches(idx, lens, tc["batch_size"], rng)
        every = max(1, int(len(batches) * tc.get("log_every", 0.1)))
        for bi, bidx in enumerate(batches):
            exs = [example(tr, i, cfg, spec, eot) for i in bidx]
            b = to_dev(collate(exs, model.arch, spec, tok, enc_len), device)
            with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
                lg, lab = model(b)
            tl = None if teacher is None else teacher.logits(tr, bidx, device, lab)
            loss, pr = seq_loss(lg, lab, tl, kd["kd_alpha"], kd["kd_T"])
            if ada is not None:
                o = ada.orth_loss()
                loss = loss + o
                pr["orth"] = float(o.detach())
            scaler.scale(loss / tc["grad_accum"]).backward()
            tot += float(loss.detach()); nbat += 1; ntok += int(lab.numel())
            for k, v in pr.items():
                parts[k] = parts.get(k, 0.0) + v
            if nbat % tc["grad_accum"] == 0 or bi == len(batches) - 1:
                lr_mult = (step + 1) / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))
                for g, b0 in zip(opt.param_groups, base_lrs):
                    g["lr"] = b0 * lr_mult
                scaler.unscale_(opt)
                if ada is not None:
                    ada.update(step)
                torch.nn.utils.clip_grad_norm_(trainable, tc["grad_clip"])
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                step += 1
            if (bi + 1) % every == 0:
                el = (time.time() - t0) / 60
                done = (ep - 1) + (bi + 1) / len(batches)
                log(f"[progress] epoch {ep} {100 * (bi + 1) / len(batches):3.0f}% | loss {tot / nbat:.4f} | "
                    + " ".join(f"{k} {v / nbat:.4f}" for k, v in parts.items())
                    + f" | {ntok / max(1e-6, el * 60):.0f} summary tok/s | {el:.1f} min | "
                    f"ETA all epochs {el / done * (len(schedule) - done):.1f} min")
            if limit and (time.time() - t0) / 60 > limit:
                log(f"[train] time limit {limit} min reached in epoch {ep} ({100 * (bi + 1) / len(batches):.0f}%): "
                    "evaluating and stopping")
                stop = True
                break
        rec = {"epoch": ep, "train_loss": round(tot / max(nbat, 1), 4), "minutes": round((time.time() - t0) / 60, 2)}
        rec.update({f"train_{k}": round(v / max(nbat, 1), 4) for k, v in parts.items()})
        dl = dev_loss(model, dev, dev_idx, cfg, spec, tok, device, tc["eval_batch_size"])
        rec.update(dev_loss=round(dl, 4), dev_ppl=round(math.exp(min(dl, 30)), 2))
        if ada is not None:
            rec["adalora_budget"] = ada.budget
        if gen_idx:
            g = {"max_new": min(256, spec["max_new"]), "min_new": min(spec["min_new"], 64), "method": "greedy"}
            texts, _, _ = summarise(model, dev, gen_idx, cfg, spec, tok, device, g)
            r = rouge_scores(texts, [dev["summaries"][i] for i in gen_idx])
            rec.update(quick_rouge1=round(r["rouge1_f"], 4), quick_rougeL=round(r["rougeL_f"], 4))
            rec["sample"] = texts[0][:300]
        if dl < best:
            best, best_epoch, bad = dl, ep, 0
            torch.save(trainable_state(model), run_dir / "best.pt"); rec["saved"] = True
        else:
            bad += 1
        hist.append(rec)
        log("[epoch] " + " | ".join(f"{k} {v}" for k, v in rec.items() if k != "sample"))
        if "sample" in rec:
            log(f"[epoch] sample: {rec['sample']}")
        save_json(hist, run_dir / "history.json")
        if stop:
            break
        if bad >= tc["patience"]:
            log(f"[train] early stop: no dev improvement for {tc['patience']} epochs")
            break
    return hist, best, best_epoch, (None if ada is None else ada.ranks())

def run(cfg, data=None, device=None, log=print):
    cfg = merge(DEFAULTS, cfg)
    roots = [cfg["ckpt_root"]] if cfg.get("ckpt_root") else []
    cfg["backbone"]["ckpt"] = resolve_ckpt(cfg["backbone"]["ckpt"], roots)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(cfg["train"]["seed"])
    run_dir = Path(cfg["out_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    tok = BPETokenizer.load(tok_path(cfg))
    data = data or load_data(cfg, log)
    tr, dev, te = data["train"], data["dev"], data["test"]
    spec = resolve_spec(cfg, data, tok)
    cfg["resolved"] = spec
    save_json(cfg, run_dir / "config.json")
    log(f"[data] train {len(tr['ids'])} | dev {len(dev['ids'])} (held out) | test {len(te['ids'])} documents | "
        f"src {spec['src_len']} tokens ({cfg['context']['strategy']}) | summary <= {spec['tgt_len']} tokens | "
        f"generation {spec['min_new']}..{spec['max_new']} tokens")

    teacher = None
    if cfg["kd"].get("teacher_run"):
        teacher = Teacher(cfg["kd"]["teacher_run"], device, spec)
        log(f"[kd] teacher {Path(cfg['kd']['teacher_run']).name} ({teacher.model.arch}, "
            f"{count_params(teacher.model)['total'] / 1e6:.1f}M) | alpha {cfg['kd']['kd_alpha']} | T {cfg['kd']['kd_T']}")

    model, info = build_model(cfg, spec, tok)
    model.to(device)
    pc = count_params(model)
    info.update(params=pc)
    save_json(info, run_dir / "model_info.json")
    log(f"[model] arch {info['arch']} ({info['class']}) | layers {info['layers_used']}/{info['phase1_layers']} | "
        f"method {cfg['method']['name']} | positions {info['rope']} | train sequence {info['train_seq_len']}"
        + (f" | FiD {info['fid_chunks']} x {model.enc_len}" if model.arch == "B" else ""))
    log(f"[model] params total {pc['total'] / 1e6:.2f}M | trainable {pc['trainable'] / 1e6:.3f}M ({pc['trainable_pct']}%)"
        + (f" | 4-bit NF4 weights {pc['nf4_weights'] / 1e6:.2f}M" if pc.get("nf4_weights") else ""))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    dapt_info = None
    if cfg["dapt"].get("steps"):
        lim = cfg["train"].get("max_minutes")
        dapt_info = dapt(model, tr, cfg, spec, tok, device, log, (t_start + 60 * lim / 3) if lim else None)
        if lim:
            cfg["train"]["max_minutes"] = max(5, lim - dapt_info["minutes"])
    hist, best, best_epoch, ada_ranks = fit(model, tr, dev, cfg, spec, tok, device, log, run_dir, teacher)
    train_min = (time.time() - t_start) / 60
    peak_mem = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
    del teacher
    if device == "cuda":
        torch.cuda.empty_cache()

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device), strict=False)
    gidx = list(range(min(cfg["eval"]["gen_docs"], len(dev["ids"]))))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    texts, ntok, speed = summarise(model, dev, gidx, cfg, spec, tok, device, log=log)
    gen_mem = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
    texts, n_empty_dev = fill_empty(texts, dev, gidx)
    scores, per_doc = score_generations(texts, dev, gidx)
    save_json([{"id": dev["ids"][i], "reference": dev["summaries"][i], "generated": t, "new_tokens": n,
                "doc_tokens": int(len(dev["tokens"][i]))} | {k: v[j] for k, v in per_doc.items()}
               for j, (i, t, n) in enumerate(zip(gidx, texts, ntok))], run_dir / "dev_generations.json")
    save_json(scores | {"generation": speed, "n_docs": len(gidx), "empty_replaced": n_empty_dev}, run_dir / "dev_metrics.json")
    log(f"[dev] {len(gidx)} documents | ROUGE-1 {scores['rouge1_f']:.4f} | ROUGE-2 {scores['rouge2_f']:.4f} | "
        f"ROUGE-L {scores['rougeL_f']:.4f} | words {scores['words']:.0f} (ref {scores['ref_words']:.0f}) | "
        f"rep-3gram {scores['rep_3gram']:.3f} | fact precision {scores['fact_precision']:.3f} | "
        f"{speed['docs_per_sec']} docs/s")
    summary = {"run": cfg["run_name"], "arch": info["arch"], "backbone": Path(cfg["backbone"]["ckpt"]).parent.name,
               "method": cfg["method"], "context": cfg["context"], "resolved": spec, "rope": info["rope"],
               "layers": info["layers_used"], "params_total": pc["total"], "params_trainable": pc["trainable"],
               "trainable_pct": pc["trainable_pct"], "nf4_weights": pc.get("nf4_weights", 0),
               "train_minutes": round(train_min, 1), "peak_train_mem_gb": peak_mem, "peak_gen_mem_gb": gen_mem,
               "best_epoch": best_epoch, "epochs_run": len(hist) - 1, "best_dev_loss": best,
               "dev_loss_before": hist[0]["dev_loss"], "kd_teacher": cfg["kd"].get("teacher_run") or None,
               "dapt": dapt_info, "adalora_final_ranks": ada_ranks, "gen": cfg["gen"],
               "dev": {k: scores[k] for k in ("rouge1_f", "rouge2_f", "rougeL_f", "words", "ref_words", "rep_3gram",
                                             "rep_sent", "novel_3gram", "fact_precision", "fact_recall")},
               "gen_docs_per_sec": speed["docs_per_sec"], "gen_tokens_per_sec": speed["tokens_per_sec"]}

    if cfg.get("predict_test", True):
        tidx = list(range(len(te["ids"])))
        tt, tn, tspeed = summarise(model, te, tidx, cfg, spec, tok, device, log=log)
        tt, n_empty = fill_empty(tt, te, tidx)
        save_json([{"id": te["ids"][i], "generated": t, "new_tokens": n} for i, t, n in zip(tidx, tt, tn)],
                  run_dir / "test_generations.json")
        zp = write_submission(te["raw_ids"], tt, run_dir / "submission")
        summary.update(submission=str(zp), test_words=round(float(np.mean([len(t.split()) for t in tt])), 1),
                       test_empty_replaced=n_empty, test_gen_seconds=tspeed["seconds"])
        log(f"[submit] {zp} | {len(tt)} documents | mean {summary['test_words']} words | "
            f"empty generations replaced: {n_empty}")
    save_json(summary, run_dir / "summary.json")
    return summary, model
