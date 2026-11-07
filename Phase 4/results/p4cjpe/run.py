"""CLI + analyses.

    python -m p4cjpe.run --config job.json                         # train one experiment
    python -m p4cjpe.run --eda --config job.json                   # data format + statistics only
    python -m p4cjpe.run --ensemble runs/R1 runs/R2 --out runs/ENS  # average the runs' probabilities
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .metrics import best_threshold, binary_report, format_report
from .train import (DEFAULTS, evaluate, load_data, load_probs, load_run, merge, predict_logits, probs_from_logits,
                    resolve_paths, run, save_json, set_by_path, write_submission)


def eda(cfg, log=print):
    D.describe_raw(cfg["data_dir"])
    data = load_data(cfg, log)
    print(json.dumps(D.stats(data), indent=1))
    return data


def ensemble(run_dirs, out_dir, data, calibrate=True):
    """Average P(label 1) of several finished runs (dev and test), tune the threshold on dev, write a submission."""
    run_dirs = [Path(r) for r in run_dirs]
    probs = {}
    for split in ("dev", "test"):
        acc = None
        for r in run_dirs:
            ids, lg = load_probs(r / f"{split}_logits.npz")
            assert ids == list(data[split]["ids"]), f"{r.name}: {split} ids differ"
            p = probs_from_logits(lg)
            acc = p if acc is None else acc + p
        probs[split] = acc / len(run_dirs)
    gold = data["dev"]["labels"]
    thr, _ = best_threshold(gold, probs["dev"]) if calibrate else (0.5, None)
    rep = binary_report(gold, (probs["dev"] >= 0.5).astype(int))
    rep_thr = binary_report(gold, (probs["dev"] >= thr).astype(int))
    print(format_report(rep_thr, f"ensemble of {[r.name for r in run_dirs]} | threshold {thr:.2f}"))
    out_dir = Path(out_dir)
    save_json({"argmax": rep, "calibrated": rep_thr, "threshold": thr}, out_dir / "dev_report.json")
    pred = (probs["test"] >= thr).astype(int)
    zp = write_submission(data["test"]["raw_ids"], pred, out_dir / "submission")
    summary = {"run": out_dir.name, "members": [r.name for r in run_dirs], "threshold": thr,
               "dev": {k: rep[k] for k in ("macro_f1", "macro_p", "macro_r", "accuracy")},
               "dev_calibrated": {k: rep_thr[k] for k in ("macro_f1", "macro_p", "macro_r", "accuracy")},
               "test_share_pred_1": round(float(pred.mean()), 3), "submission": str(zp)}
    save_json(summary, out_dir / "ensemble_summary.json")
    return summary


SECTION_VIEWS = {                       # which part of the judgment the trained model is allowed to read
    "first 512 (head)":        {"k": 1, "strategy": "head"},
    "middle 512":              {"k": 1, "strategy": "middle"},
    "last 512 (tail)":         {"k": 1, "strategy": "tail"},
    "head 2 chunks":           {"k": 2, "strategy": "head"},
    "tail 2 chunks":           {"k": 2, "strategy": "tail"},
    "trained view":            {},
    "tail 4 chunks":           {"k": 4, "strategy": "tail"},
    "uniform 4 chunks":        {"k": 4, "strategy": "uniform"},
    "uniform 8 chunks":        {"k": 8, "strategy": "uniform"},
}


def section_analysis(run_dir, data, device="cuda", views=None, max_docs=None):
    """Inference-only evidence for 'which sections of a judgment carry the outcome': the trained model is evaluated
    on dev while reading only the beginning / middle / end (or more chunks than in training). Also returns the
    average chunk-attention weight per relative position bucket (attn/gru aggregators)."""
    model, cfg = load_run(run_dir, device)
    dev = data["dev"]
    if max_docs:
        dev = {k: (v[:max_docs] if k in ("ids", "raw_ids", "tokens", "labels", "n_chars") and v is not None else v)
               for k, v in dev.items()}
    rows = []
    for name, ov in (views or SECTION_VIEWS).items():
        rep, _, p1 = evaluate(model, dev, cfg, device, chunk_override=ov)
        rows.append({"view": name, "chunks": f"{ov.get('k', cfg['chunks']['k'])} x {cfg['chunks']['len']} "
                                             f"{ov.get('strategy', cfg['chunks']['strategy'])}",
                     "macro_f1": round(rep["macro_f1"], 4), "accuracy": round(rep["accuracy"], 4),
                     "pred_1_share": round(float((p1 >= 0.5).mean()), 3)})
    _, w = predict_logits(model, dev["tokens"], cfg, device, chunk_override={"k": 8, "strategy": "uniform"},
                          return_weights=True)
    nb = cfg["head"].get("pos_buckets", 8)
    att = None
    if w and w[0][0] is not None:
        s, c = np.zeros(nb), np.zeros(nb)
        for ww, pos in w:
            for a, b in zip(ww, pos):
                s[b] += a * len(ww); c[b] += 1                  # weight x n_chunks: 1.0 = uniform attention
        att = {f"{i / nb:.2f}-{(i + 1) / nb:.2f}": round(float(s[i] / c[i]), 3) for i in range(nb) if c[i]}
    out = {"run": Path(run_dir).name, "views": rows, "chunk_attention_by_position(uniform 8 chunks)": att}
    save_json(out, Path(run_dir) / "section_analysis.json")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--set", action="append", default=[], help="dotted.key=value override")
    ap.add_argument("--eda", action="store_true")
    ap.add_argument("--ensemble", nargs="+", default=[], help="run dirs whose probabilities are averaged")
    ap.add_argument("--out", default="runs/ENSEMBLE")
    a = ap.parse_args()
    cfg = merge(DEFAULTS, json.load(open(a.config)) if a.config else {})
    for kv in a.set:
        k, v = kv.split("=", 1)
        set_by_path(cfg, k, v)
    if a.eda:
        eda(resolve_paths(cfg))
    elif a.ensemble:
        print(ensemble(a.ensemble, a.out, load_data(resolve_paths(cfg))))
    else:
        run(cfg)


if __name__ == "__main__":
    main()
