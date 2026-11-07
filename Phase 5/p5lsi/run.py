import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .metrics import apply_thresholds, format_report, frequency_buckets, multilabel_report
from .train import (DEFAULTS, calibrate, load_data, load_probs, load_run, merge, predict_logits, preds_to_ids,
                    probs_from_logits, resolve_paths, run, save_json, set_by_path, subset, thr_from_json, thr_to_json,
                    write_submission)

def eda(cfg, log=print):
    D.describe_raw(cfg["data_dir"])
    data = load_data(cfg, log)
    print(json.dumps(D.stats(data), indent=1))
    return data

def ensemble(run_dirs, out_dir, data, ccfg=None):
    ccfg = merge(DEFAULTS["calibrate"], ccfg or {})
    run_dirs = [Path(r) for r in run_dirs]
    space = data["space"]
    for r in run_dirs:
        assert json.load(open(r / "label_space.json")) == [int(x) for x in space], f"{r.name}: label space differs"
    probs = {}
    for split in ("dev", "test"):
        acc = None
        for r in run_dirs:
            ids, lg = load_probs(r / f"{split}_logits.npz")
            assert ids == list(data[split]["ids"]), f"{r.name}: {split} ids differ"
            p = probs_from_logits(lg)
            acc = p if acc is None else acc + p
        probs[split] = acc / len(run_dirs)
    Y, ml = data["dev"]["Y"], ccfg.get("min_labels", 1)
    thr, strat, cf = calibrate(Y, probs["dev"], ccfg)
    rep05 = multilabel_report(Y, apply_thresholds(probs["dev"], 0.5, ml), space)
    rep = multilabel_report(Y, apply_thresholds(probs["dev"], thr, ml), space)
    print(format_report(rep, f"ensemble of {[r.name for r in run_dirs]} | thresholds '{strat}' | cross-fitted {cf}"))
    out_dir = Path(out_dir)
    save_json([int(x) for x in space], out_dir / "label_space.json")
    save_json({"at_0.5": rep05, "calibrated": rep, "strategy": strat, "crossfit_macro_f1": cf,
               "thresholds": thr_to_json(thr, space)}, out_dir / "dev_report.json")
    P_te = apply_thresholds(probs["test"], thr, ml)
    zp = write_submission(data["test"]["raw_ids"], preds_to_ids(P_te, space), out_dir / "submission", space)
    keys = ("macro_f1", "macro_p", "macro_r", "macro_f1_present", "micro_f1", "subset_acc", "pred_per_doc")
    summary = {"run": out_dir.name, "members": [r.name for r in run_dirs], "threshold_strategy": strat,
               "crossfit_macro_f1": cf, "dev_at_0.5": {k: rep05[k] for k in keys},
               "dev_calibrated": {k: rep[k] for k in keys},
               "test_pred_per_doc": round(float(P_te.sum(1).mean()), 3), "submission": str(zp)}
    save_json(summary, out_dir / "ensemble_summary.json")
    return summary

SECTION_VIEWS = {
    "first 512 (head)":        {"k": 1, "strategy": "head"},
    "middle 512":              {"k": 1, "strategy": "middle"},
    "last 512 (tail)":         {"k": 1, "strategy": "tail"},
    "head 2 chunks":           {"k": 2, "strategy": "head"},
    "tail 2 chunks":           {"k": 2, "strategy": "tail"},
    "trained view":            {},
    "uniform 4 chunks":        {"k": 4, "strategy": "uniform"},
    "uniform 8 chunks":        {"k": 8, "strategy": "uniform"},
}

def section_analysis(run_dir, data, device="cuda", views=None, max_docs=None):
    model, cfg = load_run(run_dir, device)
    space = data["space"]
    thr = thr_from_json(json.load(open(Path(run_dir) / "dev_report.json"))["thresholds"], space)
    ml = cfg["calibrate"].get("min_labels", 1)
    dev = data["dev"] if not max_docs or max_docs >= len(data["dev"]["ids"]) else \
        subset(data["dev"], np.sort(np.random.default_rng(0).choice(len(data["dev"]["ids"]), max_docs, replace=False)))
    L = cfg["chunks"]["len"]
    rows = []
    for name, ov in (views or SECTION_VIEWS).items():
        lg = predict_logits(model, dev["tokens"], cfg, device, chunk_override=ov)
        rep = multilabel_report(dev["Y"], apply_thresholds(probs_from_logits(lg), thr, ml))
        rows.append({"view": name, "chunks": f"{ov.get('k', cfg['chunks']['k'])} x {L} "
                                             f"{ov.get('strategy', cfg['chunks']['strategy'])}",
                     "macro_f1": round(rep["macro_f1"], 4), "micro_f1": round(rep["micro_f1"], 4),
                     "subset_acc": round(rep["subset_acc"], 4), "pred_per_doc": round(rep["pred_per_doc"], 2)})
    _, w = predict_logits(model, dev["tokens"], cfg, device, chunk_override={"k": 8, "strategy": "uniform"},
                          return_weights=True)
    nb = cfg["head"].get("pos_buckets", 8)
    att = None
    if w and w[0][0] is not None:
        s, c = np.zeros(nb), np.zeros(nb)
        for ww, pos in w:
            for a, b in zip(ww, pos):
                s[b] += a * len(ww); c[b] += 1
        att = {f"{i / nb:.2f}-{(i + 1) / nb:.2f}": round(float(s[i] / c[i]), 3) for i in range(nb) if c[i]}
    out = {"run": Path(run_dir).name, "dev_docs": len(dev["ids"]), "views": rows,
           "chunk_attention_by_position(uniform 8 chunks)": att}
    save_json(out, Path(run_dir) / "section_analysis.json")
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    return out

def label_analysis(run_dir, data, top=15):
    space = data["space"]
    rep = json.load(open(Path(run_dir) / "dev_report.json"))
    per = rep["calibrated"]["per_label"]
    freq = Counter(s for labs in data["train"]["labels"] for s in labs)
    buckets = frequency_buckets(per, freq)
    pred = json.load(open(Path(run_dir) / "dev_predictions.json"))
    missed, wrong, pairs = Counter(), Counter(), Counter()
    by_card = {}
    for r in pred:
        g, p = set(r["gold"]), set(r["pred"])
        missed.update(g - p); wrong.update(p - g)
        for a in g - p:
            for b in p - g:
                pairs[(a, b)] += 1
        k = min(len(g), 4)
        by_card.setdefault(k, []).append(g == p)
    worst = sorted([x for x in per if x["gold"] > 0], key=lambda x: (x["f1"], -x["gold"]))[:top]
    out = {"run": Path(run_dir).name, "f1_by_train_frequency": buckets,
           "statutes_never_predicted_correctly": sum(1 for x in per if x["gold"] > 0 and x["f1"] == 0),
           "worst_statutes": [{**x, "train_docs": freq.get(x["id"], 0)} for x in worst],
           "most_missed": missed.most_common(top), "most_false_positive": wrong.most_common(top),
           "confused_pairs(gold_missed->predicted_instead)": [[list(k), v] for k, v in pairs.most_common(top)],
           "exact_match_by_gold_count": {f"{k}{'+' if k == 4 else ''} statutes": {"docs": len(v), "subset_acc": round(float(np.mean(v)), 4)}
                                         for k, v in sorted(by_card.items())},
           "space_size": len(space)}
    save_json(out, Path(run_dir) / "label_analysis.json")
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
        print(ensemble(a.ensemble, a.out, load_data(resolve_paths(cfg)), cfg.get("calibrate")))
    else:
        run(cfg)

if __name__ == "__main__":
    main()
