import argparse
import json
from pathlib import Path

import numpy as np

from . import data as D
from .metrics import format_table, full_report
from .train import (DEFAULTS, decode_all, get_tokenizer, load_run, load_scores, merge, predict_scores, prepare,
                    resolve_paths, run, save_json, set_by_path, write_submission)

def eda(cfg):
    D.describe_raw(cfg["data_dir"])
    data = D.load_phase3(cfg["data_dir"])
    print("labels:", data["labels"])
    tok = get_tokenizer(cfg) if cfg["backbone"].get("ckpt") else None
    print(json.dumps(D.stats(data, tok), indent=1))
    return data

def repredict(run_dir, data_dir=None, device="cuda"):
    model, cfg, labels = load_run(run_dir, device)
    data = D.load_phase3(data_dir or cfg["data_dir"])
    prep = prepare(data["test"], get_tokenizer(cfg), labels, cfg)
    pred = decode_all(model, predict_scores(model, prep, cfg, 16, device))
    return write_submission(pred, data["test"], labels, Path(run_dir) / "submission")

def ensemble(run_dirs, out_dir, data_dir=None, use_refit=False):
    run_dirs = [Path(r) for r in run_dirs]
    cfg = json.load(open(run_dirs[0] / "config.json"))
    labels = json.load(open(run_dirs[0] / "labels.json"))
    data = D.load_phase3(data_dir or cfg["data_dir"])
    out = {}
    for split, fname in (("dev", "dev_logprobs.npz"), ("test", "test_refit_logprobs.npz" if use_refit else "test_logprobs.npz")):
        probs = None
        for r in run_dirs:
            assert json.load(open(r / "labels.json")) == labels
            s = load_scores(r / fname)
            p = {d: np.exp(v) for d, v in s.items()}
            probs = p if probs is None else {d: probs[d] + p[d] for d in probs}
        out[split] = {d: [labels[k] for k in v.argmax(1)] for d, v in probs.items()}
    gold = {k: d["labels"] for k, d in data["dev"].items()}
    rep = full_report(gold, out["dev"], labels, {k: d["domain"] for k, d in data["dev"].items()})
    print(format_table(rep["all"]))
    out_dir = Path(out_dir)
    save_json(rep, out_dir / "dev_report.json")
    zp = write_submission(out["test"], data["test"], labels, out_dir / "submission")
    summary = {"run": out_dir.name, "members": [r.name for r in run_dirs], "refit_members": use_refit,
               "dev": {k: rep["all"][k] for k in ("macro_f1", "macro_p", "macro_r", "macro_f1_vocab", "accuracy")},
               "dev_CL_macro_f1": rep.get("CL", {}).get("macro_f1"), "dev_IT_macro_f1": rep.get("IT", {}).get("macro_f1"),
               "submission": str(zp)}
    save_json(summary, out_dir / "ensemble_summary.json")
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--set", action="append", default=[], help="dotted.key=value override")
    ap.add_argument("--eda", action="store_true")
    ap.add_argument("--predict", default="", help="run dir: rebuild model and write a test submission")
    ap.add_argument("--ensemble", nargs="+", default=[], help="run dirs whose probabilities are averaged")
    ap.add_argument("--out", default="runs/ENSEMBLE")
    a = ap.parse_args()
    cfg = merge(DEFAULTS, json.load(open(a.config)) if a.config else {})
    for kv in a.set:
        k, v = kv.split("=", 1)
        set_by_path(cfg, k, v)
    if a.eda:
        eda(resolve_paths(cfg))
    elif a.predict:
        print(repredict(a.predict, cfg["data_dir"] or None))
    elif a.ensemble:
        print(ensemble(a.ensemble, a.out, cfg["data_dir"] or None))
    else:
        run(cfg)

if __name__ == "__main__":
    main()
