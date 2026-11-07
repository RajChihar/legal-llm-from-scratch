"""CLI.

    python -m p2lner.run --config configs/E1_C_full.json --set data_dir=/kaggle/working/p2data/Phase_2_LNER \
                         --set backbone.ckpt=/kaggle/input/.../model_final.pt
    python -m p2lner.run --eda --config configs/E1_C_full.json      # data statistics + BIO reachability only
    python -m p2lner.run --predict runs/E1_C_full                    # re-predict test from a saved run
"""
import argparse
import json
from pathlib import Path

from . import data as D
from .spans import TagSet, reachability
from .train import DEFAULTS, decode_all, get_tokenizer, load_run, merge, predict_scores, prepare, resolve_paths, run, set_by_path, write_submission


def eda(cfg):
    data = D.load_phase2(cfg["data_dir"])
    tagset = TagSet(data["labels"])
    train_docs, _ = D.make_split(data, mode="none")
    print("labels:", data["labels"])
    print("gold checks:", D.validate_gold(train_docs, data["labels"]))
    for fine in (False, True):
        cfg["tokenizer"]["fine_punct"] = fine
        tok = get_tokenizer(cfg)
        sc, bad, skipped = reachability(train_docs, tok, tagset)
        tok.save_disk_cache()
        print(f"\n== tokenisation fine_punct={fine}: BIO round-trip macro-F1 {sc['macro_f1']:.4f} | micro {sc['micro_f1']:.4f} "
              f"| unreachable {len(bad)} | overlaps dropped {skipped}")
        for b in bad[:10]:
            print("   ", b)
    print(json.dumps(D.eda(train_docs, data["labels"], tok), indent=1))
    print("test docs:", len(data["test"]))


def repredict(run_dir, data_dir=None, device="cuda"):
    model, cfg, tagset, trim_rules = load_run(run_dir, device)
    data = D.load_phase2(data_dir or cfg["data_dir"])
    tok = get_tokenizer(cfg)
    docs = {k: {"text": v} for k, v in data["test"].items()}
    prep = prepare(docs, tok, tagset)
    sc = predict_scores(model, prep, cfg["window"]["max_len"], cfg["window"]["stride"], 16, device)
    pred = decode_all(model, docs, prep, sc, cfg, trim_rules)
    return write_submission(pred, list(data["test"]), data["labels"], Path(run_dir) / "submission")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--set", action="append", default=[], help="dotted.key=value override")
    ap.add_argument("--eda", action="store_true")
    ap.add_argument("--predict", default="", help="run dir: rebuild model and write a test submission")
    a = ap.parse_args()
    cfg = merge(DEFAULTS, json.load(open(a.config)) if a.config else {})
    for kv in a.set:
        k, v = kv.split("=", 1)
        set_by_path(cfg, k, v)
    if a.eda:
        eda(resolve_paths(cfg))
    elif a.predict:
        print(repredict(a.predict, cfg["data_dir"] or None))
    else:
        run(cfg)


if __name__ == "__main__":
    main()
