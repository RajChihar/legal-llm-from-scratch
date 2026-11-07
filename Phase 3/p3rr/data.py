import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

HF_REPO = "Exploration-Lab/CS-781-Capstone"
HF_SUBDIR = "Phase_3_RR"
DOMAINS = ("CL", "IT")
SPLITS = ("train", "dev", "test")

def download(local_dir, token=None):
    from huggingface_hub import snapshot_download
    token = token or os.environ.get("HF_TOKEN")
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=local_dir, token=token,
                      allow_patterns=[f"{HF_SUBDIR}/*"])
    return Path(local_dir) / HF_SUBDIR

def find_data_dir(candidates=()):
    roots = [Path(c) for c in candidates if c] + [Path("/kaggle/input")]
    for r in roots:
        if not r.exists():
            continue
        if (r / "Train" / "label_vocab.json").exists():
            return r
        for p in r.rglob("label_vocab.json"):
            if p.parent.name == "Train":
                return p.parent.parent
    return None

def split_path(data_dir, split, dom):
    d = Path(data_dir)
    return d / ("Test" if split == "test" else "Train") / f"{dom}_{split}.json"

def _read_json(path):
    txt = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return [json.loads(line) for line in txt.splitlines() if line.strip()]

def _records(raw):
    if isinstance(raw, dict):
        for k in ("data", "documents", "docs"):
            if isinstance(raw.get(k), list):
                return raw[k]
        recs = []
        for k, v in raw.items():
            if isinstance(v, dict):
                recs.append({"id": k, **v})
            elif isinstance(v, list):
                recs.append({"id": k, "text": v})
            else:
                raise ValueError(f"unexpected value type for document {k}: {type(v)}")
        return recs
    if isinstance(raw, list):
        return raw
    raise ValueError(f"unexpected top-level JSON type {type(raw)}")

def _first(r, keys):
    for k in keys:
        if k in r and r[k] is not None:
            return r[k]
    return None

def _norm_doc(r, dom, where):
    did = _first(r, ("id", "doc_id", "document_id", "docid"))
    sents = _first(r, ("text", "sentences", "sents"))
    labels = _first(r, ("labels", "label", "roles", "rhetorical_roles"))
    if did is None:
        raise ValueError(f"{where}: document without an id field: keys {sorted(r)}")
    if not isinstance(sents, list):
        raise ValueError(f"{where}: document {did}: 'text' must be a list of sentences, got {type(sents)}")
    if labels is not None and len(labels) == 0 and len(sents) > 0:
        labels = None
    if labels is not None and len(labels) != len(sents):
        raise ValueError(f"{where}: document {did}: {len(sents)} sentences but {len(labels)} labels")
    return {"id": str(did), "domain": dom, "sents": ["" if s is None else str(s) for s in sents],
            "labels": None if labels is None else [str(l) for l in labels]}

def load_vocab(path):
    raw = _read_json(path)
    if isinstance(raw, dict):
        for k in ("labels", "label_vocab", "vocab"):
            if k in raw:
                return load_vocab_obj(raw[k])
    return load_vocab_obj(raw)

def load_vocab_obj(raw):
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, dict):
        vals = list(raw.values())
        if all(isinstance(v, int) for v in vals):
            return [k for k, _ in sorted(raw.items(), key=lambda kv: kv[1])]
        if all(isinstance(v, str) for v in vals):
            return [v for _, v in sorted(raw.items(), key=lambda kv: int(kv[0]))]
    raise ValueError(f"cannot parse label_vocab.json: {raw!r:.200}")

def load_phase3(data_dir, max_docs=None):
    data_dir = Path(data_dir)
    labels = load_vocab(data_dir / "Train" / "label_vocab.json")
    assert len(set(labels)) == len(labels), f"duplicate labels in label_vocab.json: {labels}"
    out = {"labels": labels, "data_dir": str(data_dir)}
    for split in SPLITS:
        out[split] = {}
        for dom in DOMAINS:
            p = split_path(data_dir, split, dom)
            if not p.exists():
                raise FileNotFoundError(p)
            recs = _records(_read_json(p))
            if max_docs:
                recs = recs[:max_docs]
            for r in recs:
                d = _norm_doc(r, dom, p.name)
                if d["id"] in out[split]:
                    raise ValueError(f"document id {d['id']} appears twice in the {split} split "
                                     f"({out[split][d['id']]['domain']} and {dom}): one prediction.json cannot hold both")
                out[split][d["id"]] = d
    lab = set(labels)
    for split in ("train", "dev"):
        bad = Counter(l for d in out[split].values() for l in (d["labels"] or []) if l not in lab)
        assert not bad, f"{split} labels not in label_vocab.json: {dict(bad)}"
        missing = [k for k, d in out[split].items() if d["labels"] is None]
        assert not missing, f"{split} documents without labels: {missing[:5]}"
    return out

def describe_raw(data_dir, n_chars=300):
    data_dir = Path(data_dir)
    print("label_vocab.json:", json.dumps(_read_json(data_dir / "Train" / "label_vocab.json"))[:400])
    for split in SPLITS:
        for dom in DOMAINS:
            p = split_path(data_dir, split, dom)
            raw = _read_json(p)
            recs = _records(raw)
            first = recs[0] if recs else {}
            print(f"{p.name}: top-level {type(raw).__name__}, {len(recs)} docs, keys {sorted(first)} | "
                  f"{json.dumps(first)[:n_chars]}")

def stats(data, tok=None):
    labels = data["labels"]
    out = {}
    for split in SPLITS:
        for dom in DOMAINS:
            docs = [d for d in data[split].values() if d["domain"] == dom]
            n_s = [len(d["sents"]) for d in docs]
            rec = {"docs": len(docs), "sentences": int(sum(n_s)),
                   "sents_per_doc": {"mean": round(float(np.mean(n_s)), 1) if n_s else 0, "min": min(n_s or [0]),
                                     "max": max(n_s or [0])}}
            words = [len(s.split()) for d in docs for s in d["sents"]]
            if words:
                rec["words_per_sent"] = {q: int(np.percentile(words, p)) for q, p in (("p50", 50), ("p90", 90), ("p99", 99), ("max", 100))}
                rec["empty_sentences"] = sum(1 for d in docs for s in d["sents"] if not s.strip())
            if tok is not None and words:
                lens = [len(tok.encode(" " + s.strip())) for d in docs[:40] for s in d["sents"]]
                rec["tokens_per_sent(first 40 docs)"] = {q: int(np.percentile(lens, p)) for q, p in (("p50", 50), ("p90", 90), ("p99", 99), ("max", 100))}
            if split != "test":
                c = Counter(l for d in docs for l in d["labels"])
                rec["label_counts"] = {l: c.get(l, 0) for l in labels}
            out[f"{split}/{dom}"] = rec
    pos = {l: [] for l in labels}
    trans = Counter()
    for d in data["train"].values():
        n = len(d["labels"])
        for i, l in enumerate(d["labels"]):
            pos[l].append(i / max(1, n - 1))
            if i:
                trans[(d["labels"][i - 1], l)] += 1
    out["mean_relative_position(train)"] = {l: round(float(np.mean(v)), 2) for l, v in pos.items() if v}
    same = sum(v for (a, b), v in trans.items() if a == b)
    out["label_repeats_previous(train)"] = round(same / max(1, sum(trans.values())), 3)
    out["top_transitions(train, a!=b)"] = [f"{a}->{b}: {v}" for (a, b), v in trans.most_common(40) if a != b][:12]
    return out
