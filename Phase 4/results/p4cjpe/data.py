"""Phase-4 data: download, load, validate, tokenise once (cached), statistics.

Layout (HF dataset Exploration-Lab/CS-781-Capstone, folder Phase_4_CJPE):
    Train/train.json, Train/dev.json, Test/test.json
A case is {"id": "case_001", "text": "...", "label": 0 | 1} (no label in test.json).
The loader also accepts a dict {id: {...}}, JSON lines, a {"data": [...]} wrapper, "text" given as a list of
paragraphs, and the label under "label"/"labels"/"y"/"outcome"/... so a small format difference does not break it.

Tokenising ~0.8 GB of judgments with the pure-Python Phase-1 BPE takes a few minutes, so it is done ONCE
(multiprocessing) and stored as uint16 token arrays in cache_dir; every run (and both parallel GPU jobs) reuses it.
Cached form per split: {"ids": [str], "raw_ids": [original id], "labels": np.int64 array | None,
                        "tokens": [np.uint16 array], "n_chars": np.int64 array}
"""
import hashlib
import json
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

HF_REPO = "Exploration-Lab/CS-781-Capstone"
HF_SUBDIR = "Phase_4_CJPE"
SPLITS = ("train", "dev", "test")
ID_KEYS = ("id", "case_id", "doc_id", "uid", "name", "case")
TEXT_KEYS = ("text", "document", "doc", "case_text", "judgment", "content", "body")
LABEL_KEYS = ("label", "labels", "y", "outcome", "judgment_label", "decision", "target", "pred")


def download(local_dir, token=None):
    """Gated dataset: needs a HF token (Kaggle: Add-ons -> Secrets -> HF_TOKEN). Returns the Phase_4_CJPE path."""
    from huggingface_hub import snapshot_download
    token = token or os.environ.get("HF_TOKEN")
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=local_dir, token=token,
                      allow_patterns=[f"{HF_SUBDIR}/*"])
    return Path(local_dir) / HF_SUBDIR


def find_data_dir(candidates=()):
    """First folder that contains Train/train.json and Test/test.json (given paths first, then /kaggle/input)."""
    roots = [Path(c) for c in candidates if c] + [Path("/kaggle/input")]
    for r in roots:
        if not r.exists():
            continue
        if (r / "Train" / "train.json").exists() and (r / "Test" / "test.json").exists():
            return r
        for p in r.rglob("train.json"):
            if p.parent.name == "Train" and (p.parent.parent / "Test" / "test.json").exists():
                return p.parent.parent
    return None


def split_path(data_dir, split):
    return Path(data_dir) / ("Test" if split == "test" else "Train") / f"{split}.json"


def _read_json(path):
    txt = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(txt)
    except json.JSONDecodeError:                                   # JSON lines
        return [json.loads(line) for line in txt.splitlines() if line.strip()]


def _records(raw):
    if isinstance(raw, dict):
        for k in ("data", "documents", "docs", "cases", "examples"):
            if isinstance(raw.get(k), list):
                return raw[k]
        recs = []
        for k, v in raw.items():
            if isinstance(v, dict):
                recs.append({"id": k, **v} if not any(x in v for x in ID_KEYS) else v)
            elif isinstance(v, str):
                recs.append({"id": k, "text": v})
            else:
                raise ValueError(f"unexpected value type for case {k}: {type(v)}")
        return recs
    if isinstance(raw, list):
        return raw
    raise ValueError(f"unexpected top-level JSON type {type(raw)}")


def _first(r, keys):
    for k in keys:
        if k in r and r[k] is not None:
            return k, r[k]
    return None, None


def _label(v, where):
    if isinstance(v, list) and len(v) == 1:
        v = v[0]
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, np.integer)) and int(v) in (0, 1):
        return int(v)
    if isinstance(v, float) and v in (0.0, 1.0):
        return int(v)
    if isinstance(v, str) and v.strip() in ("0", "1"):
        return int(v.strip())
    raise ValueError(f"{where}: label {v!r} is not 0/1")


def normalise_text(t):
    """Paragraph lists are joined; every whitespace run becomes one space (judgments contain many blank lines,
    which would otherwise waste the token budget)."""
    if isinstance(t, list):
        t = " ".join("" if x is None else str(x) for x in t)
    return " ".join(str(t).split())


def norm_case(r, where, need_label):
    if not isinstance(r, dict):
        raise ValueError(f"{where}: record is {type(r)}, expected an object")
    _, rid = _first(r, ID_KEYS)
    _, text = _first(r, TEXT_KEYS)
    lk, lab = _first(r, LABEL_KEYS)
    if rid is None:
        raise ValueError(f"{where}: case without an id field: keys {sorted(r)}")
    if text is None:
        raise ValueError(f"{where}: case {rid} has no text field: keys {sorted(r)}")
    if need_label and lab is None:
        raise ValueError(f"{where}: case {rid} has no label field: keys {sorted(r)}")
    return {"id": str(rid), "raw_id": rid, "text": normalise_text(text),
            "label": None if (lab is None or not need_label) else _label(lab, f"{where}/{rid}")}


def load_split(data_dir, split, max_docs=None):
    p = split_path(data_dir, split)
    if not p.exists():
        raise FileNotFoundError(p)
    recs = _records(_read_json(p))
    if max_docs:
        recs = recs[:max_docs]
    cases = [norm_case(r, f"{p.name}[{i}]", split != "test") for i, r in enumerate(recs)]
    seen, out, dup, conflict = {}, [], 0, 0
    for c in cases:
        if c["id"] in seen:
            dup += 1
            if split == "test":
                raise ValueError(f"test.json: id {c['id']} appears twice; one prediction per id is required")
            conflict += int(seen[c["id"]]["label"] != c["label"])
            continue
        seen[c["id"]] = c
        out.append(c)
    if dup:
        print(f"[data] {split}: dropped {dup} repeated ids ({conflict} with a different label; first one kept)")
    return out


def describe_raw(data_dir, n_chars=300):
    """Raw structure of every file (first record), to confirm the format before training."""
    for split in SPLITS:
        p = split_path(data_dir, split)
        raw = _read_json(p)
        recs = _records(raw)
        first = recs[0] if recs else {}
        short = {k: (v[:n_chars] + "...") if isinstance(v, str) and len(v) > n_chars else v for k, v in first.items()}
        print(f"{p.name}: {p.stat().st_size / 2**20:.1f} MB | top-level {type(raw).__name__} | {len(recs)} records | "
              f"keys {sorted(first)} | {json.dumps(short, default=str)[:n_chars + 200]}")


# ------------------------------------------------------------------------------------------- tokenisation
_TOK = None


def _init_worker(tok_path):
    global _TOK
    from .tokenizer import BPETokenizer
    _TOK = BPETokenizer.load(tok_path)


def _encode(text):
    ids = _TOK.encode(" " + text) if text else []
    if not ids:
        ids = _TOK.encode(" .")
    return np.asarray(ids, dtype=np.uint16)


def tokenizer_fingerprint(tok_path):
    from .tokenizer import BPETokenizer
    tok = BPETokenizer.load(tok_path)
    assert tok.vocab_size <= 65535, "uint16 token cache needs vocab <= 65535"
    return tok.fingerprint()


def cache_path(cache_dir, split, tok_path, max_docs=None):
    tag = tokenizer_fingerprint(tok_path) + (f"_n{max_docs}" if max_docs else "")
    return Path(cache_dir) / f"cjpe_{split}_{tag}.pkl"


def tokenise_split(cases, tok_path, n_proc=4, log=print):
    texts = [c["text"] for c in cases]
    if n_proc and n_proc > 1 and len(texts) > 64:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(n_proc, initializer=_init_worker, initargs=(tok_path,)) as pool:
            toks = list(pool.imap(_encode, texts, chunksize=32))
    else:
        _init_worker(tok_path)
        toks = [_encode(t) for t in texts]
    return toks


def build_cache(data_dir, tok_path, cache_dir, n_proc=4, max_docs=None, log=print, splits=SPLITS):
    """Reads the JSON files and stores the tokenised splits (skips splits that are already cached)."""
    import time
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    for split in splits:
        cp = cache_path(cache_dir, split, tok_path, max_docs)
        if cp.exists():
            continue
        t0 = time.time()
        cases = load_split(data_dir, split, max_docs)
        toks = tokenise_split(cases, tok_path, n_proc, log)
        obj = {"ids": [c["id"] for c in cases], "raw_ids": [c["raw_id"] for c in cases],
               "labels": None if split == "test" else np.array([c["label"] for c in cases], dtype=np.int64),
               "tokens": toks, "n_chars": np.array([len(c["text"]) for c in cases], dtype=np.int64),
               "source": str(split_path(data_dir, split))}
        tmp = cp.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cp)
        n_tok = sum(len(t) for t in toks)
        log(f"[data] {split}: {len(cases)} cases, {n_tok / 1e6:.1f}M tokens, tokenised in {time.time() - t0:.0f}s -> {cp.name}")
        del cases, toks, obj


def load_cached(data_dir, tok_path, cache_dir, max_docs=None, n_proc=4, log=print):
    """{"train"/"dev"/"test": cached split}. Builds the cache when it is missing."""
    build_cache(data_dir, tok_path, cache_dir, n_proc, max_docs, log)
    out = {}
    for split in SPLITS:
        with open(cache_path(cache_dir, split, tok_path, max_docs), "rb") as f:
            out[split] = pickle.load(f)
    return out


def text_fingerprints(cases):
    return {hashlib.sha1(c["text"].encode("utf-8")).hexdigest(): c["id"] for c in cases}


def stats(data):
    """Statistics for the report: cases, label balance, token lengths (how much of a judgment fits in 512 tokens)."""
    out = {}
    for split in SPLITS:
        d = data[split]
        lens = np.array([len(t) for t in d["tokens"]])
        rec = {"cases": len(lens), "tokens": {q: int(np.percentile(lens, p)) for q, p in
                                              (("min", 0), ("p10", 10), ("p50", 50), ("mean", None), ("p90", 90), ("max", 100))
                                              if p is not None} if len(lens) else {}}
        if len(lens):
            rec["tokens"]["mean"] = int(lens.mean())
            rec["chars_per_token"] = round(float(d["n_chars"].sum() / max(1, lens.sum())), 2)
            for L in (512, 2048, 4096):
                rec[f"fits_in_{L}"] = round(float((lens <= L).mean()), 3)
        if d["labels"] is not None:
            c = Counter(d["labels"].tolist())
            rec["labels"] = {0: c.get(0, 0), 1: c.get(1, 0)}
            rec["share_label_1"] = round(c.get(1, 0) / max(1, len(lens)), 3)
            if len(lens):
                rec["median_tokens_by_label"] = {k: int(np.median(lens[d["labels"] == k])) if (d["labels"] == k).any() else 0
                                                 for k in (0, 1)}
        out[split] = rec
    tr = set(data["train"]["ids"])
    out["id_overlap"] = {"train&dev": len(tr & set(data["dev"]["ids"])), "train&test": len(tr & set(data["test"]["ids"]))}
    return out
