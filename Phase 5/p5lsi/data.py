import hashlib
import json
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

HF_REPO = "Exploration-Lab/CS-781-Capstone"
HF_SUBDIR = "Phase_5_LSI"
SPLITS = ("train", "dev", "test")
ID_KEYS = ("id", "case_id", "doc_id", "uid", "name", "case")
TEXT_KEYS = ("text", "document", "doc", "case_text", "facts", "judgment", "content", "body")
LABEL_KEYS = ("labels", "label", "statutes", "statute_ids", "statute", "y", "target", "pred")

def download(local_dir, token=None):
    from huggingface_hub import snapshot_download
    token = token or os.environ.get("HF_TOKEN")
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=local_dir, token=token,
                      allow_patterns=[f"{HF_SUBDIR}/*"])
    return Path(local_dir) / HF_SUBDIR

def find_data_dir(candidates=()):
    def ok(d):
        return (d / "Train" / "train.json").exists() and (d / "Test" / "test.json").exists()
    for c in candidates:
        if c and ok(Path(c)):
            return Path(c)
    for r in [Path(c) for c in candidates if c] + [Path("/kaggle/input")]:
        if not r.exists():
            continue
        for p in r.rglob("train.json"):
            d = p.parent.parent
            if p.parent.name == "Train" and ok(d) and ("lsi" in str(d).lower() or "phase_5" in str(d).lower()
                                                       or "phase5" in str(d).lower() or "phase-5" in str(d).lower()):
                return d
    return None

def split_path(data_dir, split):
    return Path(data_dir) / ("Test" if split == "test" else "Train") / f"{split}.json"

def _read_json(path):
    txt = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
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

def _statute_id(v, where):
    if isinstance(v, bool) or v is None:
        raise ValueError(f"{where}: statute id {v!r} is not an integer")
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        return int(v.strip())
    raise ValueError(f"{where}: statute id {v!r} is not an integer")

def _labels(v, where):
    if isinstance(v, str):
        t = v.strip().strip("[]")
        v = [x for x in t.replace(";", ",").split(",") if x.strip()] if t else []
    elif not isinstance(v, (list, tuple, np.ndarray)):
        v = [v]
    return sorted({_statute_id(x, where) for x in v})

def normalise_text(t):
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
            "labels": None if (lab is None or not need_label) else _labels(lab, f"{where}/{rid}")}

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
            conflict += int(seen[c["id"]]["labels"] != c["labels"] or seen[c["id"]]["text"] != c["text"])
            continue
        seen[c["id"]] = c
        out.append(c)
    if dup:
        print(f"[data] {split}: dropped {dup} repeated ids ({conflict} with different labels/text; first one kept)")
    return out

def describe_raw(data_dir, n_chars=300):
    for split in SPLITS:
        p = split_path(data_dir, split)
        raw = _read_json(p)
        recs = _records(raw)
        first = recs[0] if recs else {}
        short = {k: (v[:n_chars] + "...") if isinstance(v, str) and len(v) > n_chars else v for k, v in first.items()}
        print(f"{p.name}: {p.stat().st_size / 2**20:.1f} MB | top-level {type(raw).__name__} | {len(recs)} records | "
              f"keys {sorted(first)} | {json.dumps(short, default=str)[:n_chars + 200]}")

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
    return Path(cache_dir) / f"lsi_{split}_{tag}.pkl"

def tokenise_split(cases, tok_path, n_proc=4, log=print):
    texts = [c["text"] for c in cases]
    if n_proc and n_proc > 1 and len(texts) > 64:
        import gc
        import multiprocessing as mp
        gc.collect(); gc.freeze()
        try:
            with mp.get_context("fork").Pool(n_proc, initializer=_init_worker, initargs=(tok_path,)) as pool:
                toks = list(pool.imap(_encode, texts, chunksize=32))
        finally:
            gc.unfreeze()
    else:
        _init_worker(tok_path)
        toks = [_encode(t) for t in texts]
    return toks

def build_cache(data_dir, tok_path, cache_dir, n_proc=4, max_docs=None, log=print, splits=SPLITS):
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
               "labels": None if split == "test" else [c["labels"] for c in cases],
               "tokens": toks, "n_chars": np.array([len(c["text"]) for c in cases], dtype=np.int64),
               "source": str(split_path(data_dir, split))}
        tmp = cp.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cp)
        n_tok = sum(len(t) for t in toks)
        log(f"[data] {split}: {len(cases)} documents, {n_tok / 1e6:.1f}M tokens, tokenised in {time.time() - t0:.0f}s -> {cp.name}")
        del cases, toks, obj

def load_cached(data_dir, tok_path, cache_dir, max_docs=None, n_proc=4, log=print):
    build_cache(data_dir, tok_path, cache_dir, n_proc, max_docs, log)
    out = {}
    for split in SPLITS:
        with open(cache_path(cache_dir, split, tok_path, max_docs), "rb") as f:
            out[split] = pickle.load(f)
    return out

def text_fingerprints(cases):
    return {hashlib.sha1(c["text"].encode("utf-8")).hexdigest(): c["id"] for c in cases}

def label_space(data):
    ids = set()
    for split in ("train", "dev"):
        for labs in data[split]["labels"] or []:
            ids.update(labs)
    assert ids, "no statute ids found in train/dev"
    return sorted(ids)

def multi_hot(label_lists, space):
    pos = {s: k for k, s in enumerate(space)}
    Y = np.zeros((len(label_lists), len(space)), dtype=np.float32)
    for i, labs in enumerate(label_lists):
        for s in labs:
            if s in pos:
                Y[i, pos[s]] = 1.0
    return Y

def stats(data):
    out = {}
    space = label_space(data)
    for split in SPLITS:
        d = data[split]
        lens = np.array([len(t) for t in d["tokens"]])
        rec = {"documents": len(lens)}
        if len(lens):
            rec["tokens"] = {q: int(np.percentile(lens, p)) for q, p in
                             (("min", 0), ("p10", 10), ("p50", 50), ("p90", 90), ("max", 100))}
            rec["tokens"]["mean"] = int(lens.mean())
            rec["chars_per_token"] = round(float(d["n_chars"].sum() / max(1, lens.sum())), 2)
            for L in (512, 1024, 2048, 4096):
                rec[f"fits_in_{L}"] = round(float((lens <= L).mean()), 3)
        if d["labels"] is not None:
            card = np.array([len(x) for x in d["labels"]])
            freq = Counter(s for x in d["labels"] for s in x)
            rec["labels_per_doc"] = {"mean": round(float(card.mean()), 3), "max": int(card.max()),
                                     "share_with_0": round(float((card == 0).mean()), 4),
                                     "share_with_1": round(float((card == 1).mean()), 3),
                                     "share_with_3plus": round(float((card >= 3).mean()), 3)}
            rec["distinct_statutes"] = len(freq)
            f = np.array(sorted(freq.values(), reverse=True))
            rec["statute_freq"] = {"max": int(f[0]), "median": int(np.median(f)), "min": int(f[-1]),
                                   "statutes_with_<10_docs": int((f < 10).sum()),
                                   "share_of_assignments_top10": round(float(f[:10].sum() / f.sum()), 3)}
            rec["most_frequent"] = freq.most_common(10)
            pairs = Counter((a, b) for x in d["labels"] for i, a in enumerate(x) for b in x[i + 1:])
            rec["top_cooccurring_pairs"] = [[list(p), c] for p, c in pairs.most_common(10)]
        out[split] = rec
    out["label_space"] = {"size": len(space), "min_id": space[0], "max_id": space[-1],
                          "dev_only_ids": sorted(set(s for x in data["dev"]["labels"] for s in x)
                                                 - set(s for x in data["train"]["labels"] for s in x))}
    tr = set(data["train"]["ids"])
    out["id_overlap"] = {"train&dev": len(tr & set(data["dev"]["ids"])), "train&test": len(tr & set(data["test"]["ids"]))}
    return out
