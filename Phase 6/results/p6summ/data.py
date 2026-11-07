"""Phase-6 data: download, load, validate, held-out dev split, tokenise once (cached), statistics.

Layout (HF dataset Exploration-Lab/CS-781-Capstone, folder Phase_6_Summ):
    Train/train.json   [{"id": "doc_001", "document": "...", "summary": "Reference summary ..."}, ...]
    Test/test.json     [{"id": "doc_101", "document": "..."}, ...]
No public dev split: `dev_size` training documents are held out by a seeded hash of their id (the same split for
every run and every session), the rest is used for training.
The loader also accepts a dict {id: {...}}, JSON lines, a {"data": [...]} wrapper, the text under
"document"/"text"/"judgment"/..., the summary under "summary"/"abstract"/"headnote"/..., and paragraph/sentence lists
(joined), so a small format difference does not break it.

Documents are split into sentences and every sentence is tokenised separately (the space before a sentence stays
with it, so the concatenation is exactly the tokenisation of the whole document). Stored per document:
    tokens     uint16 array, the whole document
    sent_end   int32 array, token index where each sentence ends (cumulative)
    sent_score float32 array, salience of each sentence (TF-IDF centroid similarity, see `sentence_scores`)
Tokenising with the pure-Python Phase-1 BPE is the slowest CPU step, so it is done ONCE (multiprocessing) and
stored in cache_dir; every run (and both parallel GPU jobs) reuses it.
"""
import hashlib
import json
import math
import os
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np

HF_REPO = "Exploration-Lab/CS-781-Capstone"
HF_SUBDIR = "Phase_6_Summ"
ID_KEYS = ("id", "doc_id", "case_id", "uid", "name")
DOC_KEYS = ("document", "text", "doc", "judgment", "judgement", "source", "article", "case_text", "content", "body")
SUM_KEYS = ("summary", "abstract", "headnote", "headnotes", "reference", "target", "highlights", "summ")
SPLITS = ("train", "dev", "test")


def download(local_dir, token=None):
    """Gated dataset: needs a HF token (Kaggle: Add-ons -> Secrets -> HF_TOKEN). Returns the Phase_6_Summ path."""
    from huggingface_hub import snapshot_download
    token = token or os.environ.get("HF_TOKEN")
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=local_dir, token=token,
                      allow_patterns=[f"{HF_SUBDIR}/*"])
    return Path(local_dir) / HF_SUBDIR


def find_data_dir(candidates=()):
    """First folder that contains Train/train.json and Test/test.json: the given paths first, then a folder under
    /kaggle/input whose path names Phase 6 / Summ (so an attached dataset of another phase is never used)."""
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
            low = str(d).lower()
            if p.parent.name == "Train" and ok(d) and ("summ" in low or "phase_6" in low or "phase6" in low
                                                       or "phase-6" in low):
                return d
    return None


def file_path(data_dir, split):
    return Path(data_dir) / ("Test/test.json" if split == "test" else "Train/train.json")


def _read_json(path):
    txt = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(txt)
    except json.JSONDecodeError:                                   # JSON lines
        return [json.loads(line) for line in txt.splitlines() if line.strip()]


def _records(raw):
    if isinstance(raw, dict):
        for k in ("data", "documents", "docs", "examples", "cases"):
            if isinstance(raw.get(k), list):
                return raw[k]
        recs = []
        for k, v in raw.items():
            if isinstance(v, dict):
                recs.append({"id": k, **v} if not any(x in v for x in ID_KEYS) else v)
            elif isinstance(v, str):
                recs.append({"id": k, "document": v})
            else:
                raise ValueError(f"unexpected value type for document {k}: {type(v)}")
        return recs
    if isinstance(raw, list):
        return raw
    raise ValueError(f"unexpected top-level JSON type {type(raw)}")


def _first(r, keys):
    for k in keys:
        if k in r and r[k] is not None:
            return k, r[k]
    return None, None


def normalise_text(t):
    """Paragraph / sentence lists are joined; every whitespace run becomes one space (judgments contain many blank
    lines and indents, which would otherwise waste the token budget)."""
    if isinstance(t, list):
        t = " ".join("" if x is None else str(x) for x in t)
    return " ".join(str(t).split())


def norm_doc(r, where, need_summary):
    if not isinstance(r, dict):
        raise ValueError(f"{where}: record is {type(r)}, expected an object")
    _, rid = _first(r, ID_KEYS)
    _, doc = _first(r, DOC_KEYS)
    _, summ = _first(r, SUM_KEYS)
    if rid is None:
        raise ValueError(f"{where}: document without an id field: keys {sorted(r)}")
    if doc is None:
        raise ValueError(f"{where}: document {rid} has no text field: keys {sorted(r)}")
    if need_summary and summ is None:
        raise ValueError(f"{where}: document {rid} has no summary field: keys {sorted(r)}")
    return {"id": str(rid), "raw_id": rid, "document": normalise_text(doc),
            "summary": normalise_text(summ) if (need_summary and summ is not None) else None}


def load_file(data_dir, split_file, max_docs=None):
    """split_file: 'train' (Train/train.json, with summaries) or 'test'. Repeated ids keep their first record."""
    p = file_path(data_dir, split_file)
    if not p.exists():
        raise FileNotFoundError(p)
    recs = _records(_read_json(p))
    if max_docs:
        recs = recs[:max_docs]
    docs = [norm_doc(r, f"{p.name}[{i}]", split_file != "test") for i, r in enumerate(recs)]
    seen, out, dup = set(), [], 0
    for d in docs:
        if d["id"] in seen:
            dup += 1
            continue
        seen.add(d["id"])
        out.append(d)
    if dup:
        print(f"[data] {p.name}: dropped {dup} repeated ids (first one kept)")
    return out


def dev_ids(ids, dev_size, seed=0):
    """Deterministic held-out set: the dev_size ids with the smallest sha1(seed:id)."""
    h = sorted(ids, key=lambda i: hashlib.sha1(f"{seed}:{i}".encode()).hexdigest())
    return set(h[:dev_size])


def describe_raw(data_dir, n_chars=300):
    """Raw structure of every file (first record), to confirm the format before training."""
    for split in ("train", "test"):
        p = file_path(data_dir, split)
        raw = _read_json(p)
        recs = _records(raw)
        first = recs[0] if recs else {}
        short = {k: (v[:n_chars] + "...") if isinstance(v, str) and len(v) > n_chars else v for k, v in first.items()}
        print(f"{p.name}: {p.stat().st_size / 2**20:.1f} MB | top-level {type(raw).__name__} | {len(recs)} records | "
              f"keys {sorted(first)} | {json.dumps(short, default=str)[:2 * n_chars + 200]}")


# ------------------------------------------------------------------------------------------- sentences
# a sentence ends after . ? ! ; or : followed by a space and an upper-case letter, digit, quote or bracket
_SENT_RE = re.compile(r"(?<=[.?!;:])\s+(?=[\"'(\[A-Z0-9])")
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STOP = set("""a an the of to in and or for on at by with from as is are was were be been being this that these those it
its which who whom whose what when where why how not no nor but if then than so such there their them they he she his
her him we our us you your i me my mine shall will would should could can may might must do does did done has have had
having into upon under over about above below up down out off again further once all any both each few more most other
some same own only very s t also said per vs v""".split())


def split_sentences(text, max_chars=1500):
    """Sentence pieces whose concatenation is exactly `text` (the separating space stays at the start of the next
    piece). Very long pieces are cut at a space every ~max_chars characters."""
    if not text:
        return []
    cuts = [m.end() - 1 for m in _SENT_RE.finditer(text)]                  # index of the space before the sentence
    pieces, s = [], 0
    for c in cuts + [len(text)]:
        piece = text[s:c]
        while len(piece) > max_chars:
            k = piece.rfind(" ", 1, max_chars)
            k = k if k > 0 else max_chars
            pieces.append(piece[:k]); piece = piece[k:]
        if piece:
            pieces.append(piece)
        s = c
    return pieces


def content_words(text):
    """Lower-cased words without stop words; numbers are kept when they have >= 3 digits (years, section numbers)."""
    return [w for w in _WORD_RE.findall(text.lower())
            if (w.isdigit() and len(w) >= 3) or (not w.isdigit() and w not in STOP)]


def sentence_scores(sentences):
    """Salience of every sentence = cosine(TF-IDF(sentence), TF-IDF(whole document)), with IDF computed over the
    document's own sentences (log(1 + n / df)). Unsupervised, identical for train and test documents."""
    bags = [Counter(content_words(s)) for s in sentences]
    n = len(bags)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    df = Counter(w for b in bags for w in b)
    idf = {w: math.log(1 + n / c) for w, c in df.items()}
    cent = Counter()
    for b in bags:
        cent.update(b)
    cv = {w: c * idf[w] for w, c in cent.items()}
    cn = math.sqrt(sum(v * v for v in cv.values())) or 1.0
    out = np.zeros(n, dtype=np.float32)
    for i, b in enumerate(bags):
        if not b:
            continue
        sv = {w: c * idf[w] for w, c in b.items()}
        sn = math.sqrt(sum(v * v for v in sv.values())) or 1.0
        out[i] = sum(v * cv[w] for w, v in sv.items()) / (sn * cn)
    return out


# ------------------------------------------------------------------------------------------- tokenisation
_TOK = None


def _init_worker(tok_path):
    global _TOK
    from .tokenizer import BPETokenizer
    _TOK = BPETokenizer.load(tok_path)


def _encode_doc(item):
    """(document, summary | None) -> (tokens uint16, sent_end int32, sent_score float32, summary tokens | None)."""
    doc, summ = item
    sents = split_sentences(doc) or ["."]
    toks, ends = [], []
    for s in sents:
        toks.extend(_TOK.encode(s))
        ends.append(len(toks))
    if not toks:
        toks, ends = _TOK.encode("."), [1]
    st = None if summ is None else np.asarray(_TOK.encode(summ) or _TOK.encode("."), dtype=np.uint16)
    return (np.asarray(toks, dtype=np.uint16), np.asarray(ends, dtype=np.int32), sentence_scores(sents), st)


def tokenizer_fingerprint(tok_path):
    from .tokenizer import BPETokenizer
    tok = BPETokenizer.load(tok_path)
    assert tok.vocab_size <= 65535, "uint16 token cache needs vocab <= 65535"
    return tok.fingerprint()


def cache_path(cache_dir, tok_path, max_docs=None, dev_size=300, dev_frac=0.05, split_seed=0):
    """The file name encodes the tokenizer and the dev split, so a changed split never reuses a stale cache."""
    tag = tokenizer_fingerprint(tok_path) + f"_dev{dev_size}-{dev_frac}-{split_seed}" + (f"_n{max_docs}" if max_docs else "")
    return Path(cache_dir) / f"summ_{tag}.pkl"


def tokenise(items, tok_path, n_proc=4):
    if n_proc and n_proc > 1 and len(items) > 64:
        import gc
        import multiprocessing as mp
        gc.collect(); gc.freeze()                 # forked workers do not copy the parent's objects when GC runs
        try:
            with mp.get_context("fork").Pool(n_proc, initializer=_init_worker, initargs=(tok_path,)) as pool:
                return list(pool.imap(_encode_doc, items, chunksize=8))
        finally:
            gc.unfreeze()
    _init_worker(tok_path)
    return [_encode_doc(x) for x in items]


def _pack(docs, enc, source):
    return {"ids": [d["id"] for d in docs], "raw_ids": [d["raw_id"] for d in docs],
            "tokens": [e[0] for e in enc], "sent_end": [e[1] for e in enc], "sent_score": [e[2] for e in enc],
            "sum_tokens": None if enc and enc[0][3] is None else [e[3] for e in enc],
            "documents": [d["document"] for d in docs],
            "summaries": None if docs and docs[0]["summary"] is None else [d["summary"] for d in docs],
            "source": source}


def build_cache(data_dir, tok_path, cache_dir, dev_size=300, dev_frac=0.05, split_seed=0, n_proc=4, max_docs=None,
                log=print):
    """Reads the JSON files, holds out the dev split, tokenises everything once. Returns the cache path."""
    import time
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cp = cache_path(cache_dir, tok_path, max_docs, dev_size, dev_frac, split_seed)
    if cp.exists():
        return cp
    t0 = time.time()
    train = load_file(data_dir, "train", max_docs)
    test = load_file(data_dir, "test", max_docs)
    n_dev = min(dev_size, max(1, int(round(dev_frac * len(train))))) if len(train) > 20 else max(1, len(train) // 4)
    dv = dev_ids([d["id"] for d in train], n_dev, split_seed)
    parts = {"train": [d for d in train if d["id"] not in dv], "dev": [d for d in train if d["id"] in dv], "test": test}
    obj = {"split_seed": split_seed, "n_dev": n_dev}
    for split, docs in parts.items():
        enc = tokenise([(d["document"], d["summary"]) for d in docs], tok_path, n_proc)
        obj[split] = _pack(docs, enc, str(file_path(data_dir, "test" if split == "test" else "train")))
        nt = sum(len(e[0]) for e in enc)
        ns = sum(len(e[3]) for e in enc) if split != "test" else 0
        log(f"[data] {split}: {len(docs)} documents | {nt / 1e6:.1f}M document tokens"
            + (f" | {ns / 1e6:.2f}M summary tokens" if ns else ""))
    tmp = cp.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cp)
    log(f"[data] tokenised in {time.time() - t0:.0f}s -> {cp.name}")
    return cp


def load_cached(data_dir, tok_path, cache_dir, max_docs=None, n_proc=4, dev_size=300, dev_frac=0.05, split_seed=0,
                log=print):
    cp = build_cache(data_dir, tok_path, cache_dir, dev_size, dev_frac, split_seed, n_proc, max_docs, log)
    with open(cp, "rb") as f:
        return pickle.load(f)


def subset(split, idx):
    """A split restricted to document indices idx."""
    return {k: ([v[i] for i in idx] if isinstance(v, list) else v) for k, v in split.items()}


# ------------------------------------------------------------------------------------------- statistics
def stats(data):
    """Statistics for the report: lengths (words / tokens), compression ratio, how much fits in a token budget,
    abstractiveness of the references (novel n-grams), id overlap / duplicate texts across splits."""
    out = {}
    for split in SPLITS:
        d = data[split]
        dl = np.array([len(t) for t in d["tokens"]])
        dw = np.array([len(x.split()) for x in d["documents"]])
        rec = {"documents": len(dl)}
        if len(dl):
            q = lambda a: {k: int(np.percentile(a, p)) for k, p in (("min", 0), ("p10", 10), ("p50", 50), ("p90", 90),
                                                                     ("max", 100))} | {"mean": int(a.mean())}
            rec["doc_words"], rec["doc_tokens"] = q(dw), q(dl)
            rec["tokens_per_word_doc"] = round(float(dl.sum() / max(1, dw.sum())), 3)
            rec["sentences_per_doc_p50"] = int(np.median([len(e) for e in d["sent_end"]]))
            for L in (1024, 2048, 4096, 8192):
                rec[f"doc_fits_in_{L}"] = round(float((dl <= L).mean()), 3)
        if d.get("sum_tokens"):
            sl = np.array([len(t) for t in d["sum_tokens"]])
            sw = np.array([len(x.split()) for x in d["summaries"]])
            rec["sum_words"], rec["sum_tokens"] = q(sw), q(sl)
            rec["compression_words_p50"] = round(float(np.median(dw / np.maximum(sw, 1))), 2)
            for L in (256, 384, 512, 640, 768, 1024):
                rec[f"sum_fits_in_{L}"] = round(float((sl <= L).mean()), 3)
            k = min(300, len(sw))
            for n in (1, 2, 3):
                rec[f"ref_novel_{n}gram"] = round(float(np.mean([novel_ngram_ratio(d["summaries"][i], d["documents"][i], n)
                                                                  for i in range(k)])), 4)
        out[split] = rec
    tr = set(data["train"]["ids"])
    fp = lambda s: {hashlib.sha1(t.encode()).hexdigest() for t in data[s]["documents"]}
    out["overlap"] = {"ids train&test": len(tr & set(data["test"]["ids"])), "ids train&dev": len(tr & set(data["dev"]["ids"])),
                      "identical_texts train&test": len(fp("train") & fp("test"))}
    return out


def ngrams(words, n):
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def simple_words(text):
    return re.findall(r"\w+", text.lower())


def novel_ngram_ratio(summary, document, n):
    """Share of the summary's n-grams that never occur in the document (0 = purely extractive)."""
    s = ngrams(simple_words(summary), n)
    if not s:
        return 0.0
    dset = set(ngrams(simple_words(document), n))
    return sum(g not in dset for g in s) / len(s)


def summary_source_positions(data, split="train", max_docs=500, n_bins=10):
    """Where in the document does the reference summary's content come from? For every summary sentence, the
    document sentence with the highest content-word overlap is found; its relative position is histogrammed."""
    d = data[split]
    hist = np.zeros(n_bins)
    for i in range(min(max_docs, len(d["ids"]))):
        dsents = split_sentences(d["documents"][i])
        dsets = [set(content_words(s)) for s in dsents]
        if not dsets:
            continue
        for s in split_sentences(d["summaries"][i]):
            sw = set(content_words(s))
            if len(sw) < 3:
                continue
            ov = [len(sw & x) / len(sw) for x in dsets]
            j = int(np.argmax(ov))
            if ov[j] >= 0.3:
                hist[min(n_bins - 1, int(n_bins * (j + 0.5) / len(dsets)))] += 1
    return (hist / max(1, hist.sum())).round(4).tolist()
