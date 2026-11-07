"""Phase-7 data: download, load, validate, tokenise once (cached), statistics.

Layout (HF dataset Exploration-Lab/CS-781-Capstone, folder Phase_7_PCR):
    Train/train.json  Train/train_candidates.json
    Dev/dev.json      Dev/dev_candidates.json
    Test/test.json    Test/test_candidates.json
A query is {"id": "0000021652", "text": [sentences], "relevant_candidates": ["0000590455", ...]} (no
relevant_candidates in test.json); a candidate is {"id": ..., "text": [sentences]}. Each split has its own pool:
queries of a split are only ever ranked against that split's candidates.
The loader also accepts a dict {id: {...}} / {id: text}, JSON lines, a {"data": [...]} wrapper, text as one string,
and the relevance list under a few other names, so a small format difference does not break it.

Ids: IndianKanoon ids such as "0000021652". They are compared through a canonical key (digits without leading zeros,
otherwise the stripped string) so "0000021652", "21652" and 21652 match; the submission always writes the ids
exactly as they appear in the files (query ids as in <split>.json, candidate ids as in <split>_candidates.json).

Tokenising with the pure-Python Phase-1 BPE is the slowest CPU step, so every document is tokenised ONCE
(multiprocessing) into uint16 arrays and cached; all runs (and both parallel GPU jobs) reuse the cache.
Cached form per split: {"q_ids": [key], "q_raw": [raw id], "q_tokens": [array], "gold": {key: [cand keys]} | None,
                        "c_ids": [key], "c_raw": [raw id], "c_tokens": [array], "q_text"/"c_text": [str] (BM25),
                        "gold_missing": n gold ids not in the pool}
"""
import hashlib
import json
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

HF_REPO = "Exploration-Lab/CS-781-Capstone"
HF_SUBDIR = "Phase_7_PCR"
SPLITS = ("train", "dev", "test")
ID_KEYS = ("id", "case_id", "doc_id", "query_id", "candidate_id", "uid", "name")
TEXT_KEYS = ("text", "sentences", "document", "doc", "case_text", "content", "body", "judgment")
REL_KEYS = ("relevant_candidates", "relevant", "relevant_ids", "positives", "labels", "gold", "citations", "cited")


def download(local_dir, token=None):
    """Gated dataset: needs a HF token (Kaggle: Add-ons -> Secrets -> HF_TOKEN). Returns the Phase_7_PCR path."""
    from huggingface_hub import snapshot_download
    token = token or os.environ.get("HF_TOKEN")
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=local_dir, token=token,
                      allow_patterns=[f"{HF_SUBDIR}/*"])
    return Path(local_dir) / HF_SUBDIR


def files(data_dir, split):
    d = Path(data_dir) / split.capitalize()
    return d / f"{split}.json", d / f"{split}_candidates.json"


def find_data_dir(candidates=()):
    """First folder with Train/train.json, Dev/dev.json and Test/test.json (+ candidate files): the given paths
    first, then a folder under /kaggle/input whose path names Phase 7 / PCR."""
    def ok(d):
        return all(p.exists() for s in SPLITS for p in files(d, s))
    for c in candidates:
        if c and ok(Path(c)):
            return Path(c)
    for r in [Path(c) for c in candidates if c] + [Path("/kaggle/input")]:
        if not r.exists():
            continue
        for p in r.rglob("train_candidates.json"):
            d = p.parent.parent
            low = str(d).lower()
            if ok(d) and ("pcr" in low or "phase_7" in low or "phase7" in low or "phase-7" in low):
                return d
    return None


def canon(x):
    """Canonical id key: '0000021652' / '21652' / 21652 -> '21652'; other strings stripped."""
    s = str(x).strip()
    if s.isdigit():
        return str(int(s))
    if s.endswith(".txt") and s[:-4].isdigit():
        return str(int(s[:-4]))
    return s


def _read_json(path):
    txt = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(txt)
    except json.JSONDecodeError:                                   # JSON lines
        return [json.loads(line) for line in txt.splitlines() if line.strip()]


def _records(raw, kind):
    """-> list of dicts with at least an id field."""
    if isinstance(raw, dict):
        for k in ("data", "queries", "candidates", "documents", "docs", "cases"):
            if isinstance(raw.get(k), list):
                return raw[k]
        recs = []
        for k, v in raw.items():
            if isinstance(v, dict):
                recs.append(v if any(x in v for x in ID_KEYS) else {"id": k, **v})
            elif isinstance(v, (str, list)):
                recs.append({"id": k, "text": v})
            else:
                raise ValueError(f"{kind}: unexpected value type for {k}: {type(v)}")
        return recs
    if isinstance(raw, list):
        return raw
    raise ValueError(f"{kind}: unexpected top-level JSON type {type(raw)}")


def _first(r, keys):
    for k in keys:
        if k in r and r[k] is not None:
            return k, r[k]
    return None, None


def normalise_text(t):
    """Sentence lists are joined with spaces; whitespace runs become one space."""
    if isinstance(t, list):
        t = " ".join("" if x is None else str(x) for x in t)
    return " ".join(str(t).split())


def _ids_list(v):
    if v is None:
        return None
    if isinstance(v, str):
        v = [x for x in v.replace(";", ",").replace("[", "").replace("]", "").split(",") if x.strip()]
    elif not isinstance(v, (list, tuple)):
        v = [v]
    return [x.strip() if isinstance(x, str) else x for x in v]


def load_split(data_dir, split, max_queries=None, max_cands=None):
    qp, cp = files(data_dir, split)
    qrecs, crecs = _records(_read_json(qp), qp.name), _records(_read_json(cp), cp.name)
    cands, seen = [], set()
    for i, r in enumerate(crecs):
        _, rid = _first(r, ID_KEYS)
        _, text = _first(r, TEXT_KEYS)
        if rid is None or text is None:
            raise ValueError(f"{cp.name}[{i}]: candidate without id/text: keys {sorted(r)}")
        k = canon(rid)
        if k in seen:
            continue
        seen.add(k)
        cands.append({"key": k, "raw": rid, "text": normalise_text(text)})
    queries, qseen = [], set()
    for i, r in enumerate(qrecs):
        _, rid = _first(r, ID_KEYS)
        _, text = _first(r, TEXT_KEYS)
        _, rel = _first(r, REL_KEYS)
        if rid is None or text is None:
            raise ValueError(f"{qp.name}[{i}]: query without id/text: keys {sorted(r)}")
        if split != "test" and rel is None:
            raise ValueError(f"{qp.name}[{i}]: query {rid} has no relevant_candidates field: keys {sorted(r)}")
        k = canon(rid)
        if k in qseen:
            continue
        qseen.add(k)
        queries.append({"key": k, "raw": rid, "text": normalise_text(text),
                        "rel": None if split == "test" else [canon(x) for x in _ids_list(rel)]})
    if max_queries:
        queries = queries[:max_queries]
    if max_cands:                                                  # debug: keep the gold candidates + the first n others
        need = {g for q in queries for g in (q["rel"] or [])}
        cands = [c for c in cands if c["key"] in need] + [c for c in cands if c["key"] not in need][:max_cands]
    return queries, cands


def describe_raw(data_dir, n_chars=250):
    for split in SPLITS:
        for p in files(data_dir, split):
            raw = _read_json(p)
            recs = _records(raw, p.name)
            first = recs[0] if recs else {}
            short = {}
            for k, v in first.items():
                if isinstance(v, list) and len(v) > 4:
                    short[k] = v[:3] + [f"... ({len(v)} items)"]
                elif isinstance(v, str) and len(v) > n_chars:
                    short[k] = v[:n_chars] + "..."
                else:
                    short[k] = v
            print(f"{p.parent.name}/{p.name}: {p.stat().st_size / 2**20:.1f} MB | top-level {type(raw).__name__} | "
                  f"{len(recs)} records | keys {sorted(first)} | {json.dumps(short, default=str)[:2 * n_chars + 200]}")


# ------------------------------------------------------------------------------------------- tokenisation
_TOK = None


def _init_worker(tok_path):
    global _TOK
    from .tokenizer import BPETokenizer
    _TOK = BPETokenizer.load(tok_path)


def _encode(text):
    ids = _TOK.encode(" " + text) if text else []
    return np.asarray(ids or _TOK.encode(" ."), dtype=np.uint16)


def tokenise(texts, tok_path, n_proc=4):
    if n_proc and n_proc > 1 and len(texts) > 64:
        import gc
        import multiprocessing as mp
        gc.collect(); gc.freeze()
        try:
            with mp.get_context("fork").Pool(n_proc, initializer=_init_worker, initargs=(tok_path,)) as pool:
                return list(pool.imap(_encode, texts, chunksize=16))
        finally:
            gc.unfreeze()
    _init_worker(tok_path)
    return [_encode(t) for t in texts]


def tokenizer_fingerprint(tok_path):
    from .tokenizer import BPETokenizer
    tok = BPETokenizer.load(tok_path)
    assert tok.vocab_size <= 65535, "uint16 token cache needs vocab <= 65535"
    return tok.fingerprint()


def cache_path(cache_dir, tok_path, max_queries=None, max_cands=None):
    tag = tokenizer_fingerprint(tok_path) + (f"_q{max_queries}_c{max_cands}" if max_queries or max_cands else "")
    return Path(cache_dir) / f"pcr_{tag}.pkl"


def build_cache(data_dir, tok_path, cache_dir, n_proc=4, max_queries=None, max_cands=None, log=print):
    import time
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cp = cache_path(cache_dir, tok_path, max_queries, max_cands)
    if cp.exists():
        return cp
    t0 = time.time()
    obj = {}
    for split in SPLITS:
        qs, cs = load_split(data_dir, split, max_queries, max_cands)
        toks = tokenise([q["text"] for q in qs] + [c["text"] for c in cs], tok_path, n_proc)
        pool = {c["key"] for c in cs}
        gold, missing = None, 0
        if split != "test":
            gold = {}
            for q in qs:
                g = [x for x in dict.fromkeys(q["rel"]) if x != q["key"]]
                missing += sum(x not in pool for x in g)
                gold[q["key"]] = [x for x in g if x in pool]
        obj[split] = {"q_ids": [q["key"] for q in qs], "q_raw": [q["raw"] for q in qs], "q_tokens": toks[:len(qs)],
                      "q_text": [q["text"] for q in qs], "gold": gold, "gold_missing": missing,
                      "c_ids": [c["key"] for c in cs], "c_raw": [c["raw"] for c in cs], "c_tokens": toks[len(qs):],
                      "c_text": [c["text"] for c in cs], "source": [str(p) for p in files(data_dir, split)]}
        nt = sum(len(t) for t in toks)
        log(f"[data] {split}: {len(qs)} queries | {len(cs)} candidates | {nt / 1e6:.1f}M tokens"
            + (f" | {missing} gold ids not in the pool (ignored)" if missing else ""))
    tmp = cp.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cp)
    log(f"[data] tokenised in {time.time() - t0:.0f}s -> {cp.name}")
    return cp


def load_cached(data_dir, tok_path, cache_dir, n_proc=4, max_queries=None, max_cands=None, log=print):
    cp = build_cache(data_dir, tok_path, cache_dir, n_proc, max_queries, max_cands, log)
    with open(cp, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------------------------------- statistics
def stats(data):
    """For the report: queries / pool sizes, relevant candidates per query, document lengths, overlap of ids
    between splits and between queries and pools."""
    out = {}
    q = lambda a: {k: int(np.percentile(a, p)) for k, p in (("min", 0), ("p10", 10), ("p50", 50), ("p90", 90),
                                                            ("max", 100))} | {"mean": round(float(np.mean(a)), 1)}
    for split in SPLITS:
        d = data[split]
        rec = {"queries": len(d["q_ids"]), "candidates": len(d["c_ids"]),
               "query_tokens": q([len(t) for t in d["q_tokens"]]), "candidate_tokens": q([len(t) for t in d["c_tokens"]]),
               "query_words": q([len(t.split()) for t in d["q_text"]]),
               "queries_in_own_pool": len(set(d["q_ids"]) & set(d["c_ids"]))}
        for L in (512, 2048, 4096):
            rec[f"candidates_fit_in_{L}"] = round(float(np.mean([len(t) <= L for t in d["c_tokens"]])), 3)
        if d["gold"] is not None:
            g = np.array([len(d["gold"][k]) for k in d["q_ids"]])
            rec["relevant_per_query"] = q(g) | {"share_0": round(float((g == 0).mean()), 3),
                                                "share_le_7": round(float((g <= 7).mean()), 3)}
            rec["gold_missing_from_pool"] = d["gold_missing"]
            rec["oracle_micro_f1@7"] = round(oracle_f1(d["gold"], 7), 4)
            cnt = Counter(c for k in d["q_ids"] for c in d["gold"][k])
            rec["distinct_relevant_candidates"] = len(cnt)
            rec["most_cited_candidates"] = cnt.most_common(5)
            rec["share_of_pool_ever_relevant"] = round(len(cnt) / max(1, len(d["c_ids"])), 3)
        out[split] = rec
    ids = {s: set(data[s]["c_ids"]) for s in SPLITS}
    out["pool_overlap"] = {"train&dev": len(ids["train"] & ids["dev"]), "train&test": len(ids["train"] & ids["test"]),
                           "dev&test": len(ids["dev"] & ids["test"])}
    out["query_overlap"] = {"train&test": len(set(data["train"]["q_ids"]) & set(data["test"]["q_ids"]))}
    fp = lambda s: {hashlib.sha1(t.encode()).hexdigest() for t in data[s]["q_text"]}
    out["identical_query_texts train&test"] = len(fp("train") & fp("test"))
    return out


def oracle_f1(gold, k=7):
    """Best possible micro-F1@k when exactly k candidates must be returned (upper bound of any system)."""
    tp = sum(min(k, len(g)) for g in gold.values())
    p = tp / (k * len(gold)) if gold else 0.0
    r = tp / max(1, sum(len(g) for g in gold.values()))
    return 2 * p * r / (p + r) if p + r else 0.0
