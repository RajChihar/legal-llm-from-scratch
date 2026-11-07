"""Phase-2 data: download, load, validate, split.

Layout (HF dataset Exploration-Lab/CS-781-Capstone, folder Phase_2_LNER):
    Train/fold1-text.json  {doc_id: text}
    Train/fold1-gold.json  {doc_id: [[start, end, label, text], ...]}   (format auto-detected, see _parse_gold)
    Train/fold2-text.json, Train/fold2-gold.json, Train/labels.txt
    Test/fold3-text.json
"""
import json
import os
import random
from collections import Counter
from pathlib import Path

HF_REPO = "Exploration-Lab/CS-781-Capstone"
HF_SUBDIR = "Phase_2_LNER"


def download(local_dir, token=None):
    """Gated dataset: needs a HF token (Kaggle: Add-ons -> Secrets -> HF_TOKEN). Returns the Phase_2_LNER path."""
    from huggingface_hub import snapshot_download
    token = token or os.environ.get("HF_TOKEN")
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=local_dir, token=token,
                      allow_patterns=[f"{HF_SUBDIR}/*"])
    return Path(local_dir) / HF_SUBDIR


def find_data_dir(candidates=()):
    """First folder that contains Train/labels.txt (searches the given paths, then /kaggle/input)."""
    roots = [Path(c) for c in candidates] + [Path("/kaggle/input")]
    for r in roots:
        if not r.exists():
            continue
        if (r / "Train" / "labels.txt").exists():
            return r
        for p in r.rglob("labels.txt"):
            if p.parent.name == "Train":
                return p.parent.parent
    return None


def _parse_gold(raw, texts):
    """Normalise gold annotations to {doc_id: [(start, end, label), ...]} with `end` exclusive.

    Accepts {id: [[s, e, label, (text)], ...]}, {id: [{"start":..,"end":..,"label":..}, ...]} or a list of
    {"id":.., "entities"/"spans":[...]} records. If entity texts are given, checks whether `end` is exclusive
    (text[s:e] == entity) or inclusive (text[s:e+1] == entity) and converts.
    """
    if isinstance(raw, list):
        raw = {str(r.get("id", r.get("doc_id"))): r.get("entities", r.get("spans", r.get("annotations")))
               for r in raw}
    out, ent_text = {}, {}
    for did, ents in raw.items():
        did = str(did)
        lst, txts = [], []
        for e in ents or []:
            if isinstance(e, dict):
                s = e.get("start", e.get("start_char", e.get("begin")))
                t = e.get("end", e.get("end_char"))
                lab = e.get("label", e.get("type", e.get("labels")))
                if isinstance(lab, list):
                    lab = lab[0]
                txt = e.get("text", e.get("entity_text"))
            else:
                s, t, lab = e[0], e[1], e[2]
                txt = e[3] if len(e) > 3 else None
            lst.append((int(s), int(t), str(lab))); txts.append(txt)
        out[did] = lst; ent_text[did] = txts
    # end-offset convention check
    exc = inc = 0
    for did, lst in out.items():
        doc = texts.get(did, "")
        for (s, t, _), txt in zip(lst, ent_text[did]):
            if txt is None:
                continue
            exc += doc[s:t] == txt
            inc += doc[s:t + 1] == txt
    if inc > exc:
        print(f"[data] gold `end` looks INCLUSIVE ({inc} vs {exc} matches) -> converting to exclusive")
        out = {d: [(s, t + 1, l) for s, t, l in lst] for d, lst in out.items()}
    elif exc + inc:
        print(f"[data] gold `end` is exclusive: {exc}/{sum(len(v) for v in ent_text.values())} entity texts match text[s:e]")
    return out


def load_phase2(data_dir):
    data_dir = Path(data_dir)
    tr = data_dir / "Train"
    labels = [l.strip() for l in open(tr / "labels.txt", encoding="utf-8") if l.strip()]
    folds = {}
    for k in (1, 2):
        texts = {str(i): t for i, t in json.load(open(tr / f"fold{k}-text.json", encoding="utf-8")).items()}
        gold = _parse_gold(json.load(open(tr / f"fold{k}-gold.json", encoding="utf-8")), texts)
        for d in texts:
            gold.setdefault(d, [])
        folds[k] = {"texts": texts, "gold": gold}
    test = {str(i): t for i, t in json.load(open(data_dir / "Test" / "fold3-text.json", encoding="utf-8")).items()}
    return {"labels": labels, "folds": folds, "test": test}


def make_split(data, mode="random", dev_frac=0.15, seed=1337):
    """Returns (train_docs, dev_docs) as {id: {"text":..., "gold":[...]}}.

    mode="random": pool fold1+fold2, hold out dev_frac of the documents (fixed seed -> same split for every run).
    mode="fold1"/"fold2": hold out that whole fold (2-fold CV style).
    mode="none": all documents for training (final retrain), empty dev.
    """
    pool = {}
    for k, f in data["folds"].items():
        for d, t in f["texts"].items():
            pool[d] = {"text": t, "gold": sorted(f["gold"][d]), "fold": k}
    ids = sorted(pool)
    if mode == "none":
        return pool, {}
    if mode in ("fold1", "fold2"):
        k = int(mode[-1])
        dev = [d for d in ids if pool[d]["fold"] == k]
    else:
        rng = random.Random(seed)
        rng.shuffle(ids)
        dev = ids[:max(1, round(dev_frac * len(ids)))]
    dev = set(dev)
    return ({d: v for d, v in pool.items() if d not in dev}, {d: v for d, v in pool.items() if d in dev})


def validate_gold(docs, labels):
    """Sanity report: unknown labels, out-of-range spans, overlaps (BIO cannot represent nesting)."""
    bad_label = bad_range = overlaps = nested = 0
    for d in docs.values():
        n = len(d["text"])
        g = sorted(d["gold"])
        for s, e, l in g:
            bad_label += l not in labels
            bad_range += not (0 <= s < e <= n)
        for (s1, e1, _), (s2, e2, _) in zip(g, g[1:]):
            if s2 < e1:
                overlaps += 1
                nested += e2 <= e1
    return {"unknown_label": bad_label, "bad_range": bad_range, "overlapping_pairs": overlaps, "nested_pairs": nested}


def eda(docs, labels, tok=None):
    """Corpus statistics used in the report (counts, lengths, label distribution)."""
    lab = Counter(l for d in docs.values() for _, _, l in d["gold"])
    chars = [len(d["text"]) for d in docs.values()]
    ent_len = Counter()
    for d in docs.values():
        for s, e, l in d["gold"]:
            ent_len[l] += e - s
    rep = {"n_docs": len(docs), "n_entities": sum(lab.values()),
           "chars_per_doc": {"min": min(chars), "mean": round(sum(chars) / len(chars)), "max": max(chars)},
           "label_counts": {l: lab.get(l, 0) for l in labels},
           "mean_entity_chars": {l: round(ent_len[l] / lab[l], 1) if lab[l] else 0 for l in labels}}
    if tok is not None:
        ntok = [len(tok.encode(d["text"])) for d in docs.values()]
        rep["tokens_per_doc"] = {"min": min(ntok), "mean": round(sum(ntok) / len(ntok)), "max": max(ntok),
                                 "total": sum(ntok)}
    return rep
