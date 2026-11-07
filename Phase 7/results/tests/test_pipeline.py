"""Correctness checks for the Phase-7 pipeline (run before trusting any number).

    python tests/test_pipeline.py [--ckpt_dir <folder with the Phase-1 run folders>]
"""
import argparse
import json
import os
import random
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from p7pcr import peft                                                       # noqa: E402
from p7pcr.backbones import load_backbone                                    # noqa: E402
from p7pcr.bm25 import BM25, terms                                           # noqa: E402
from p7pcr.data import _records, canon, oracle_f1                            # noqa: E402
from p7pcr.metrics import micro_prf, rank, report                            # noqa: E402
from p7pcr.model import Retriever, kd_loss, ranking_loss, similarity         # noqa: E402
from p7pcr.phase1_models import MODEL_CLASSES                                # noqa: E402
from p7pcr.run import case_year, fuse                                        # noqa: E402
from p7pcr.train import (build_batch, check_prediction, collate_docs, mine_negatives, query_batches,  # noqa: E402
                         select_chunks, write_submission)

V = 300
TINY = {"LlamaLM": dict(vocab_size=V, block_size=64, n_layer=2, n_head=4, n_kv_head=2, d_model=32),
        "EncDecLM": dict(vocab_size=V, enc_block=64, dec_block=32, n_enc_layer=2, n_dec_layer=1, n_head=4, d_model=32),
        "HybridLM": dict(vocab_size=V, block_size=64, n_layer=2, n_head=4, d_model=32),
        "EfficientLM": dict(vocab_size=V, block_size=64, window=8, n_layer=2, n_head=4, n_kv_head=1, d_model=32)}
TMP = tempfile.mkdtemp()


def tiny_ckpt(cls):
    torch.manual_seed(0)
    m = MODEL_CLASSES[cls](**TINY[cls])
    p = os.path.join(TMP, f"{cls}.pt")
    torch.save({"model_class": cls, "model_kw": TINY[cls], "model": m.state_dict()}, p)
    return p


def test_ids_and_loader():
    assert canon("0000021652") == canon(21652) == canon("21652") == "21652" and canon(" abc ") == "abc"
    recs = _records({"0001": {"text": ["a.", "b."]}, "0002": ["c."]}, "x")
    assert [r["id"] for r in recs] == ["0001", "0002"]
    assert abs(oracle_f1({"a": ["x"] * 7}, 7) - 1.0) < 1e-9 and oracle_f1({"a": ["x", "y"]}, 7) < 1.0
    print("ok  ids (canonical keys) + loader formats + oracle F1")


def test_metrics():
    rng = np.random.default_rng(0)
    for _ in range(50):
        nq, nc = int(rng.integers(1, 6)), int(rng.integers(8, 30))
        q_ids = [f"q{i}" for i in range(nq)]
        c_ids = [f"c{j}" for j in range(nc)] + (["q0"] if rng.random() < 0.5 else [])
        s = rng.normal(size=(nq, len(c_ids)))
        gold = {q: list(rng.choice(c_ids[:nc], int(rng.integers(0, min(12, nc))), replace=False)) for q in q_ids}
        order = rank(s, q_ids, c_ids)
        pred = {}
        for i, q in enumerate(q_ids):
            lst = [c_ids[j] for j in np.argsort(-s[i]) if c_ids[j] != q]          # brute force, self excluded
            assert [c_ids[j] for j in order[i] if c_ids[j] != q][:7] == lst[:7]
            assert q not in [c_ids[j] for j in order[i][:7]]
            pred[q] = lst[:7]
        tp = sum(len(set(pred[q]) & set(g)) for q, g in gold.items())
        P = tp / (7 * nq); R = tp / max(1, sum(len(g) for g in gold.values()))
        F = 2 * P * R / (P + R) if P + R else 0.0
        p, r, f, _ = micro_prf(pred, gold)
        assert abs(p - P) < 1e-12 and abs(r - R) < 1e-12 and abs(f - F) < 1e-12
        rep, _, _ = report(s, q_ids, c_ids, gold)
        assert abs(rep["micro_f1@7"] - F) < 1e-12 and rep["recall@200"] >= rep["recall@7"] - 1e-12
        top = rank(s, q_ids, c_ids, 7)
        assert all(list(top[i]) == list(order[i][:7]) for i in range(nq))
    print("ok  micro P/R/F1@7 == brute force (self never returned), partial top-k == full sort")


def test_bm25():
    docs = ["the court held section 302 applies to murder", "contract breach damages awarded",
            "murder conviction under section 302 upheld by the court", "tax assessment appeal"]
    bm = BM25(docs, k1=1.2, b=0.75)
    q = ["murder section 302 conviction"]
    s = bm.scores(q, "count")[0]
    # brute force Okapi BM25
    bags = [terms(d) for d in docs]
    N, avg = len(docs), np.mean([len(b) for b in bags])
    ref = []
    for b in bags:
        tot = 0.0
        for t in set(terms(q[0])):
            df = sum(t in x for x in bags)
            if not df:
                continue
            tf = b.count(t)
            idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
            tot += terms(q[0]).count(t) * idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(b) / avg))
        ref.append(tot)
    assert np.allclose(s, ref, atol=1e-4), (s, ref)
    assert s.argmax() == 2
    print("ok  BM25 == brute-force Okapi formula")


def test_chunks_and_collate():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n = int(rng.integers(1, 6000)); L = int(rng.choice([16, 64, 512])); K = int(rng.integers(1, 9))
        for st in ("head", "tail", "headtail", "uniform", "middle"):
            sp = select_chunks(n, L, K, st)
            assert all(0 <= s < e <= n and e - s <= L for s, e in sp) and len(sp) <= max(K, 1) or n <= K * L
    toks = [np.arange(100, dtype=np.uint16), np.arange(30, dtype=np.uint16)]
    b = collate_docs(toks, {"len": 16, "k": 4, "strategy": "uniform"}, 0)
    assert b["doc"].tolist().count(0) == 4 and b["doc"].tolist().count(1) == 2 and b["keep"].sum() == 64 + 30
    b2 = collate_docs(toks, {"len": 16, "k": 4, "strategy": "uniform", "train_k": 2}, 0, random.Random(0))
    assert b2["doc"].tolist().count(0) == 2 and b2["doc"].tolist().count(1) == 2
    print("ok  chunk selection + collate + chunk dropout")


def test_batches_and_masks():
    split = {"q_ids": ["q0", "q1", "q2"], "c_ids": ["a", "b", "c", "d", "e", "q1"],
             "gold": {"q0": ["a", "b"], "q1": ["b", "c"], "q2": ["d"]}}
    inst = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 3)]
    rng = random.Random(0)
    for _ in range(20):
        for bt in query_batches(inst, 2, rng):
            assert len(bt) <= 2
    negs = mine_negatives(np.random.default_rng(0).normal(size=(3, 6)), split, top=10)
    for i, q in enumerate(split["q_ids"]):
        assert not {split["c_ids"][j] for j in negs[i]} & (set(split["gold"][q]) | {q})
    batch = [(0, 0), (1, 1)]                                   # q0 -> a, q1 -> b
    qidx, cand, target, allowed, positives, own = build_batch(batch, split, [np.array([4]), np.array([4, 5])], 2, True, rng)
    names = [split["c_ids"][j] for j in cand]
    col = {n: k for k, n in enumerate(names)}
    assert names[target[0]] == "a" and names[target[1]] == "b"
    assert not allowed[0, col["b"]] and allowed[0, col["a"]]          # q0's other relevant candidate is masked
    assert allowed[1, col["b"]] and positives[0, col["b"]]
    if "q1" in col:
        assert not allowed[1, col["q1"]]                             # a query is never its own candidate
    _, _, _, al2, _, _ = build_batch(batch, split, [np.array([4]), np.array([3])], 1, False, rng)
    assert al2.sum(1).tolist() == [2, 2]                              # in_batch off: own positive + own negative
    # losses: masking works, KD zero for identical scores
    S = torch.randn(2, len(cand), requires_grad=True)
    for name in ("infonce", "multipos", "pairwise"):
        l = ranking_loss(S, target, allowed, positives, name, own)
        l.backward(); assert torch.isfinite(l) and torch.isfinite(S.grad).all()
        S.grad = None
    Sm = S.detach().clone(); Sm[0, col["b"]] = 1e6                   # a masked entry cannot change InfoNCE
    assert torch.allclose(ranking_loss(Sm, target, allowed, positives), ranking_loss(S.detach(), target, allowed, positives))
    assert float(kd_loss(S.detach(), S.detach(), allowed)) < 1e-6 and float(kd_loss(S.detach(), torch.randn_like(S), allowed)) > 0
    print("ok  batches (distinct queries), negatives exclude gold/self, false-negative masking, losses, KD")


def test_encoder():
    """Representations do not depend on padding / batch composition; PEFT methods are identity at init."""
    for cls in TINY:
        bb, _ = load_backbone(tiny_ckpt(cls), attention="bidirectional")
        for rcfg in ({"pool": "mean", "agg": "mean", "proj": 16}, {"pool": "attn", "agg": "attn", "proj": 0, "sim": "maxsim"},
                     {"pool": "last", "agg": "max", "proj": 8}):
            m = Retriever(bb, rcfg).eval()
            cc = {"len": 16, "k": 3, "strategy": "uniform"}
            toks = [np.random.default_rng(i).integers(0, 290, n).astype(np.uint16) for i, n in enumerate((70, 9, 33))]
            with torch.no_grad():
                full = m.encode(collate_docs(toks, cc, 0))
                for i, t in enumerate(toks):
                    one = m.encode(collate_docs([t], cc, 0))
                    assert torch.allclose(one["doc"][0], full["doc"][i], atol=1e-5), (cls, rcfg)
                s = similarity(full, full, m.sim)
                assert s.shape == (3, 3) and torch.allclose(s.diag(), torch.ones(3), atol=1e-4)
    p = tiny_ckpt("LlamaLM")
    bb0, _ = load_backbone(p)
    ids = torch.randint(0, 290, (2, 20)); keep = torch.ones(2, 20, dtype=torch.bool)
    y0 = bb0(ids, keep)
    for mc in ({"name": "lorma", "r": 4}, {"name": "ia3"}, {"name": "lora", "r": 4}, {"name": "partial", "train_last_k": 1},
               {"name": "qlora", "r": 4}, {"name": "prefix", "n_prefix": 4}, {"name": "prompt", "n_prompt": 4}):
        bb, _ = load_backbone(p)
        peft.apply_method(bb, mc)
        assert peft.count_params(bb)["trainable"] > 0
        if mc["name"] not in ("prefix", "prompt", "qlora"):
            assert (bb(ids, keep) - y0).abs().max() < 1e-4, mc
    print("ok  retriever: padding / batch invariant (A, B, C, D x pooling/agg/sim), PEFT identity at init")


def test_fusion_and_years():
    assert case_year("decided on 12 March 1998, following the 1950 Act") == 1998 and case_year("no year") is None
    a, b = np.random.default_rng(0).normal(size=(3, 9)), np.random.default_rng(1).normal(size=(3, 9))
    fm = np.zeros((3, 9), dtype=bool); fm[:, 0] = True
    s = fuse([a, b], [0.5, 0.5], fm)
    assert (s[:, 0] < s[:, 1:].min(1)).all()
    print("ok  fusion (z-scores, future candidates pushed down), year extraction")


def test_submission():
    te = {"q_ids": ["21652", "30721"], "q_raw": ["0000021652", "0000030721"],
          "c_ids": [str(i) for i in range(1, 12)] + ["21652"], "c_raw": [f"{i:010d}" for i in range(1, 12)] + ["0000021652"]}
    s = np.random.default_rng(0).normal(size=(2, 12)); s[0, -1] = 99                 # self has the top score
    zp = write_submission(s, te, tempfile.mkdtemp())
    import zipfile
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"]
        obj = json.loads(z.read("prediction.json"))
    assert set(obj) == {"0000021652", "0000030721"} and all(len(v) == 7 for v in obj.values())
    assert "0000021652" not in obj["0000021652"]
    for bad in ({"0000021652": obj["0000021652"]}, {**obj, "0000021652": obj["0000021652"][:6]},
                {**obj, "0000021652": ["0000000001"] * 7}, {**obj, "0000030721": ["0000009999"] + obj["0000030721"][1:]}):
        try:
            check_prediction(bad, te); raise AssertionError("accepted an invalid prediction")
        except AssertionError as e:
            if "accepted" in str(e):
                raise
    print("ok  submission ({query: 7 unique pool ids}, raw ids kept, self excluded, zip root)")


def test_checkpoints(ckpt_dir):
    from p7pcr.train import CKPT_FOLDERS
    for key, folder in CKPT_FOLDERS.items():
        p = os.path.join(ckpt_dir, folder, "model_final.pt")
        if not os.path.exists(p):
            continue
        bb, info = load_backbone(p, attention="bidirectional", keep_layers=6 if key == "D" else None)
        m = Retriever(bb, {"proj": 256}).eval()
        toks = [np.random.default_rng(i).integers(0, 30000, n).astype(np.uint16) for i, n in enumerate((900, 200))]
        with torch.no_grad():
            r = m.encode(collate_docs(toks, {"len": 256, "k": 2, "strategy": "uniform"}, 0))
        assert torch.isfinite(r["doc"]).all()
        print(f"ok  checkpoint {folder} ({info['class']}, {info['layers_used']}/{info['phase1_layers']} layers) encodes")
        del bb, m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="")
    a = ap.parse_args()
    test_ids_and_loader()
    test_metrics()
    test_bm25()
    test_chunks_and_collate()
    test_batches_and_masks()
    test_encoder()
    test_fusion_and_years()
    test_submission()
    if a.ckpt_dir:
        test_checkpoints(a.ckpt_dir)
    print("ALL TESTS PASSED")
