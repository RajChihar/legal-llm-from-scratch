"""Correctness checks for the Phase-4 pipeline (run before trusting any number).

    python tests/test_pipeline.py [--ckpt_dir <folder with the Phase-1 run folders>]
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from p4cjpe import peft                                                   # noqa: E402
from p4cjpe.data import _label, _records, norm_case                      # noqa: E402
from p4cjpe.metrics import best_threshold, binary_report                 # noqa: E402
from p4cjpe.model import CJPEModel                                        # noqa: E402
from p4cjpe.train import check_prediction, collate, doc_schedule, select_chunks, write_submission   # noqa: E402

TINY = {"LlamaLM": dict(vocab_size=20100, block_size=64, n_layer=2, n_head=4, n_kv_head=2, d_model=32),
        "EncDecLM": dict(vocab_size=20100, enc_block=64, dec_block=32, n_enc_layer=2, n_dec_layer=1, n_head=4, d_model=32),
        "HybridLM": dict(vocab_size=20100, block_size=64, n_layer=2, n_head=4, d_model=32),
        "EfficientLM": dict(vocab_size=20100, block_size=64, window=8, n_layer=2, n_head=4, n_kv_head=1, d_model=32)}


def test_chunks():
    rng = np.random.default_rng(0)
    for _ in range(3000):
        n = int(rng.integers(1, 6000)); L = int(rng.choice([16, 64, 512])); K = int(rng.integers(1, 7))
        ov = float(rng.choice([0.0, 0.25, 0.5]))
        for st in ("head", "tail", "headtail", "uniform", "middle"):
            sp = select_chunks(n, L, K, st, ov)
            assert 1 <= len(sp) <= max(K, 1) or (n <= K * L), (n, L, K, st, sp)
            assert all(0 <= s < e <= n and e - s <= L for s, e in sp), (n, L, K, st, sp)
            assert all(b[0] > a[0] for a, b in zip(sp, sp[1:])), f"not in document order: {sp}"
            if n <= K * L:
                assert sp[0][0] == 0 and sp[-1][1] == n and sum(e - s for s, e in sp) == n, "short case not fully covered"
            else:
                assert len(sp) == K and all(e - s == L for s, e in sp)
                if st in ("tail", "headtail"):
                    assert sp[-1][1] == n, "tail chunk must end at the last token"
                if st == "head" or (st == "headtail" and K > 1):
                    assert sp[0][0] == 0
                if ov == 0.0 and st != "uniform":
                    assert all(b[0] >= a[1] for a, b in zip(sp, sp[1:])), f"chunks overlap without overlap: {sp}"
    assert select_chunks(10_000, 512, 4, "headtail") == [(0, 512), (10_000 - 1536, 10_000 - 1024),
                                                         (10_000 - 1024, 10_000 - 512), (10_000 - 512, 10_000)]
    sch = doc_schedule(10, {"seed": 0, "epochs": 3, "docs_per_epoch": 4})
    assert [len(e) for e in sch] == [4, 4, 4] and len(set(np.concatenate(sch[:2]).tolist())) == 8, "epochs must see new cases"
    print("ok  chunk selection: in bounds, ordered, short cases fully covered, head/tail anchored; epoch schedule")


def test_data_formats():
    assert [_label(v, "") for v in (0, 1, "0", "1", True, 1.0, [1])] == [0, 1, 0, 1, 1, 1, 1]
    for bad in (2, "accepted", None, -1):
        try:
            _label(bad, ""); raise AssertionError(f"label {bad!r} accepted")
        except ValueError:
            pass
    recs = _records({"c1": {"text": "a  b\n\nc", "label": "1"}, "c2": {"text": ["p1", "p2"], "label": 0}})
    cs = [norm_case(r, "x", True) for r in recs]
    assert cs[0] == {"id": "c1", "raw_id": "c1", "text": "a b c", "label": 1} and cs[1]["text"] == "p1 p2"
    c = norm_case({"id": 7, "text": "t"}, "x", False)
    assert c["raw_id"] == 7 and c["id"] == "7" and c["label"] is None
    print("ok  data loader: list / dict / paragraph-list formats, label spellings, whitespace normalisation, raw ids kept")


def test_metrics():
    rng = np.random.default_rng(0)
    from sklearn.metrics import f1_score, precision_score, recall_score
    for _ in range(50):
        g = rng.integers(0, 2, 40); p = rng.integers(0, 2, 40)
        r = binary_report(g, p)
        assert abs(r["macro_f1"] - f1_score(g, p, average="macro", zero_division=0)) < 1e-9
        assert abs(r["macro_p"] - precision_score(g, p, average="macro", zero_division=0)) < 1e-9
        assert abs(r["macro_r"] - recall_score(g, p, average="macro", zero_division=0)) < 1e-9
        assert np.array(r["confusion"]).sum() == 40
    g = np.array([0, 0, 0, 1, 1, 1]); prob = np.array([0.1, 0.2, 0.35, 0.3, 0.6, 0.9])
    t, f = best_threshold(g, prob)
    assert f == binary_report(g, (prob >= t).astype(int))["macro_f1"] and f >= binary_report(g, (prob >= 0.5).astype(int))["macro_f1"]
    r = binary_report([1, 1, 1], [1, 1, 1])
    assert r["macro_f1"] == 0.5, "absent class must count as F1 = 0 (sklearn labels=[0,1] convention)"
    print("ok  macro-F1 / macro-P / macro-R equal sklearn; threshold search never worse than 0.5 on dev")


def test_submission():
    with tempfile.TemporaryDirectory() as td:
        ids = ["case_001", "case_002", 3]
        zp = write_submission(ids, np.array([1, 0, 1]), td)
        import zipfile
        obj = json.loads(zipfile.ZipFile(zp).read("prediction.json"))
        assert obj == [{"id": "case_001", "pred": 1}, {"id": "case_002", "pred": 0}, {"id": 3, "pred": 1}]
        for bad in ([{"id": "case_001", "pred": 1}, {"id": "case_002", "pred": 0}],
                    [{"id": "case_001", "pred": 1}, {"id": "case_001", "pred": 0}, {"id": 3, "pred": 1}],
                    [{"id": "case_001", "pred": 1.0}, {"id": "case_002", "pred": 0}, {"id": 3, "pred": 1}],
                    [{"id": "case_001", "pred": 2}, {"id": "case_002", "pred": 0}, {"id": 3, "pred": 1}],
                    [{"id": "case_001", "pred": 1}, {"id": "case_002", "pred": 0}, {"id": "3", "pred": 1}]):
            try:
                check_prediction(bad, ids); raise AssertionError(f"accepted bad submission {bad}")
            except AssertionError as e:
                if "accepted bad" in str(e):
                    raise
    print("ok  submission: list of {id, pred}, ids exactly as in test.json (type kept), int 0/1, zip root")


def test_peft_identity():
    torch.manual_seed(0)
    base = torch.nn.Linear(16, 48, bias=False)
    x = torch.randn(2, 5, 16)
    y0 = base(x)
    for mod in (peft.LoRALinear(base, 4, 8), peft.LoRMALinear(base, 4, 8, "plus"), peft.LoRMALinear(base, 4, 8, "pi"),
                peft.HoulsbyAdapter(base, 8), peft.BitFitLinear(base),
                peft.IA3Linear(base, "out", torch.cat([torch.zeros(16), torch.ones(32)]))):
        assert torch.allclose(mod(x), y0, atol=1e-5), type(mod).__name__
    print("ok  LoRA / LoRMA+ / LoRMApi / adapter / BitFit / IA3 start as the identity")


def tiny_ckpts(td):
    from p4cjpe.phase1_models import MODEL_CLASSES
    torch.manual_seed(0)
    out = {}
    for cls, kw in TINY.items():
        path = os.path.join(td, cls + ".pt")
        torch.save({"model_class": cls, "model_kw": kw, "model": MODEL_CLASSES[cls](**kw).state_dict()}, path)
        out[cls] = path
    return out


def test_model_invariance():
    """Tiny random A/B/C/D: a case's logits do not depend on what else is in the batch (token padding, chunk-slot
    padding), every method trains only what it should, and KD reduces to CE for alpha = 1."""
    from p4cjpe.model import build_model
    head = {"chunk_pool": "attn", "agg": "attn", "use_pos": True, "pos_buckets": 8, "dropout": 0.1, "hidden": 16, "att_hidden": 8}
    cc = {"len": 16, "k": 3, "strategy": "headtail", "overlap": 0.0}
    rng = np.random.default_rng(0)
    docs = [rng.integers(0, 20000, n).astype(np.uint16) for n in (5, 40, 200, 16, 33)]
    methods = [{"name": "full"}, {"name": "lorma", "r": 2, "alpha": 4}, {"name": "adapter", "bottleneck": 4},
               {"name": "bitfit"}, {"name": "partial", "train_last_k": 1}, {"name": "lora", "r": 2, "alpha": 4}]
    with tempfile.TemporaryDirectory() as td:
        paths = tiny_ckpts(td)
        for cls, path in paths.items():
            for att in ("bidirectional", "causal"):
                for agg in ("attn", "gru", "mean", "max"):
                    cfg = {"backbone": {"ckpt": path, "attention": att}, "method": {"name": "full"}, "head": {**head, "agg": agg}}
                    model, _ = build_model(cfg)
                    with torch.no_grad():                      # non-trivial head weights
                        for p in model.head.parameters():
                            p.add_(torch.randn_like(p) * 0.1)
                    model.eval()
                    with torch.no_grad():
                        batch = collate(docs, cc, 8, pad_id=31)
                        lg_all, _ = model(*batch, len(docs))
                        lg_cb, _ = model(*batch, len(docs), chunk_bs=2)
                        assert torch.allclose(lg_all, lg_cb, atol=1e-4), f"{cls}/{att}/{agg}: chunk slicing changes logits"
                        for i, d in enumerate(docs):
                            lg1, _ = model(*collate([d], cc, 8, pad_id=99), 1)
                            assert torch.allclose(lg1[0], lg_all[i], atol=1e-4), f"{cls}/{att}/{agg}: case {i} depends on the batch"
            for m in methods:
                model, _ = build_model({"backbone": {"ckpt": path}, "method": m, "head": head})
                names = [n for n, p in model.named_parameters() if p.requires_grad]
                assert any(n.startswith("head.") for n in names)
                bb = [n for n in names if n.startswith("backbone.")]
                if m["name"] == "full":
                    assert len(bb) == sum(1 for n, _ in model.backbone.named_parameters())
                elif m["name"] == "partial":
                    assert bb and all(".blocks.1." in n or n.startswith("backbone.norm.") for n in bb), bb
                elif m["name"] == "bitfit":
                    assert bb and all("bitfit_" in n or "norm" in n for n in bb), bb
                else:
                    assert bb and all(peft.is_new_param(n) for n in bb), (m, bb)
    lg = torch.randn(4, 2); y = torch.tensor([0, 1, 1, 0])
    m = CJPEModel.__new__(CJPEModel); torch.nn.Module.__init__(m); m.register_buffer("class_w", torch.ones(2))
    ce = m.loss(lg, y, {})
    assert torch.allclose(m.loss(lg, y, {"kd_alpha": 1.0}, lg.clone() * 3), ce)
    assert torch.allclose(m.loss(lg, y, {"kd_alpha": 0.0, "kd_T": 2.0}, lg.clone()), torch.tensor(0.0), atol=1e-6)
    print("ok  A/B/C/D x causal/bidirectional x mean/max/attn/gru: logits independent of batch padding and chunk slicing; "
          "full/partial/bitfit/adapter/LoRA/LoRMA train the right parameters; KD loss limits")


def test_backbones(ckpt_dir):
    from p4cjpe.backbones import load_backbone, load_phase1
    from p4cjpe.tokenizer import BPETokenizer
    for folder in sorted(os.listdir(ckpt_dir)):
        p = os.path.join(ckpt_dir, folder, "model_final.pt")
        if not os.path.exists(p):
            continue
        tok = BPETokenizer.load(os.path.join(ckpt_dir, folder, "tokenizer.json"))
        ids = torch.tensor([tok.encode(" The appellant filed the appeal before the High Court under section 260A.")])
        full, cls, kw = load_phase1(p)
        full.eval()
        bb, info = load_backbone(p, attention="causal")
        bb.eval()
        with torch.no_grad():
            if cls == "EncDecLM":
                ref = full.encode(ids)
                bb2, _ = load_backbone(p, attention="bidirectional"); bb2.eval()
                ok = torch.allclose(bb2(ids, torch.ones_like(ids, dtype=torch.bool)), ref, atol=1e-4)
            elif cls == "HybridLM":
                bbc, _ = load_backbone(p, attention="causal", c_segment=1); bbc.eval()
                hc = bbc(ids, torch.ones_like(ids, dtype=torch.bool))
                ok = torch.allclose(hc @ full.lm_head.weight.T, full(ids)[0], atol=1e-3)
            else:
                h = bb(ids, torch.ones_like(ids, dtype=torch.bool))
                ok = torch.allclose(h @ full.lm_head.weight.T, full(ids)[0], atol=1e-3)
        print(f"{'ok ' if ok else 'BAD'} backbone {folder:22s} ({cls}) reproduces the Phase-1 model | params {info['params'] / 1e6:.1f}M"
              f" | positions {info['max_len']}")
        assert ok
        assert info["max_len"] >= 512, "512-token chunks need >= 512 positions"
        del full, bb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="")
    a = ap.parse_args()
    test_chunks(); test_data_formats(); test_metrics(); test_submission(); test_peft_identity(); test_model_invariance()
    if a.ckpt_dir:
        test_backbones(a.ckpt_dir)
    print("all checks passed")
