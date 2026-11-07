import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from p5lsi import peft
from p5lsi.data import _labels, _records, label_space, multi_hot, norm_case
from p5lsi.metrics import (apply_thresholds, best_global_threshold, best_label_thresholds, crossfit_strategies,
                           multilabel_report)
from p5lsi.model import LSIModel
from p5lsi.train import (check_prediction, collate, doc_schedule, preds_to_ids, rare_weights, select_chunks,
                         thr_from_json, thr_to_json, write_submission)

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
                assert sp[0][0] == 0 and sp[-1][1] == n and sum(e - s for s, e in sp) == n, "short document not fully covered"
            else:
                assert len(sp) == K and all(e - s == L for s, e in sp)
    sch = doc_schedule(10, {"seed": 0, "epochs": 3, "docs_per_epoch": 4})
    assert [len(e) for e in sch] == [4, 4, 4] and len(set(np.concatenate(sch[:2]).tolist())) == 8, "epochs must see new documents"
    Y = np.zeros((200, 3), dtype=np.float32); Y[:, 0] = 1; Y[:5, 2] = 1
    w = rare_weights(Y)
    assert abs(w.sum() - 1) < 1e-9 and w[0] > w[100] * 5, "rare-statute documents must be sampled more often"
    sch = doc_schedule(200, {"seed": 0, "epochs": 2, "docs_per_epoch": 50, "sampler": "rare"}, Y)
    assert all(len(set(e.tolist())) == 50 for e in sch), "rare sampler: no repeats inside an epoch"
    print("ok  chunk selection; uniform epoch schedule sees new documents; rare sampler favours rare statutes")

def test_data_formats():
    assert _labels([7, 1, 7, "23"], "") == [1, 7, 23]
    assert _labels(5, "") == [5] and _labels("1, 7", "") == [1, 7] and _labels("[3]", "") == [3] and _labels([], "") == []
    assert _labels([4.0], "") == [4]
    for bad in (["Section 302"], [True], [1.5], [None]):
        try:
            _labels(bad, ""); raise AssertionError(f"labels {bad!r} accepted")
        except ValueError:
            pass
    recs = _records({"c1": {"text": "a  b\n\nc", "labels": [3, 1]}, "c2": {"text": ["s1", "s2"], "labels": [2]}})
    cs = [norm_case(r, "x", True) for r in recs]
    assert cs[0] == {"id": "c1", "raw_id": "c1", "text": "a b c", "labels": [1, 3]} and cs[1]["text"] == "s1 s2"
    c = norm_case({"id": 7, "text": "t"}, "x", False)
    assert c["raw_id"] == 7 and c["id"] == "7" and c["labels"] is None
    data = {"train": {"labels": [[5, 1], [9]]}, "dev": {"labels": [[1, 12]]}}
    sp = label_space(data)
    assert sp == [1, 5, 9, 12]
    assert multi_hot([[5, 1], [12, 99]], sp).tolist() == [[1, 1, 0, 0], [0, 0, 0, 1]]
    print("ok  data loader: list/dict/sentence-list formats, statute-id spellings, unique sorted labels, label space, multi-hot")

def test_metrics():
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    rng = np.random.default_rng(0)
    for _ in range(60):
        n, L = int(rng.integers(5, 60)), int(rng.integers(2, 12))
        G = rng.random((n, L)) < rng.uniform(0.05, 0.5); P = rng.random((n, L)) < rng.uniform(0.05, 0.5)
        G[:, 0] = False
        r = multilabel_report(G, P)
        for k, fn in (("macro_f1", f1_score), ("macro_p", precision_score), ("macro_r", recall_score)):
            assert abs(r[k] - fn(G, P, average="macro", zero_division=0)) < 1e-9, k
        assert abs(r["micro_f1"] - f1_score(G, P, average="micro", zero_division=0)) < 1e-9
        assert abs(r["subset_acc"] - accuracy_score(G, P)) < 1e-9
        pres = G.sum(0) > 0
        if pres.sum() >= 2:
            assert abs(r["macro_f1_present"] - f1_score(G[:, pres], P[:, pres], average="macro", zero_division=0)) < 1e-9
    G = rng.random((300, 6)) < np.array([0.4, 0.2, 0.1, 0.05, 0.02, 0.3])
    prob = np.clip(G * 0.35 + rng.random(G.shape) * 0.6, 0, 1)
    t, f = best_global_threshold(G, prob)
    assert f >= multilabel_report(G, apply_thresholds(prob, 0.5))["macro_f1"] - 1e-12
    thr = best_label_thresholds(G, prob, t, min_support=3)
    f_pl = multilabel_report(G, prob >= thr)["macro_f1"]
    assert f_pl >= multilabel_report(G, prob >= t)["macro_f1"] - 1e-12, "per-label thresholds must not be worse in-sample"
    assert np.all(thr[G.sum(0) < 3] == t)
    cf = crossfit_strategies(G, prob)
    assert set(cf) == {"fixed", "global", "per_label"} and all(0 <= v <= 1 for v in cf.values())
    from p5lsi.train import calibrate
    thr_c, strat, cf_c = calibrate(G, prob, {"strategy": "auto"})
    assert strat == max(cf_c, key=lambda k: (cf_c[k], -["fixed", "global", "per_label"].index(k)))
    assert calibrate(G, prob, {"strategy": "global"})[1] == "global" and np.ndim(calibrate(G, prob, {"strategy": "per_label"})[0]) == 1
    P = apply_thresholds(np.array([[0.1, 0.3], [0.9, 0.2]]), 0.5)
    assert P.tolist() == [[False, True], [True, False]], "empty prediction must fall back to the argmax statute"
    assert thr_from_json(thr_to_json(thr, [3, 5, 7, 9, 11, 13]), [3, 5, 7, 9, 11, 13]).tolist() == thr.round(4).tolist()
    print("ok  macro-F1/P/R, micro-F1, subset accuracy equal sklearn; global/per-label thresholds never worse in-sample; "
          "argmax fallback; cross-fitting")

def test_submission():
    space = [1, 7, 23, 81]
    with tempfile.TemporaryDirectory() as td:
        ids = ["case_001", "case_002", 3]
        pl = preds_to_ids(np.array([[1, 1, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]), space)
        assert pl == [[1, 7, 23], [81], [1]] and all(type(x) is int for p in pl for x in p)
        zp = write_submission(ids, pl, td, space)
        import zipfile
        obj = json.loads(zipfile.ZipFile(zp).read("prediction.json"))
        assert obj == [{"id": "case_001", "pred": [1, 7, 23]}, {"id": "case_002", "pred": [81]}, {"id": 3, "pred": [1]}]
        good = [{"id": "case_001", "pred": [1]}, {"id": "case_002", "pred": []}, {"id": 3, "pred": [7]}]
        check_prediction(good, ids, space)
        for bad in ([{"id": "case_001", "pred": [1]}, {"id": "case_002", "pred": [7]}],
                    good[:2] + [{"id": "case_001", "pred": [1]}],
                    good[:2] + [{"id": "3", "pred": [1]}],
                    good[:2] + [{"id": 3, "pred": 1}],
                    good[:2] + [{"id": 3, "pred": [1.0]}],
                    good[:2] + [{"id": 3, "pred": [1, 1]}],
                    good[:2] + [{"id": 3, "pred": [2]}]):
            try:
                check_prediction(bad, ids, space); raise AssertionError(f"accepted bad submission {bad}")
            except AssertionError as e:
                if "accepted bad" in str(e):
                    raise
    print("ok  submission: list of {id, pred:[int]}, ids exactly as in test.json (type kept), unique valid statute ids, zip root")

def test_peft_identity():
    torch.manual_seed(0)
    base = torch.nn.Linear(16, 48, bias=False)
    x = torch.randn(2, 5, 16)
    y0 = base(x)
    for mod in (peft.LoRALinear(base, 4, 8), peft.LoRMALinear(base, 4, 8, "plus"), peft.LoRMALinear(base, 4, 8, "pi"),
                peft.HoulsbyAdapter(base, 8), peft.BitFitLinear(base),
                peft.IA3Linear(base, "out", torch.cat([torch.zeros(16), torch.ones(32)]))):
        assert torch.allclose(mod(x), y0, atol=1e-5), type(mod).__name__
    w = torch.randn(64, 128)
    packed, absmax = peft.nf4_quantize(w)
    wq = peft.nf4_dequantize(packed, absmax, w.shape)
    assert packed.dtype == torch.uint8 and packed.numel() == w.numel() // 2
    rel = ((wq - w).norm() / w.norm()).item()
    assert rel < 0.12, f"NF4 relative error {rel:.3f}"
    codes = torch.tensor(peft.NF4_CODES)
    blk = (w.view(-1, 64) / w.view(-1, 64).abs().amax(1, keepdim=True)).reshape(-1)
    nearest = codes[(blk[:, None] - codes[None]).abs().argmin(1)]
    assert torch.allclose((wq.view(-1, 64) / absmax[:, None]).reshape(-1), nearest, atol=1e-6), "nearest NF4 code"
    big = torch.nn.Linear(128, 64, bias=False)
    q = peft.QLoRALinear(big, 4, 8)
    xx = torch.randn(3, 128)
    assert torch.allclose(q(xx), xx @ q.dequantized().T, atol=1e-5)
    assert ((q(xx) - big(xx)).norm() / big(xx).norm()).item() < 0.15
    assert [n for n, _ in q.named_parameters()] == ["lora_A", "lora_B"] and not q.state_dict().get("nf4_packed")
    print(f"ok  LoRA / LoRMA+ / LoRMApi / adapter / BitFit / IA3 start as the identity; QLoRA NF4 round trip "
          f"(rel. error {rel:.3f}), packed 2 codes/byte, only LoRA weights trainable")

def tiny_ckpts(td):
    from p5lsi.phase1_models import MODEL_CLASSES
    torch.manual_seed(0)
    out = {}
    for cls, kw in TINY.items():
        path = os.path.join(td, cls + ".pt")
        torch.save({"model_class": cls, "model_kw": kw, "model": MODEL_CLASSES[cls](**kw).state_dict()}, path)
        out[cls] = path
    return out

def test_model_invariance():
    from p5lsi.model import build_model
    NL = 7
    head = {"chunk_pool": "attn", "agg": "attn", "use_pos": True, "pos_buckets": 8, "dropout": 0.1, "hidden": 16, "att_hidden": 8}
    cc = {"len": 16, "k": 3, "strategy": "headtail", "overlap": 0.0}
    rng = np.random.default_rng(0)
    docs = [rng.integers(0, 20000, n).astype(np.uint16) for n in (5, 40, 200, 16, 33)]
    methods = [{"name": "full"}, {"name": "lorma", "r": 2, "alpha": 4}, {"name": "adapter", "bottleneck": 4},
               {"name": "partial", "train_last_k": 1}, {"name": "lora", "r": 2, "alpha": 4},
               {"name": "qlora", "r": 2, "alpha": 4}, {"name": "prefix", "n_prefix": 3}]
    with tempfile.TemporaryDirectory() as td:
        paths = tiny_ckpts(td)
        for cls, path in paths.items():
            for att in ("bidirectional", "causal"):
                for agg in ("attn", "gru", "mean", "max", "label_attn"):
                    cfg = {"backbone": {"ckpt": path, "attention": att}, "method": {"name": "full"}, "head": {**head, "agg": agg}}
                    model, _ = build_model(cfg, NL)
                    model.head.init_bias(np.linspace(0.01, 0.4, NL))
                    with torch.no_grad():
                        for p in model.head.parameters():
                            p.add_(torch.randn_like(p) * 0.1)
                    model.eval()
                    with torch.no_grad():
                        batch = collate(docs, cc, 8, pad_id=31)
                        lg_all, _ = model(*batch, len(docs))
                        assert lg_all.shape == (len(docs), NL)
                        lg_cb, _ = model(*batch, len(docs), chunk_bs=2)
                        assert torch.allclose(lg_all, lg_cb, atol=1e-4), f"{cls}/{att}/{agg}: chunk slicing changes logits"
                        for i, d in enumerate(docs):
                            lg1, _ = model(*collate([d], cc, 8, pad_id=99), 1)
                            assert torch.allclose(lg1[0], lg_all[i], atol=1e-4), f"{cls}/{att}/{agg}: document {i} depends on the batch"
            for m in methods:
                model, _ = build_model({"backbone": {"ckpt": path}, "method": m, "head": head}, NL)
                names = [n for n, p in model.named_parameters() if p.requires_grad]
                assert any(n.startswith("head.") for n in names)
                bb = [n for n in names if n.startswith("backbone.")]
                if m["name"] == "full":
                    assert len(bb) == sum(1 for n, _ in model.backbone.named_parameters())
                elif m["name"] == "partial":
                    assert bb and all(".blocks.1." in n or n.startswith("backbone.norm.") for n in bb), bb
                else:
                    assert bb and all(peft.is_new_param(n) for n in bb), (m, bb)
                if m["name"] == "qlora":
                    assert not any(isinstance(x, torch.nn.Linear) for x in model.backbone.blocks.modules()), "fp32 linears left"
                lg, _ = model(*collate(docs, cc, 8, pad_id=0), len(docs))
                lg.sum().backward()
                assert all(p.grad is not None for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")), m
        model, _ = build_model({"backbone": {"ckpt": paths["LlamaLM"]}, "method": {"name": "full"}, "head": head}, 3)
        model.head.init_bias([0.5, 0.1, 0.0])
        assert torch.allclose(torch.sigmoid(model.head.out.bias), torch.tensor([0.5, 0.1, 1e-4]), atol=1e-6)
    m = LSIModel.__new__(LSIModel); torch.nn.Module.__init__(m); m.register_buffer("pos_weight", torch.ones(4))
    lg = torch.randn(5, 4); y = (torch.rand(5, 4) < 0.3).float()
    bce = m.loss(lg, y, {"name": "bce"})
    ref = torch.nn.functional.binary_cross_entropy_with_logits(lg, y, reduction="sum") / 5
    assert torch.allclose(bce, ref, atol=1e-6), "BCE = sum over statutes, mean over documents"
    asl0 = m.loss(lg, y, {"name": "asl", "gamma_pos": 0.0, "gamma_neg": 0.0, "asl_clip": 0.0})
    assert torch.allclose(asl0, bce, atol=1e-5), "ASL with gamma=0, clip=0 must equal BCE"
    asl = m.loss(lg, y, {"name": "asl"})
    assert asl < bce, "ASL down-weights easy negatives"
    assert torch.allclose(m.loss(lg, y, {"name": "bce", "kd_alpha": 1.0}, lg.clone() * 3), bce)
    assert torch.allclose(m.loss(lg, y, {"name": "bce", "kd_alpha": 0.0, "kd_T": 2.0}, lg.clone()), torch.tensor(0.0), atol=1e-6)
    assert m.loss(lg, y, {"name": "bce", "kd_alpha": 0.0}, -lg) > 0.1
    print("ok  A/B/C/D x causal/bidirectional x mean/max/attn/gru/label_attn: logits independent of batch padding and chunk "
          "slicing; full/partial/adapter/LoRA/LoRMA/QLoRA/prefix train the right parameters; prior bias; BCE/ASL/KD limits")

def test_backbones(ckpt_dir):
    from p5lsi.backbones import load_backbone, load_phase1
    from p5lsi.tokenizer import BPETokenizer
    for folder in sorted(os.listdir(ckpt_dir)):
        p = os.path.join(ckpt_dir, folder, "model_final.pt")
        if not os.path.exists(p):
            continue
        tok = BPETokenizer.load(os.path.join(ckpt_dir, folder, "tokenizer.json"))
        ids = torch.tensor([tok.encode(" The appellant was convicted under section 302 of the Indian Penal Code.")])
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
