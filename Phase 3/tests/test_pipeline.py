import argparse
import itertools
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from p3rr.model import CRF, crf_log_marginals_np, viterbi_np
from p3rr.metrics import scores
from p3rr.train import sent_windows, truncate
from p3rr import peft

def brute(emis, trans, start, end):
    N, K = emis.shape
    paths = list(itertools.product(range(K), repeat=N))
    sc = np.array([start[p[0]] + sum(emis[t, p[t]] for t in range(N)) + sum(trans[p[t - 1], p[t]] for t in range(1, N)) + end[p[-1]]
                   for p in paths])
    logZ = np.log(np.exp(sc - sc.max()).sum()) + sc.max()
    marg = np.zeros((N, K))
    for p, s in zip(paths, sc):
        for t in range(N):
            marg[t, p[t]] += math.exp(s - logZ)
    return paths[int(sc.argmax())], marg, logZ, sc

def test_crf():
    rng = np.random.default_rng(0)
    for N in (1, 2, 4):
        K = 3
        emis, trans, start, end = rng.normal(size=(N, K)), rng.normal(size=(K, K)), rng.normal(size=K), rng.normal(size=K)
        best, marg, logZ, sc = brute(emis, trans, start, end)
        assert list(best) == viterbi_np(emis, trans, start, end), "viterbi != brute force"
        assert np.allclose(np.exp(crf_log_marginals_np(emis, trans, start, end)), marg, atol=1e-5), "marginals != brute force"
        crf = CRF(K)
        with torch.no_grad():
            crf.trans.copy_(torch.tensor(trans)); crf.start.copy_(torch.tensor(start)); crf.end.copy_(torch.tensor(end))
        tags = torch.tensor([best]); e = torch.tensor(emis, dtype=torch.float32)[None]
        nll = crf.nll(e, tags, torch.ones(1, N, dtype=torch.bool)).item() * N
        assert abs(nll - (logZ - sc.max())) < 1e-4, (nll, logZ - sc.max())
        e2 = torch.cat([e, torch.randn(1, 2, K)], 1); t2 = torch.cat([tags, torch.zeros(1, 2, dtype=torch.long)], 1)
        m2 = torch.tensor([[True] * N + [False, False]])
        assert abs(crf.nll(e2, t2, m2).item() * N - nll) < 1e-4
    print("ok  CRF: viterbi, forward-backward marginals and NLL match brute force (incl. padding)")

def test_windows():
    rng = np.random.default_rng(1)
    for _ in range(200):
        lens = rng.integers(1, 128, size=rng.integers(1, 60))
        L = int(rng.choice([128, 256, 512]))
        for ov in (0.0, 0.25, 0.5):
            for st in (0, 3):
                ws = sent_windows(lens, L, ov, start=min(st, len(lens) - 1))
                covered = set(k for i, j in ws for k in range(i, j))
                assert covered == set(range(min(st, len(lens) - 1), len(lens))), "a sentence is not covered"
                assert all(lens[i:j].sum() <= L or j - i == 1 for i, j in ws), "window over budget"
                assert all(b[0] > a[0] for a, b in zip(ws, ws[1:])), "windows do not advance"
    x = np.arange(300)
    t = truncate(x, 128, 0.75)
    assert len(t) == 128 and t[0] == 0 and t[95] == 95 and t[96] == 300 - 32 and t[-1] == 299
    print("ok  sentence windows cover every sentence, respect the token budget; truncation keeps head+tail")

def test_metrics():
    gold = {"a": ["Fact", "Fact", "Issue", "None"], "b": ["Statute", "Fact"]}
    pred = {"a": ["Fact", "Issue", "Issue", "Fact"], "b": ["Statute", "Dissent"]}
    labels = ["Fact", "Issue", "Statute", "Dissent", "None", "RatioOfTheDecision"]
    sc = scores(gold, pred, labels)
    try:
        from sklearn.metrics import f1_score
        g = sum(gold.values(), []); p = sum(pred.values(), [])
        ref = f1_score(g, p, average="macro", zero_division=0)
        assert abs(sc["macro_f1"] - ref) < 1e-9, (sc["macro_f1"], ref)
        print(f"ok  macro-F1 equals sklearn f1_score(average='macro') ({ref:.4f})")
    except ImportError:
        print("--  sklearn not installed: metric cross-check skipped")

def test_peft_identity():
    torch.manual_seed(0)
    base = torch.nn.Linear(16, 48, bias=False)
    x = torch.randn(2, 5, 16)
    y0 = base(x)
    for mod in (peft.LoRALinear(base, 4, 8), peft.LoRALinear(base, 4, 8, rslora=True), peft.LoRMALinear(base, 4, 8, "plus"),
                peft.IA3Linear(base, "out", torch.cat([torch.zeros(16), torch.ones(32)])), peft.IA3Linear(base, "in")):
        assert torch.allclose(mod(x), y0, atol=1e-6), type(mod).__name__
    assert peft.LoRALinear(base, 16, 8, rslora=True).scale == 2.0 and peft.LoRALinear(base, 16, 8).scale == 0.5
    m = peft.IA3Linear(base, "out", torch.cat([torch.zeros(16), torch.ones(32)]))
    with torch.no_grad():
        m.ia3_l.fill_(2.0)
    y = m(x)
    assert torch.allclose(y[..., :16], y0[..., :16]) and torch.allclose(y[..., 16:], 2 * y0[..., 16:]), "IA3 must not scale q"
    print("ok  LoRA / rsLoRA / LoRMA+ / IA3 start as the identity; IA3 leaves the query block unscaled")

def test_prefix_masks():
    import tempfile
    from p3rr.backbones import load_backbone
    from p3rr.phase1_models import MODEL_CLASSES
    kws = {"LlamaLM": dict(vocab_size=20100, block_size=64, n_layer=2, n_head=4, n_kv_head=2, d_model=32),
           "EncDecLM": dict(vocab_size=20100, enc_block=64, dec_block=32, n_enc_layer=2, n_dec_layer=1, n_head=4, d_model=32),
           "HybridLM": dict(vocab_size=20100, block_size=64, n_layer=2, n_head=4, d_model=32),
           "EfficientLM": dict(vocab_size=20100, block_size=64, window=8, n_layer=2, n_head=4, n_kv_head=1, d_model=32)}
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as td:
        for cls, kw in kws.items():
            path = os.path.join(td, cls + ".pt")
            torch.save({"model_class": cls, "model_kw": kw, "model": MODEL_CLASSES[cls](**kw).state_dict()}, path)
            for att in ("bidirectional", "causal"):
                bb, _ = load_backbone(path, attention=att)
                base_out = None
                peft.apply_method(bb, {"name": "prefix", "n_prefix": 4})
                bb.eval()
                ids = torch.randint(0, 20000, (2, 20))
                keep = torch.ones(2, 20, dtype=torch.bool); keep[1, 14:] = False
                with torch.no_grad():
                    h = bb(ids, keep)
                    ids2 = ids.clone(); ids2[1, 14:] = 7
                    assert torch.allclose(h[1, :14], bb(ids2, keep)[1, :14], atol=1e-5), f"{cls}/{att}: padding leaks"
                    if att == "causal":
                        ids3 = ids.clone(); ids3[0, 10:] = 5
                        assert torch.allclose(h[0, :10], bb(ids3, keep)[0, :10], atol=1e-5), f"{cls}: future leaks"
                    saved = bb.prefix_h; bb.prefix_h = None
                    base_out = bb(ids, keep); bb.prefix_h = saved
                    assert not torch.allclose(h, base_out), f"{cls}/{att}: prefix has no effect"
                n_train = sum(p.numel() for n, p in bb.named_parameters() if p.requires_grad)
                assert n_train == 2 * 4 * 32 and all(peft.is_new_param(n) for n, p in bb.named_parameters() if p.requires_grad)
    print("ok  prefix tuning (A/B/C/D, causal + bidirectional): no padding / future leak, only prefix states trainable")

def test_backbones(ckpt_dir):
    from p3rr.backbones import load_backbone, load_phase1
    from p3rr.tokenizer import BPETokenizer
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
            h = bb(ids, torch.ones_like(ids, dtype=torch.bool))
            if cls == "EncDecLM":
                ref = full.encode(ids)
                bb2, _ = load_backbone(p, attention="bidirectional"); bb2.eval()
                h = bb2(ids, torch.ones_like(ids, dtype=torch.bool))
                ok = torch.allclose(h, ref, atol=1e-4)
            else:
                logits = full(ids)[0]
                ok = torch.allclose(bb.norm.weight.new_tensor(0) + h @ full.lm_head.weight.T, logits, atol=1e-3) if cls != "HybridLM" else True
                if cls == "HybridLM":
                    bbc, _ = load_backbone(p, attention="causal", c_segment=1); bbc.eval()
                    hc = bbc(ids, torch.ones_like(ids, dtype=torch.bool))
                    ok = torch.allclose(hc @ full.lm_head.weight.T, logits, atol=1e-3)
        print(f"{'ok ' if ok else 'BAD'} backbone {folder:22s} ({cls}) reproduces the Phase-1 model | params {info['params'] / 1e6:.1f}M")
        assert ok
        del full, bb

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="")
    a = ap.parse_args()
    test_crf(); test_windows(); test_metrics(); test_peft_identity(); test_prefix_masks()
    if a.ckpt_dir:
        test_backbones(a.ckpt_dir)
    print("all checks passed")
