import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from p6summ import peft
from p6summ.context import _cost, fid_chunks, select_source, select_spans
from p6summ.data import _records, norm_doc, sentence_scores, split_sentences
from p6summ.generate import GEN_DEFAULTS, generate, process_logits
from p6summ.metrics import legal_facts, redundancy, rouge_fallback, rouge_scores
from p6summ.model import Summarizer, load_phase1, map_state, seq_loss
from p6summ.phase1_models import MODEL_CLASSES
from p6summ.train import check_prediction, collate, make_target, postprocess, write_submission

V = 300
TINY = {"LlamaLM": dict(vocab_size=V, block_size=64, n_layer=2, n_head=4, n_kv_head=2, d_model=32),
        "EncDecLM": dict(vocab_size=V, enc_block=32, dec_block=32, n_enc_layer=2, n_dec_layer=2, n_head=4, d_model=32),
        "HybridLM": dict(vocab_size=V, block_size=64, n_layer=2, n_head=4, d_model=32),
        "EfficientLM": dict(vocab_size=V, block_size=64, window=8, n_layer=2, n_head=4, n_kv_head=1, d_model=32)}
TMP = tempfile.mkdtemp()

class FakeTok:
    special_to_id = {"<|endoftext|>": V - 5, "<|pad|>": V - 4, "<|bos|>": V - 3, "<|sep|>": V - 2, "<|mask|>": V - 1}
    pad_id, bos_id, mask_id = V - 4, V - 3, V - 1

    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids)

def tiny_ckpt(cls, seed=0):
    torch.manual_seed(seed)
    m = MODEL_CLASSES[cls](**TINY[cls])
    if cls == "HybridLM":
        torch.nn.init.normal_(m.seg_emb.weight, std=0.02)
    p = os.path.join(TMP, f"{cls}_{seed}.pt")
    torch.save({"model_class": cls, "model_kw": TINY[cls], "model": m.state_dict()}, p)
    return p, m.eval()

def test_sentences_and_context():
    rng = np.random.default_rng(0)
    for text in ["The appellant filed a suit. It was dismissed; the High Court allowed it. Held: appeal fails.",
                 "No punctuation at all " * 200, "A. B. C. 1. 2. (a) x", ""]:
        assert "".join(split_sentences(text)) == text
    s = split_sentences("Section 302 IPC applies. The accused was convicted. Appeal dismissed.")
    assert len(s) == 3 and len(sentence_scores(s)) == 3
    for _ in range(3000):
        n = int(rng.integers(1, 5000)); L = int(rng.choice([16, 64, 512, 1400])); gl = int(rng.integers(0, 3))
        m = int(rng.integers(1, 80))
        ends = np.unique(np.sort(np.concatenate([rng.integers(1, n + 1, m), [n]]))).astype(np.int32)
        score = rng.random(len(ends)).astype(np.float32)
        for st in ("head", "tail", "headtail", "uniform", "salient"):
            sp = select_spans(n, L, st, ends, score, float(rng.choice([0.3, 0.5])), int(rng.integers(1, 6)),
                              int(rng.choice([0, 64, 256])), gl)
            assert all(0 <= a < b <= n for a, b in sp), (n, L, st, sp)
            assert all(sp[i][1] < sp[i + 1][0] for i in range(len(sp) - 1)), (st, sp)
            assert _cost(sp, gl) <= max(L, 0) or n <= L, (n, L, st, sp, gl)
            if n <= L:
                assert sp == [(0, n)]
    toks = np.arange(1000, dtype=np.uint16)
    out = select_source(toks, 100, "headtail", gap_ids=[7, 7])
    assert len(out) == 100 and list(out[:49]) == list(range(49)) and list(out[49:51]) == [7, 7] and out[-1] == 999
    assert [len(c) for c in fid_chunks(np.arange(1700), 768)] == [768, 768, 164]
    print("ok  sentences + context selection (spans in budget, ordered, gap markers)")

def test_loader():
    recs = _records({"d1": {"document": ["para one.", "para two."], "summary": "s"}, "d2": {"id": "d2", "text": "x",
                                                                                          "headnote": "h"}})
    a, b = norm_doc(recs[0], "t", True), norm_doc(recs[1], "t", True)
    assert a["id"] == "d1" and a["document"] == "para one. para two." and b["summary"] == "h"
    try:
        norm_doc({"id": 1, "document": "x"}, "t", True); raise AssertionError("missing summary accepted")
    except ValueError:
        pass
    print("ok  loader (dict / list formats, key aliases, missing summary rejected)")

def test_targets_and_collate():
    tok = FakeTok()
    eot = tok.special_to_id["<|endoftext|>"]
    assert make_target([1, 2, 3], 5, eot) == [1, 2, 3, eot] and make_target([1, 2, 3, 4, 5, 6], 5, eot) == [1, 2, 3, 4, 5]
    assert make_target([1, 2, 3, 4], 5, eot) == [1, 2, 3, 4, eot]
    spec = {"sep_ids": [9, 9]}
    exs = [{"src": np.array([5, 6, 7]), "tgt": [1, 2, eot]}, {"src": np.array([5]), "tgt": [3, 4, 8, 8, eot]}]
    b = collate(exs, "A", spec, tok)
    for r in range(2):
        ids, lab = b["ids"][r].tolist(), b["labels"][r].tolist()
        full = list(exs[r]["src"]) + [9, 9] + exs[r]["tgt"]
        for p, l in enumerate(lab):
            if l != -100:
                assert full[p + 1] == l
        assert [l for l in lab if l != -100] == exs[r]["tgt"]
        assert b["prefix_len"][r] == len(exs[r]["src"]) + 2
    bb = collate([{"src": np.arange(70), "tgt": [1, 2, eot]}, {"src": np.arange(10), "tgt": [4, eot]}], "B", spec, tok, 32)
    assert bb["src"].shape == (4, 32) and bb["k_max"] == 3 and bb["doc"].tolist() == [0, 0, 0, 1]
    assert bb["dec_in"][0].tolist() == [tok.bos_id, 1, 2] and bb["labels"][1].tolist() == [4, eot, -100]
    print("ok  targets (<eot> only when the summary fits) + collate alignment (decoder-style and FiD)")

def _build(cls, method=None, seed=0):
    p, ref = tiny_ckpt(cls, seed)
    s = Summarizer(p).eval()
    s.bos_id = FakeTok.bos_id
    if method:
        peft.apply_method(s, method)
    s.set_lengths(32 if cls == "EncDecLM" else 64)
    return s, ref

def test_equivalence():
    for cls in TINY:
        s, m = _build(cls)
        torch.manual_seed(1)
        if cls == "EncDecLM":
            src = torch.randint(0, 290, (2, 30)); keep = torch.ones(2, 30, dtype=torch.bool); keep[1, 20:] = False
            dec = torch.randint(0, 290, (2, 20))
            ref, _ = m(src, keep, dec)
            got, _ = s.forward_encdec(src, keep, torch.arange(2), torch.zeros(2, dtype=torch.long), 2, 1, dec,
                                      torch.zeros(2, 20, dtype=torch.long))
            assert (ref.reshape(-1, V) - got).abs().max() < 1e-4
            st = s.prefill([[src[0]], [src[1, :20]]])
            lg, seq = [st["logits"]], [torch.full((2,), FakeTok.bos_id)]
            for _ in range(12):
                t = torch.randint(0, 290, (2,)); seq.append(t); lg.append(s.step(st, t))
            dec_in = torch.stack(seq, 1)
            for b, L in ((0, 30), (1, 20)):
                r, _ = m(src[b:b + 1, :L], None, dec_in[b:b + 1])
                assert (r[0] - torch.stack([x[b] for x in lg])).abs().max() < 1e-4, cls
            continue
        ids = torch.randint(0, 290, (2, 48))
        pl = torch.tensor([20, 31]) if cls == "HybridLM" else None
        ref, _ = m(ids, pl) if cls == "HybridLM" else m(ids)
        got, _ = s.forward_dec(ids, torch.zeros(2, 48, dtype=torch.long), pl)
        assert (ref.reshape(-1, V) - got).abs().max() < 1e-4, cls
        P = [ids[0, :5], ids[1, :31]]
        st = s.prefill(P)
        lg, toks = [st["logits"]], []
        for _ in range(14):
            t = torch.randint(0, 290, (2,)); toks.append(t); lg.append(s.step(st, t))
        for b in range(2):
            full = torch.cat([P[b], torch.stack(toks, 1)[b]])[None]
            r, _ = m(full, torch.tensor([len(P[b])])) if cls == "HybridLM" else m(full)
            assert (r[0, len(P[b]) - 1:] - torch.stack([x[b] for x in lg])).abs().max() < 1e-4, (cls, b)
    print("ok  Summarizer == Phase-1 forward (A, B, C, D) and KV-cached batched generation == full forward")

def _gen_vs_forward(s, P, n=8):
    st = s.prefill(P)
    lg, toks = [st["logits"]], []
    for _ in range(n):
        t = torch.randint(0, 290, (len(P),)); toks.append(t); lg.append(s.step(st, t))
    for b in range(len(P)):
        full = torch.cat([P[b], torch.stack(toks, 1)[b]])
        lab = torch.zeros(1, len(full), dtype=torch.long)
        lab[0, :len(P[b]) - 1] = -100
        got, _ = s.forward_dec(full[None], lab, torch.tensor([len(P[b])]))
        assert (got - torch.stack([x[b] for x in lg])[:got.size(0)]).abs().max() < 1e-4

def test_peft():
    torch.manual_seed(0)
    ident = [{"name": "lora", "r": 4}, {"name": "lorma", "r": 4}, {"name": "lorma", "variant": "pi", "r": 4},
             {"name": "houlsby", "bottleneck": 8}, {"name": "pfeiffer", "bottleneck": 8}, {"name": "ia3"},
             {"name": "bitfit"}, {"name": "adalora", "r": 4}]
    for cls in TINY:
        base, _ = _build(cls)
        ids = torch.randint(0, 290, (2, 30))
        if cls == "EncDecLM":
            args = (ids, torch.ones(2, 30, dtype=torch.bool), torch.arange(2), torch.zeros(2, dtype=torch.long), 2, 1,
                    ids[:, :12], torch.zeros(2, 12, dtype=torch.long))
            f = lambda m: m.forward_encdec(*args)[0]
        else:
            f = lambda m: m.forward_dec(ids, torch.zeros(2, 30, dtype=torch.long), torch.tensor([10, 12]))[0]
        y0 = f(base)
        for mc in ident + [{"name": "qlora", "r": 4}, {"name": "partial", "train_last_k": 1}]:
            m, _ = _build(cls, mc)
            n_tr = peft.count_params(m)["trainable"]
            assert n_tr > 0, (cls, mc)
            if mc["name"] != "partial":
                assert all(peft.is_new_param(n) for n, p in m.named_parameters() if p.requires_grad
                           and not ("norm" in n and mc["name"] == "bitfit")), (cls, mc)
            d = (f(m) - y0).abs().max().item()
            assert d < (0.5 if mc["name"] == "qlora" else 1e-4), (cls, mc, d)
            out = f(m); out.float().sum().backward()
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters() if p.requires_grad), (cls, mc)
    for cls in ("LlamaLM", "HybridLM"):
        for mc in ({"name": "prefix", "n_prefix": 3}, {"name": "prompt", "n_prompt": 4}):
            m, _ = _build(cls, mc)
            with torch.no_grad():
                for p in m.parameters():
                    if p.requires_grad:
                        p.add_(torch.randn_like(p) * 0.1)
            _gen_vs_forward(m, [torch.randint(0, 290, (7,)), torch.randint(0, 290, (12,))])
    m, _ = _build("EfficientLM", {"name": "prompt", "n_prompt": 4})
    _gen_vs_forward(m, [torch.randint(0, 290, (5,)), torch.randint(0, 290, (12,))])
    print("ok  PEFT methods on A/B/C/D: identity at init, only new params trainable, gradients flow; "
          "prefix/prompt generation == forward")

def test_adalora_controller():
    m, _ = _build("LlamaLM", {"name": "adalora", "r": 6})
    ctl = peft.AdaLoRAController(m, total_steps=100, target_r=2, init_warmup=0.1, final_warmup=0.2, delta_t=5)
    b = [ctl.schedule(t) for t in range(100)]
    assert b[0] == ctl.b0 and b[-1] == ctl.bT and all(x >= y for x, y in zip(b, b[1:]))
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.01)
    ids = torch.randint(0, 290, (2, 20))
    for t in range(100):
        loss = m.forward_dec(ids, torch.zeros(2, 20, dtype=torch.long))[0].float().pow(2).mean() + ctl.orth_loss()
        opt.zero_grad(); loss.backward(); ctl.update(t); opt.step()
        if t >= 80:
            for mod, keep in zip(ctl.mods, ctl.final_mask):
                mod.lora_E.data.masked_fill_(~keep, 0.0)
    assert sum(ctl.ranks()) <= ctl.bT, (ctl.ranks(), ctl.bT)
    print(f"ok  AdaLoRA budget schedule {ctl.b0} -> {ctl.bT}, final ranks {ctl.ranks()}")

def test_decoding():
    s, _ = _build("LlamaLM")
    tok = FakeTok()
    P = [torch.randint(0, 290, (9,)), torch.randint(0, 290, (15,)), torch.randint(0, 290, (4,))]
    g = dict(GEN_DEFAULTS, max_new=20, min_new=5, no_repeat_ngram=3, batch_size=2)
    greedy, _ = generate(s, P, dict(g, method="greedy"), tok, "cpu")
    beam1, _ = generate(s, P, dict(g, method="beam", beams=1), tok, "cpu")
    assert greedy == beam1, (greedy, beam1)
    beam4, _ = generate(s, P, dict(g, method="beam", beams=4), tok, "cpu")
    samp, _ = generate(s, P, dict(g, method="sample"), tok, "cpu")
    for out in (greedy, beam4, samp):
        for o in out:
            assert len(o) <= 20 and not set(o) & set(tok.special_to_id.values())
            grams = [tuple(o[i:i + 3]) for i in range(len(o) - 2)]
            assert len(grams) == len(set(grams)), o
    lg = torch.zeros(1, V); lg[0, FakeTok.special_to_id["<|endoftext|>"]] = 10
    sp = {"eot": V - 5, "banned": torch.tensor([V - 4, V - 3, V - 2, V - 1])}
    assert process_logits(lg.clone(), torch.zeros(1, 2, dtype=torch.long), 2, dict(g), sp).argmax() != V - 5
    assert process_logits(lg.clone(), torch.zeros(1, 6, dtype=torch.long), 6, dict(g), sp).argmax() == V - 5
    sb, _ = _build("EncDecLM")
    Pb = [[torch.randint(0, 290, (20,)), torch.randint(0, 290, (7,))], [torch.randint(0, 290, (12,))]]
    gb, _ = generate(sb, Pb, dict(g, method="greedy"), tok, "cpu")
    bb, _ = generate(sb, Pb, dict(g, method="beam", beams=1), tok, "cpu")
    assert gb == bb
    generate(sb, Pb, dict(g, method="beam", beams=3), tok, "cpu")
    print("ok  decoding: beam(1) == greedy (A and B), no repeated 3-grams, specials never generated, min length")

def test_kd_loss():
    torch.manual_seed(0)
    s, t, lab = torch.randn(7, 11), torch.randn(7, 11), torch.randint(0, 11, (7,))
    l0, _ = seq_loss(s, lab)
    l1, p = seq_loss(s, lab, s.clone(), 0.5, 2.0)
    assert abs(p["kd"]) < 1e-5 and abs(float(l1) - 0.5 * float(l0)) < 1e-5
    l2, p2 = seq_loss(s, lab, t, 0.5, 2.0)
    assert p2["kd"] > 0
    print("ok  KD loss (zero for identical teacher, positive otherwise)")

def test_metrics():
    a = "The appeal is allowed and the conviction under Section 302 IPC is set aside."
    b = "Appeal allowed; conviction under section 302 of the IPC set aside by this Court."
    f = rouge_fallback(a, b)
    assert 0 < f["rouge1"][2] < 1 and rouge_fallback(a, a)["rougeL"][2] == 1.0
    try:
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False).score(b, a)
        assert abs(sc["rouge1"].fmeasure - f["rouge1"][2]) < 0.02 and abs(sc["rougeL"].fmeasure - f["rougeL"][2]) < 0.02
        r = rouge_scores([a], [b])
        assert r["rouge_impl"] == "rouge_score"
    except ImportError:
        print("    (rouge_score not installed: fallback only)")
    facts = legal_facts("Under Section 302 IPC and Article 21, the Indian Penal Code Act, 1860 applies; "
                        "decided on 12.03.1999 in Criminal Appeal No. 45 of 1998, fine of Rs. 5,000.")
    assert {"section 302", "article 21", "12 03 1999", "rs 5 000"} <= facts, facts
    assert redundancy("a b c a b c a b c")[0] > 0.5 and redundancy("one two three four")[0] == 0
    assert postprocess("This is a sentence. This is a sentence. Another one here.", {"dedup_sentences": True}) == \
        "This is a sentence. Another one here."
    print("ok  metrics (ROUGE fallback ~ rouge_score, legal facts, redundancy, sentence de-duplication)")

def test_submission():
    ids = ["doc_1", "doc_2", 3]
    d = tempfile.mkdtemp()
    zp = write_submission(ids, ["a summary", "b", "c"], d)
    import zipfile
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"]
        obj = json.loads(z.read("prediction.json"))
    assert obj[2] == {"id": 3, "summary": "c"}
    for bad in ([{"id": "doc_1", "summary": "x"}],
                [{"id": i, "summary": "x"} for i in ["doc_1", "doc_2", "doc_9"]],
                [{"id": i, "summary": ""} for i in ids],
                [{"id": i, "summary": "x", "extra": 1} for i in ids]):
        try:
            check_prediction(bad, ids); raise AssertionError(f"accepted {bad}")
        except AssertionError as e:
            if "accepted" in str(e):
                raise
    print("ok  submission (list of {id, summary}, exact ids, non-empty, zip root)")

def test_checkpoints(ckpt_dir):
    from p6summ.train import CKPT_FOLDERS
    found = 0
    for key, folder in CKPT_FOLDERS.items():
        p = os.path.join(ckpt_dir, folder, "model_final.pt")
        if not os.path.exists(p):
            continue
        found += 1
        m, cls, kw = load_phase1(p)
        m.eval()
        s = Summarizer(p).eval()
        s.bos_id = 31998
        torch.manual_seed(0)
        if cls == "EncDecLM":
            s.set_lengths(kw["dec_block"])
            src, dec = torch.randint(0, 31000, (1, 64)), torch.randint(0, 31000, (1, 16))
            with torch.no_grad():
                r, _ = m(src, torch.ones(1, 64, dtype=torch.bool), dec)
                g, _ = s.forward_encdec(src, torch.ones(1, 64, dtype=torch.bool), torch.zeros(1, dtype=torch.long),
                                        torch.zeros(1, dtype=torch.long), 1, 1, dec, torch.zeros(1, 16, dtype=torch.long))
            d = (r.reshape(-1, r.size(-1)) - g).abs().max().item()
        else:
            s.set_lengths(kw.get("block_size", 1024))
            ids = torch.randint(0, 31000, (1, 80))
            with torch.no_grad():
                r, _ = m(ids, torch.tensor([40])) if cls == "HybridLM" else m(ids)
                g, _ = s.forward_dec(ids, torch.zeros(1, 80, dtype=torch.long), torch.tensor([40]))
                d = (r.reshape(-1, r.size(-1)) - g).abs().max().item()
                st = s.prefill([ids[0, :40]])
                gl = [st["logits"][0]] + [s.step(st, ids[0, 40 + k:41 + k])[0] for k in range(5)]
                d = max(d, (torch.stack(gl) - r[0, 39:45]).abs().max().item()) if cls != "HybridLM" else d
        assert d < 2e-3, (folder, d)
        print(f"ok  checkpoint {folder} ({cls}, {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M): max |diff| {d:.2e}")
        del m, s
    if not found:
        print("    no Phase-1 checkpoints found under", ckpt_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="")
    a = ap.parse_args()
    test_sentences_and_context()
    test_loader()
    test_targets_and_collate()
    test_equivalence()
    test_peft()
    test_adalora_controller()
    test_decoding()
    test_kd_loss()
    test_metrics()
    test_submission()
    if a.ckpt_dir:
        test_checkpoints(a.ckpt_dir)
    print("ALL TESTS PASSED")
