import argparse
import itertools
import json
import random
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2lner.backbones import load_backbone, load_phase1
from p2lner.heads import CRF
from p2lner.model import NERModel
from p2lner.peft import apply_method
from p2lner.spans import (TagSet, bio_constraints, decode_doc, spans_to_tags, tags_to_spans,
                          token_views, viterbi)
from p2lner.tokenizer import BPETokenizer

LABELS = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT", "AUTH", "COURT", "STAT", "PREC", "DATE", "CASENO"]
NAMES = ["Ram Kumar", "M/s. Bharat Steel Ltd.", "State of U.P.", "Smt. Kamla Devi", "Union of India", "Rajesh Sharma"]

def synth_doc(rng):
    parts, gold = [], []
    def add(t, lab=None):
        s = sum(len(p) for p in parts)
        parts.append(t)
        if lab:
            gold.append((s, s + len(t), lab))
    add("REPORTABLE\nIN THE "); add("SUPREME COURT OF INDIA", "COURT"); add("\nCIVIL APPEAL NO. ")
    add(f"{rng.randint(100, 9999)} OF {rng.randint(1990, 2020)}", "CASENO"); add("\n\n")
    add(rng.choice(NAMES), "APP"); add(" ...Appellant(s)\nVersus\n"); add(rng.choice(NAMES), "RESP"); add(" ...Respondent(s)\n")
    add("Heard Mr. "); add("K. K. Venugopal", "A.COUNSEL"); add(", learned counsel for the appellant and Ms. ")
    add("Indira Jaising", "R.COUNSEL"); add(", for the respondent. The order dated ")
    add(f"{rng.randint(1, 28)}.0{rng.randint(1, 9)}.20{rng.randint(10, 20)}", "DATE"); add(" passed by the ")
    add("High Court of Delhi", "COURT"); add(" under "); add("Section 302 of the Indian Penal Code", "STAT")
    add(" is challenged. Reliance was placed on "); add("Maneka Gandhi v. Union of India", "PREC")
    add(" (1978) 1 SCC 248. PW-1 "); add("Suresh", "WIT"); add(" deposed before the "); add("Collector", "AUTH")
    add(". Café “quoted” text — résumé. ")
    add("J. Chelameswar", "JUDGE"); add(", J.\n")
    return "".join(parts), gold

def find_ckpts(root):
    out = {}
    for p in sorted(Path(root).glob("*/model_final.pt")) or sorted(Path(root).rglob("model_final.pt")):
        out[p.parent.name] = p
    return out

def test_tokenizer(tok_path):
    tok = BPETokenizer.load(tok_path, fine_punct=False)
    rng = random.Random(0)
    for _ in range(20):
        text, _ = synth_doc(rng)
        enc = tok.encode_with_offsets(text)
        assert enc.ids == tok.encode(text), "offset encoder must give the Phase-1 ids"
        assert tok.decode(enc.ids) == text
        assert all(0 <= s < e <= len(text) for s, e in zip(enc.starts, enc.ends))
        assert enc.starts[0] == 0 and enc.ends[-1] == len(text)
        for i, s, e in zip(enc.ids, enc.starts, enc.ends):
            piece = tok.vocab[i]
            if len(text[s:e].encode()) == len(piece):
                assert text[s:e].encode() == piece
    print("[ok] tokenizer offsets and ids")

def test_roundtrip(tok_path):
    tagset = TagSet(LABELS)
    rng = random.Random(1)
    for fine in (False, True):
        tok = BPETokenizer.load(tok_path, fine_punct=fine)
        miss = tot = 0
        for _ in range(30):
            text, gold = synth_doc(rng)
            enc = tok.encode_with_offsets(text)
            tags, skipped = spans_to_tags(text, enc, gold, tagset)
            cs, ce, ws = token_views(text, enc)
            back = set(tags_to_spans(cs, ce, ws, tags, tagset))
            miss += len(set(gold) - back); tot += len(gold)
            sc = np.full((len(tags), tagset.n), -10.0, dtype=np.float32); sc[np.arange(len(tags)), tags] = 0
            assert set(decode_doc(text, enc, sc, tagset)) == back
        print(f"[ok] BIO round trip fine_punct={fine}: {tot - miss}/{tot} gold spans recovered")
        if fine:
            assert miss == 0, "fine_punct tokenisation must make every synthetic span reachable"

def test_backbones(ckpts):
    torch.manual_seed(0)
    for name, path in ckpts.items():
        model, cls, kw = load_phase1(path)
        model.eval()
        ids = torch.randint(0, 31000, (2, 300))
        keep = torch.ones(2, 300, dtype=torch.bool)
        with torch.no_grad():
            if cls == "EncDecLM":
                ref = model.encode(ids)
                bb, _ = load_backbone(path, "bidirectional"); bb.eval()
                out = bb(ids, keep)
            elif cls == "HybridLM":
                ref = model.norm_f(_hidden_c(model, ids))
                bb, _ = load_backbone(path, "bidirectional"); bb.eval()
                out = bb(ids, keep)
                refc, _ = model(ids)
                bbc, _ = load_backbone(path, "causal", c_segment=1); bbc.eval()
                assert torch.allclose(model.lm_head(bbc(ids, keep)), refc, atol=1e-3), "C causal mismatch"
            else:
                ref, _ = model(ids)
                bb, _ = load_backbone(path, "causal"); bb.eval()
                out = model.lm_head(bb(ids, keep))
            err = (out - ref).abs().max().item()
            assert err < 1e-3, f"{name}: wrapper differs from original forward ({err})"
            bbb, _ = load_backbone(path, "bidirectional"); bbb.eval()
            keep2 = keep.clone(); keep2[1, 200:] = False
            ids2 = ids.clone(); ids2[1, 200:] = 0
            full = bbb(ids2, keep2)[1, :200]
            alone = bbb(ids[1:, :200], keep[1:, :200])[0]
            perr = (full - alone).abs().max().item()
            assert perr < 1e-3, f"{name}: padding leaks into real tokens ({perr})"
        print(f"[ok] backbone {name} ({cls}): forward matches Phase-1 model (max err {err:.1e}), padding-invariant ({perr:.1e})")

def _hidden_c(model, ids):
    T = ids.size(1)
    x = model.tok_emb(ids) + model.seg_emb(torch.zeros_like(ids))
    for blk in model.blocks:
        x = blk(x, model.rope, None, False)
    return x

def test_peft(path):
    tagset = TagSet(LABELS)
    ids = torch.randint(0, 31000, (2, 64)); keep = torch.ones(2, 64, dtype=torch.bool)
    bb0, _ = load_backbone(path, "bidirectional", keep_layers=3); bb0.eval()
    with torch.no_grad():
        ref = bb0(ids, keep)
    for m in [{"name": "lora"}, {"name": "lorma", "variant": "plus"}, {"name": "lorma", "variant": "pi", "targets": ["attn"]},
              {"name": "adapter"}, {"name": "bitfit"}, {"name": "partial", "train_last_k": 1}, {"name": "frozen"}]:
        bb, _ = load_backbone(path, "bidirectional", keep_layers=3)
        apply_method(bb, m); bb.eval()
        with torch.no_grad():
            err = (bb(ids, keep) - ref).abs().max().item()
        ntr = sum(p.numel() for p in bb.parameters() if p.requires_grad)
        assert err < 1e-3, f"{m} is not identity at init ({err})"
        model = NERModel(bb, tagset); model.train()
        loss = model.loss(model.emissions(ids, keep), torch.randint(0, tagset.n, (2, 64)), keep, {})
        loss.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
        print(f"[ok] {m}: identity at init (err {err:.1e}), trainable backbone params {ntr:,}")

def test_crf():
    tagset = TagSet(["X", "Y"])
    crf = CRF(tagset)
    with torch.no_grad():
        crf.trans.normal_(); crf.start.normal_(); crf.end.normal_()
    emis = torch.randn(1, 4, tagset.n)
    tr, st, en = (x.detach().numpy() for x in crf.full_trans())
    paths = list(itertools.product(range(tagset.n), repeat=4))
    def score(p):
        return st[p[0]] + sum(emis[0, t, p[t]].item() for t in range(4)) + sum(tr[p[t - 1], p[t]] for t in range(1, 4)) + en[p[-1]]
    scores = np.array([score(p) for p in paths])
    best = paths[int(scores.argmax())]
    v = viterbi(emis[0].numpy(), tr, st, en)
    assert tuple(v) == best, (v, best)
    st = crf.start.detach().numpy()
    scores = np.array([score(p) for p in paths])
    logZ = np.log(np.exp(scores - scores.max()).sum()) + scores.max()
    gold = torch.tensor([best])
    nll = crf.nll(emis, gold, torch.ones(1, 4, dtype=torch.bool)).item() * 4
    assert abs(nll - (logZ - score(best))) < 1e-3, (nll, logZ - score(best))
    assert crf.nll(emis, torch.tensor([[2, 2, 0, 0]]), torch.ones(1, 4, dtype=torch.bool)).item() < 50
    print("[ok] CRF forward algorithm and Viterbi match brute force")

def test_end_to_end(path, tok_path):
    from p2lner.train import run
    rng = random.Random(2)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "Phase_2_LNER"
        (root / "Train").mkdir(parents=True); (root / "Test").mkdir()
        for k in (1, 2):
            texts, gold = {}, {}
            for i in range(6):
                t, g = synth_doc(rng)
                texts[f"{k}{i:03d}"] = t * 3
                n = len(t)
                gold[f"{k}{i:03d}"] = [[s + j * n, e + j * n, l, t[s:e]] for j in range(3) for s, e, l in g]
            json.dump(texts, open(root / "Train" / f"fold{k}-text.json", "w"))
            json.dump(gold, open(root / "Train" / f"fold{k}-gold.json", "w"))
        json.dump({f"9{i}": synth_doc(rng)[0] for i in range(3)}, open(root / "Test" / "fold3-text.json", "w"))
        open(root / "Train" / "labels.txt", "w").write("\n".join(LABELS))
        cfg = {"run_name": "e2e", "out_dir": str(Path(td) / "runs"), "data_dir": str(root),
               "backbone": {"ckpt": str(path), "keep_layers": 2}, "tokenizer": {"path": str(tok_path)},
               "window": {"max_len": 128, "stride": 96}, "method": {"name": "lorma", "r": 4},
               "train": {"epochs": 2, "batch_size": 4, "lr_new": 3e-3}, "split": {"dev_frac": 0.2}}
        summary, _ = run(cfg, device="cpu")
        with zipfile.ZipFile(summary["submission"]) as z:
            assert z.namelist() == ["prediction.json"]
            pred = json.loads(z.read("prediction.json"))
        assert set(pred) == {"90", "91", "92"}
        from p2lner.run import repredict
        repredict(Path(td) / "runs" / "e2e", str(root), device="cpu")
        print(f"[ok] end-to-end run: dev macro-F1 {summary['dev']['macro_f1']:.3f} after 2 tiny epochs, submission valid")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--skip_e2e", action="store_true")
    a = ap.parse_args()
    ck = find_ckpts(a.ckpt_dir)
    print("checkpoints:", {k: str(v) for k, v in ck.items()})
    first = next(iter(ck.values()))
    tok_path = first.parent / "tokenizer.json"
    test_tokenizer(tok_path)
    test_roundtrip(tok_path)
    test_crf()
    test_backbones(ck)
    test_peft(first)
    if not a.skip_e2e:
        test_end_to_end(first, tok_path)
    print("\nALL CHECKS PASSED")
