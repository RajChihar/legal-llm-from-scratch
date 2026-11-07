import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .phase1_models import MODEL_CLASSES

ARCH_OF_CLASS = {"LlamaLM": "A", "EncDecLM": "B", "HybridLM": "C", "EfficientLM": "D"}

def _clean_state_dict(sd):
    out = {}
    for k, v in sd.items():
        for p in ("_orig_mod.", "module."):
            while k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out

def load_phase1(ckpt_path, random_init=False):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cls_name = ck.get("model_class")
    kw = dict(ck["model_kw"])
    if cls_name is None:
        cls_name = ("EncDecLM" if "enc_block" in kw else "HybridLM" if "n_segments" in kw
                    else "EfficientLM" if "window" in kw else "LlamaLM")
    model = MODEL_CLASSES[cls_name](**kw)
    if not random_init:
        missing, unexpected = model.load_state_dict(_clean_state_dict(ck["model"]), strict=False)
        missing = [m for m in missing if not m.endswith("lm_head.weight")]
        assert not missing and not unexpected, f"checkpoint mismatch: missing={missing} unexpected={unexpected}"
    return model, cls_name, kw

class ScaledRope(nn.Module):
    def __init__(self, head_dim, length, theta=10000.0, scale=1.0):
        super().__init__()
        inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        pos = torch.arange(length).float()
        freqs = torch.outer(pos / scale if scale != 1.0 else pos, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)
        self.length, self.scale = length, scale

    def forward(self, x, pos=0):
        T = x.size(-2)
        if isinstance(pos, int):
            assert pos + T <= self.length, f"position {pos + T} > RoPE table {self.length}"
            cos, sin = self.cos[pos:pos + T], self.sin[pos:pos + T]
        else:
            assert int(pos.max()) < self.length, f"position {int(pos.max())} >= RoPE table {self.length}"
            cos, sin = self.cos[pos][:, None], self.sin[pos][:, None]
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)

def project_qkv(attn, x):
    B, T, _ = x.shape
    H, hd = attn.n_head, attn.hd
    if hasattr(attn, "qkv"):
        q, k, v = attn.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        return q, k, v
    q = attn.wq(x).view(B, T, H, hd).transpose(1, 2)
    k, v = attn.wkv(x).view(B, T, 2, attn.n_kv, hd).permute(2, 0, 3, 1, 4)
    return q, k, v

def expand_kv(t, H):
    return t if t.size(1) == H else t.repeat_interleave(H // t.size(1), dim=1)

def prefix_kv(attn, h):
    P = h.size(0)
    H, hd = attn.n_head, attn.hd
    if hasattr(attn, "qkv"):
        _, k, v = attn.qkv(h).view(P, 3, H, hd).permute(1, 2, 0, 3)
    else:
        k, v = attn.wkv(h).view(P, 2, attn.n_kv, hd).permute(1, 2, 0, 3)
        k, v = expand_kv(k[None], H)[0], expand_kv(v[None], H)[0]
    return k, v

def self_attention(attn, x, rope, pos=0, mask=None, is_causal=False, past=None, prefix=None, window=None,
                   blocked=False):
    B, T, _ = x.shape
    q, k, v = project_qkv(attn, x)
    q, k = rope(q, pos), rope(k, pos)
    if past is not None:
        k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
    if window is not None and k.size(2) > window:
        k, v = k[:, :, -window:], v[:, :, -window:]
    if blocked:
        y = attn._blocked(q, k, v)
    else:
        kk, vv = expand_kv(k, attn.n_head), expand_kv(v, attn.n_head)
        if prefix is not None:
            pk, pv = prefix
            kk = torch.cat([pk[None].expand(B, -1, -1, -1).to(kk.dtype), kk], 2)
            vv = torch.cat([pv[None].expand(B, -1, -1, -1).to(vv.dtype), vv], 2)
        y = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask, is_causal=is_causal and mask is None,
                                           dropout_p=attn.dropout if attn.training else 0.0)
    return attn.resid_drop(attn.proj(y.transpose(1, 2).reshape(B, T, -1))), (k, v)

def memory_kv(attn, memory):
    B, S, _ = memory.shape
    k, v = attn.kv(memory).view(B, S, 2, attn.n_head, attn.hd).permute(2, 0, 3, 1, 4)
    return k, v

def cross_attention(attn, x, kv, mask):
    B, T, _ = x.shape
    q = attn.q(x).view(B, T, attn.n_head, attn.hd).transpose(1, 2)
    y = F.scaled_dot_product_attention(q, kv[0], kv[1], attn_mask=mask,
                                       dropout_p=attn.dropout if attn.training else 0.0)
    return attn.resid_drop(attn.proj(y.transpose(1, 2).reshape(B, T, -1)))

class Summarizer(nn.Module):
    def __init__(self, ckpt, keep_layers=None, grad_ckpt=False, random_init=False):
        super().__init__()
        m, cls_name, kw = load_phase1(ckpt, random_init)
        self.arch, self.cls_name, self.kw = ARCH_OF_CLASS[cls_name], cls_name, kw
        self.tok_emb, self.lm_head, self.drop = m.tok_emb, m.lm_head, m.drop
        self.theta = kw.get("rope_theta", 10000.0)
        self.grad_ckpt = grad_ckpt
        self.prefix_h = None
        self.soft_prompt = None
        self.bos_id = None
        if self.arch == "B":
            n_e, n_d = len(m.enc_blocks), len(m.dec_blocks)
            self.phase1_layers = n_e + n_d
            self.enc_blocks = nn.ModuleList(list(m.enc_blocks)[:keep_layers] if keep_layers else list(m.enc_blocks))
            self.dec_blocks = nn.ModuleList(list(m.dec_blocks)[:keep_layers] if keep_layers else list(m.dec_blocks))
            self.enc_norm, self.dec_norm = m.enc_norm, m.dec_norm
            self.hd = kw["d_model"] // kw["n_head"]
            self.enc_len, self.dec_pretrained = kw["enc_block"], kw["dec_block"]
        else:
            self.phase1_layers = len(m.blocks)
            self.blocks = nn.ModuleList(list(m.blocks)[:keep_layers] if keep_layers else list(m.blocks))
            self.norm_f = m.norm_f
            self.hd = kw["d_model"] // kw["n_head"]
            self.seg_emb = m.seg_emb if self.arch == "C" else None
            self.window = m.window if self.arch == "D" else None
            self.pretrained_len = m.max_pos if self.arch == "D" else kw["block_size"]
        self.rope = self.enc_rope = self.dec_rope = None
        del m

    def peft_blocks(self):
        return list(self.enc_blocks) + list(self.dec_blocks) if self.arch == "B" else list(self.blocks)

    def last_blocks(self, k):
        return list(self.dec_blocks)[-k:] if self.arch == "B" else list(self.blocks)[-k:]

    def final_norm(self):
        return self.dec_norm if self.arch == "B" else self.norm_f

    def n_layers(self):
        return len(self.peft_blocks())

    def set_lengths(self, total_len, rope_scale="auto", table_len=None):
        n_p = 0 if self.soft_prompt is None else self.soft_prompt.size(0)
        need = total_len + n_p
        table = max(need, (table_len or 0) + n_p)
        dev = self.tok_emb.weight.device
        base = self.dec_pretrained if self.arch == "B" else self.pretrained_len
        s = max(1.0, need / base) if rope_scale == "auto" else float(rope_scale)
        assert s >= 1.0, "rope_scale must be >= 1"
        if self.arch == "B":
            self.enc_rope = ScaledRope(self.hd, self.enc_len, self.theta).to(dev)
            self.dec_rope = ScaledRope(self.hd, max(table, base), self.theta, s).to(dev)
            return {"dec_positions": max(table, base), "dec_rope_scale": round(s, 4)}
        self.rope = ScaledRope(self.hd, max(table, base), self.theta, s).to(dev)
        return {"positions": max(table, base), "rope_scale": round(s, 4)}

    def logits(self, h):
        return self.lm_head(h)

    def _prefix(self, li, dtype):
        if self.prefix_h is None:
            return None
        blk = self.blocks[li]
        return prefix_kv(blk.attn, blk.norm1(self.prefix_h[li].to(dtype)))

    def _embed(self, ids, seg=None):
        x = self.tok_emb(ids)
        if self.seg_emb is not None:
            x = x + (self.seg_emb(seg) if torch.is_tensor(seg) else self.seg_emb.weight[seg])
        return x

    def _train_mask(self, prefix_len, T, n_pre, device):
        if self.arch == "D":
            assert n_pre == 0, "prefix tuning is not implemented for D"
            return None, True, True
        if self.arch == "C" and prefix_len is not None and not bool((prefix_len == 0).all()):
            pos = torch.arange(T, device=device)
            m = (pos[None, None, :] < prefix_len[:, None, None]) | (pos[None, :, None] >= pos[None, None, :])
            m = m[:, None]
        elif n_pre:
            m = torch.ones(T, T, dtype=torch.bool, device=device).tril()[None, None]
        else:
            return None, True, False
        if n_pre:
            m = torch.cat([torch.ones(*m.shape[:-1], n_pre, dtype=torch.bool, device=device), m], -1)
        return m, False, False

    def _dec_block(self, li, x, mask, causal, blocked):
        blk = self.blocks[li]
        a, _ = self_attention(blk.attn, blk.norm1(x), self.rope, 0, mask, causal,
                              prefix=self._prefix(li, x.dtype), blocked=blocked)
        x = x + a
        return x + blk.ffn(blk.norm2(x))

    def forward_dec(self, ids, labels, prefix_len=None):
        B, T = ids.shape
        n_p = 0
        if self.arch == "C":
            pl = prefix_len if prefix_len is not None else torch.zeros(B, dtype=torch.long, device=ids.device)
            seg = (torch.arange(T, device=ids.device)[None, :] >= pl[:, None]).long()
            x = self._embed(ids, seg)
        else:
            x = self._embed(ids)
        if self.soft_prompt is not None:
            n_p = self.soft_prompt.size(0)
            sp = self.soft_prompt[None].expand(B, -1, -1).to(x.dtype)
            if self.seg_emb is not None:
                sp = sp + self.seg_emb.weight[0 if prefix_len is not None and bool((prefix_len > 0).all()) else 1]
            x = torch.cat([sp, x], 1)
            labels = torch.cat([torch.full((B, n_p), -100, dtype=labels.dtype, device=labels.device), labels], 1)
            if prefix_len is not None:
                prefix_len = torch.where(prefix_len > 0, prefix_len + n_p, prefix_len)
            T = T + n_p
        x = self.drop(x)
        n_pre = 0 if self.prefix_h is None else self.prefix_h[0].size(0)
        mask, causal, blocked = self._train_mask(prefix_len if self.arch == "C" else None, T, n_pre, ids.device)
        for li in range(len(self.blocks)):
            if self.grad_ckpt and self.training:
                x = checkpoint(self._dec_block, li, x, mask, causal, blocked, use_reentrant=False)
            else:
                x = self._dec_block(li, x, mask, causal, blocked)
        sel = labels != -100
        return self.logits(self.norm_f(x[sel])), labels[sel]

    def encode(self, ids, keep):
        x = self.drop(self.tok_emb(ids))
        mask = None if bool(keep.all()) else keep[:, None, None, :]
        for blk in self.enc_blocks:
            if self.grad_ckpt and self.training:
                x = checkpoint(self._enc_block, blk, x, mask, use_reentrant=False)
            else:
                x = self._enc_block(blk, x, mask)
        return self.enc_norm(x)

    def _enc_block(self, blk, x, mask):
        x = x + self_attention(blk.attn, blk.norm1(x), self.enc_rope, 0, mask)[0]
        return x + blk.ffn(blk.norm2(x))

    @staticmethod
    def fid_memory(enc, keep, doc, slot, n_docs, k_max):
        N, S, C = enc.shape
        mem = enc.new_zeros(n_docs, k_max, S, C)
        mm = torch.zeros(n_docs, k_max, S, dtype=torch.bool, device=enc.device)
        mem[doc, slot] = enc
        mm[doc, slot] = keep
        return mem.view(n_docs, k_max * S, C), mm.view(n_docs, k_max * S)

    def _dec_block_b(self, blk, x, memory, mm4):
        x = x + self_attention(blk.self_attn, blk.norm1(x), self.dec_rope, 0, None, True)[0]
        x = x + cross_attention(blk.cross_attn, blk.norm2(x), memory_kv(blk.cross_attn, memory), mm4)
        return x + blk.ffn(blk.norm3(x))

    def forward_encdec(self, src, keep, doc, slot, n_docs, k_max, dec_in, labels):
        memory, mm = self.fid_memory(self.encode(src, keep), keep, doc, slot, n_docs, k_max)
        mm4 = None if bool(mm.all()) else mm[:, None, None, :]
        x = self.drop(self.tok_emb(dec_in))
        for blk in self.dec_blocks:
            if self.grad_ckpt and self.training:
                x = checkpoint(self._dec_block_b, blk, x, memory, mm4, use_reentrant=False)
            else:
                x = self._dec_block_b(blk, x, memory, mm4)
        sel = labels != -100
        return self.logits(self.dec_norm(x[sel])), labels[sel]

    def forward(self, batch):
        if self.arch == "B":
            return self.forward_encdec(batch["src"], batch["keep"], batch["doc"], batch["slot"], batch["n_docs"],
                                       batch["k_max"], batch["dec_in"], batch["labels"])
        return self.forward_dec(batch["ids"], batch["labels"], batch.get("prefix_len"))

    @torch.no_grad()
    def prefill(self, inputs):
        if self.arch == "B":
            return self._prefill_b(inputs)
        dev = self.tok_emb.weight.device
        L = len(self.blocks)
        ks, vs, lens, last = [[] for _ in range(L)], [[] for _ in range(L)], [], []
        for p in inputs:
            ids = p.to(dev)[None]
            x = self._embed(ids, 0)
            if self.soft_prompt is not None:
                sp = self.soft_prompt[None].to(x.dtype)
                if self.seg_emb is not None:
                    sp = sp + self.seg_emb.weight[0]
                x = torch.cat([sp, x], 1)
            T = x.size(1)
            for li, blk in enumerate(self.blocks):
                pre = self._prefix(li, x.dtype)
                if self.arch == "A":
                    mask = None if pre is None else torch.cat(
                        [torch.ones(T, pre[0].size(1), dtype=torch.bool, device=dev),
                         torch.ones(T, T, dtype=torch.bool, device=dev).tril()], 1)[None, None]
                    a, (k, v) = self_attention(blk.attn, blk.norm1(x), self.rope, 0, mask, pre is None, prefix=pre)
                elif self.arch == "C":
                    a, (k, v) = self_attention(blk.attn, blk.norm1(x), self.rope, 0, None, False, prefix=pre)
                else:
                    a, (k, v) = self_attention(blk.attn, blk.norm1(x), self.rope, 0, blocked=True)
                    k, v = k[:, :, -self.window:], v[:, :, -self.window:]
                x = x + a
                x = x + blk.ffn(blk.norm2(x))
                ks[li].append(k[0]); vs[li].append(v[0])
            lens.append(T)
            last.append(self.logits(self.norm_f(x[:, -1]))[0])
        Tk = max(k.size(1) for k in ks[0])
        n = len(inputs)
        mask = torch.zeros(n, Tk, dtype=torch.bool, device=dev)
        state = {"k": [], "v": [], "pos": torch.tensor(lens, dtype=torch.long, device=dev)}
        for li in range(L):
            K = ks[li][0].new_zeros(n, ks[li][0].size(0), Tk, self.hd)
            V = torch.zeros_like(K)
            for b in range(n):
                t = ks[li][b].size(1)
                K[b, :, Tk - t:], V[b, :, Tk - t:] = ks[li][b], vs[li][b]
                if li == 0:
                    mask[b, Tk - t:] = True
            state["k"].append(K); state["v"].append(V)
        state["mask"] = mask
        state["logits"] = torch.stack(last).float()
        return state

    @torch.no_grad()
    def step(self, state, tok):
        if self.arch == "B":
            return self._step_b(state, tok)
        B = tok.size(0)
        x = self._embed(tok[:, None], 1)
        pos = state["pos"][:, None]
        mask = torch.cat([state["mask"], torch.ones(B, 1, dtype=torch.bool, device=tok.device)], 1)
        if self.window is not None:
            mask = mask[:, -self.window:]
        m4 = mask[:, None, None, :]
        for li, blk in enumerate(self.blocks):
            pre = self._prefix(li, x.dtype)
            mm = m4 if pre is None else torch.cat(
                [torch.ones(B, 1, 1, pre[0].size(1), dtype=torch.bool, device=tok.device), m4], -1)
            a, (k, v) = self_attention(blk.attn, blk.norm1(x), self.rope, pos, mm,
                                       past=(state["k"][li], state["v"][li]), prefix=pre, window=self.window)
            state["k"][li], state["v"][li] = k, v
            x = x + a
            x = x + blk.ffn(blk.norm2(x))
        state["mask"] = mask
        state["pos"] = state["pos"] + 1
        return self.logits(self.norm_f(x[:, 0])).float()

    def _prefill_b(self, inputs):
        dev = self.tok_emb.weight.device
        n = len(inputs)
        chunks = [(b, k, c) for b, doc in enumerate(inputs) for k, c in enumerate(doc)]
        S = max(len(c) for _, _, c in chunks)
        k_max = max(len(doc) for doc in inputs)
        ids = torch.full((len(chunks), S), self.kw.get("pad_id") or 0, dtype=torch.long, device=dev)
        keep = torch.zeros(len(chunks), S, dtype=torch.bool, device=dev)
        doc = torch.tensor([b for b, _, _ in chunks], device=dev)
        slot = torch.tensor([k for _, k, _ in chunks], device=dev)
        for r, (_, _, c) in enumerate(chunks):
            ids[r, :len(c)] = c.to(dev)
            keep[r, :len(c)] = True
        memory, mm = self.fid_memory(self.encode(ids, keep), keep, doc, slot, n, k_max)
        state = {"self_k": [None] * len(self.dec_blocks), "self_v": [None] * len(self.dec_blocks),
                 "cross_k": [], "cross_v": [], "mem_mask": mm, "pos": torch.zeros(n, dtype=torch.long, device=dev)}
        for blk in self.dec_blocks:
            k, v = memory_kv(blk.cross_attn, memory)
            state["cross_k"].append(k); state["cross_v"].append(v)
        bos = torch.full((n,), self.bos_id, dtype=torch.long, device=dev)
        state["logits"] = self._step_b(state, bos)
        return state

    def _step_b(self, state, tok):
        x = self.tok_emb(tok[:, None])
        pos = state["pos"][:, None]
        mm4 = state["mem_mask"][:, None, None, :]
        for li, blk in enumerate(self.dec_blocks):
            past = None if state["self_k"][li] is None else (state["self_k"][li], state["self_v"][li])
            a, (k, v) = self_attention(blk.self_attn, blk.norm1(x), self.dec_rope, pos, None, past=past)
            state["self_k"][li], state["self_v"][li] = k, v
            x = x + a
            x = x + cross_attention(blk.cross_attn, blk.norm2(x), (state["cross_k"][li], state["cross_v"][li]), mm4)
            x = x + blk.ffn(blk.norm3(x))
        state["pos"] = state["pos"] + 1
        return self.logits(self.dec_norm(x[:, 0])).float()

def map_state(state, fn):
    out = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            out[k] = fn(v)
        elif isinstance(v, list):
            out[k] = [None if t is None else fn(t) for t in v]
        else:
            out[k] = v
    return out

def seq_loss(logits, labels, teacher_logits=None, kd_alpha=0.5, kd_T=2.0):
    ce = F.cross_entropy(logits.float(), labels)
    if teacher_logits is None:
        return ce, {"ce": float(ce.detach())}
    T = kd_T
    kl = F.kl_div(F.log_softmax(logits.float() / T, -1), F.log_softmax(teacher_logits.float() / T, -1),
                  reduction="batchmean", log_target=True) * T * T
    return (1 - kd_alpha) * ce + kd_alpha * kl, {"ce": float(ce.detach()), "kd": float(kl.detach())}

def trainable_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
