"""Byte-level BPE tokenizer from Phase 1 (load-only copy) + character offsets for NER.

Phase 1 trained the merges; here we only *load* them (same ids as pretraining) and add
`encode_with_offsets`, which returns for every token the character span it covers in the original text.

Offsets are computed per pre-token: the regex gives each pre-token's char span, BPE splits the pre-token's
UTF-8 bytes into tokens, and a byte->char table maps every token's byte range back to characters.
"""
import json
from dataclasses import dataclass

import regex as re_u

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|bos|>", "<|sep|>", "<|mask|>"]

# Phase-1 pattern (kept identical so token ids match pretraining). The punctuation branch
# ` ?[^\s\p{L}\p{N}]+[\r\n]*` glues punctuation runs such as ".," or ")." into one pre-token.
# FINE_PUNCT_PATTERN splits every punctuation character into its own pre-token, so an entity that ends
# between "." and "," (e.g. "M/s. ABC Ltd.,") becomes reachable. Same vocabulary (bytes are always valid).
FINE_PUNCT_PATTERN = (r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}{1,3}| ?[^\s\p{L}\p{N}]|\s*[\r\n]+|\s+(?!\S)|\s+""")


def _merge_ids(ids, pair, new_id):
    out, i, n = [], 0, len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out


@dataclass
class Encoded:
    ids: list        # token ids
    starts: list     # char start of each token (inclusive)
    ends: list       # char end of each token (exclusive)


class BPETokenizer:
    def __init__(self, merges, pattern, specials=SPECIAL_TOKENS, fine_punct=False):
        self.base_pattern = pattern
        self.fine_punct = fine_punct
        self.pattern = FINE_PUNCT_PATTERN if fine_punct else pattern
        self.pat = re_u.compile(self.pattern)
        self.merges = [tuple(m) for m in merges]
        self.specials = list(specials)
        self.ranks = {p: i for i, p in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        base = 256 + len(self.merges)
        self.n_regular = base
        self.special_to_id = {s: base + i for i, s in enumerate(self.specials)}
        self.id_to_special = {v: k for k, v in self.special_to_id.items()}
        self.vocab_size = base + len(self.specials)
        self.cache = {}

    @classmethod
    def load(cls, path, fine_punct=False):
        d = json.load(open(path, encoding="utf-8"))
        return cls(d["merges"], d["pattern"], d.get("specials", SPECIAL_TOKENS), fine_punct=fine_punct)

    @property
    def pad_id(self): return self.special_to_id["<|pad|>"]
    @property
    def bos_id(self): return self.special_to_id["<|bos|>"]
    @property
    def mask_id(self): return self.special_to_id["<|mask|>"]

    def _encode_word(self, s):
        hit = self.cache.get(s)
        if hit is not None:
            return hit
        ids = list(s.encode("utf-8"))
        ranks = self.ranks
        while len(ids) >= 2:
            best, best_r = None, None
            for p in zip(ids, ids[1:]):
                r = ranks.get(p)
                if r is not None and (best_r is None or r < best_r):
                    best, best_r = p, r
            if best is None:
                break
            ids = _merge_ids(ids, best, 256 + best_r)
        if len(self.cache) > 500_000:
            self.cache.clear()
        self.cache[s] = ids
        return ids

    def encode(self, text):
        out = []
        for w in self.pat.findall(text):
            out.extend(self._encode_word(w))
        return out

    def encode_with_offsets(self, text):
        ids, starts, ends = [], [], []
        for m in self.pat.finditer(text):
            w, c0 = m.group(), m.start()
            wid = self._encode_word(w)
            if len(wid) == 1:                                   # common fast path: whole pre-token = 1 token
                ids.append(wid[0]); starts.append(c0); ends.append(m.end())
                continue
            # byte index -> char index inside this pre-token
            b2c = []
            for ci, ch in enumerate(w):
                b2c.extend([ci] * len(ch.encode("utf-8")))
            b = 0
            for t in wid:
                nb = len(self.vocab[t])
                ids.append(t)
                starts.append(c0 + b2c[b])
                ends.append(c0 + b2c[b + nb - 1] + 1)
                b += nb
        return Encoded(ids, starts, ends)

    def decode(self, ids):
        return b"".join(self.vocab[int(i)] for i in ids if int(i) < self.n_regular).decode("utf-8", errors="replace")
