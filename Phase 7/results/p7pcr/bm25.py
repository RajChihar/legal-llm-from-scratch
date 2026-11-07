"""Okapi BM25 over the candidate pool of a split (own sparse implementation, no external index).

Not a learned model: used as (1) the lexical baseline of the analysis, (2) the hard-negative miner for contrastive
training (top-ranked non-relevant candidates), (3) an optional lexical signal fused with the dense scores.

    w(d, t) = idf(t) * tf(d,t) (k1 + 1) / (tf(d,t) + k1 (1 - b + b |d| / avgdl)),   idf(t) = ln(1 + (N - df + .5) / (df + .5))
    score(q, d) = sum_t qw(q, t) w(d, t),   qw = 1 (binary), tf(q,t) (count) or tf (k3 + 1) / (tf + k3) (sat)
Terms: lower-cased words (letters / digits), stop words removed, optionally + word bigrams.
"""
import re
from collections import Counter

import numpy as np
import scipy.sparse as sp

_WORD = re.compile(r"[a-z0-9]+")
STOP = set("""a an the of to in and or for on at by with from as is are was were be been being this that these those it
its which who whom whose what when where why how not no nor but if then than so such there their them they he she his
her him we our us you your i me my shall will would should could can may might must do does did done has have had having
into upon under over about above below up down out off again further once all any both each few more most other some
same own only very s t also said per vs v""".split())


def terms(text, ngrams=(1,)):
    w = [x for x in _WORD.findall(text.lower()) if x not in STOP and (len(x) > 1 or x.isdigit())]
    out = list(w) if 1 in ngrams else []
    if 2 in ngrams:
        out += [a + "_" + b for a, b in zip(w, w[1:])]
    return out


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75, ngrams=(1,), min_df=1):
        self.k1, self.b, self.ngrams = k1, b, tuple(ngrams)
        bags = [Counter(terms(d, self.ngrams)) for d in docs]
        df = Counter(t for bag in bags for t in bag)
        self.vocab = {t: i for i, t in enumerate(sorted(t for t, c in df.items() if c >= min_df))}
        N = len(docs)
        dfv = np.zeros(len(self.vocab))
        for t, i in self.vocab.items():
            dfv[i] = df[t]
        self.idf = np.log(1 + (N - dfv + 0.5) / (dfv + 0.5))
        dl = np.array([sum(b.values()) for b in bags], dtype=np.float64)
        avg = dl.mean() if N else 1.0
        rows, cols, vals = [], [], []
        for r, bag in enumerate(bags):
            norm = k1 * (1 - b + b * dl[r] / max(avg, 1e-9))
            for t, tf in bag.items():
                i = self.vocab.get(t)
                if i is not None:
                    rows.append(r); cols.append(i); vals.append(self.idf[i] * tf * (k1 + 1) / (tf + norm))
        self.W = sp.csr_matrix((vals, (rows, cols)), shape=(N, len(self.vocab)), dtype=np.float32)

    def query_matrix(self, queries, query_tf="sat", k3=1.5):
        rows, cols, vals = [], [], []
        for r, q in enumerate(queries):
            for t, tf in Counter(terms(q, self.ngrams)).items():
                i = self.vocab.get(t)
                if i is None:
                    continue
                v = 1.0 if query_tf == "binary" else float(tf) if query_tf == "count" else tf * (k3 + 1) / (tf + k3)
                rows.append(r); cols.append(i); vals.append(v)
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(queries), len(self.vocab)), dtype=np.float32)

    def scores(self, queries, query_tf="sat"):
        """-> dense [n_queries, n_docs] float32."""
        return np.asarray((self.query_matrix(queries, query_tf) @ self.W.T).todense(), dtype=np.float32)


def bm25_scores(split, k1=1.2, b=0.75, ngrams=(1,), query_tf="sat"):
    """BM25 scores of a cached split's queries against its own pool."""
    return BM25(split["c_text"], k1, b, ngrams).scores(split["q_text"], query_tf)
