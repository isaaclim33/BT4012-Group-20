"""Hybrid retrieval over the typology playbook (lecture 06).

Two rankers are fused with reciprocal rank fusion, RRF(d) = sum 1 / (60 + rank):
* lexical: BM25 on word tokens, good for exact terms such as 'duplicate';
* dense: latent semantic analysis (TF-IDF reduced by truncated SVD). An offline
  stand-in for a neural embedding model, chosen because the workspace cannot
  download embedding weights; it still matches paraphrases through shared
  co-occurrence structure. Swapping in a sentence-embedding model only changes
  `_dense`.

Retrieval is evaluated separately from generation (recall@k, MRR) on a labelled
query set written before looking at results.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

CORPUS = Path(__file__).with_name("corpus") / "typology_playbook.json"
TOKEN = re.compile(r"[a-z0-9]+")

# Gold queries: phrased with different vocabulary from the chunk text on purpose.
GOLD_QUERIES = [
    ("claims for appointments that never happened", "T1"),
    ("provider bills for care it did not deliver", "T1"),
    ("same patient same day same amount billed twice", "T2"),
    ("repeat submission of an identical bill", "T2"),
    ("diagnosis codes chosen to justify a procedure", "T3"),
    ("claim describes different treatment than delivered", "T3"),
    ("charging a pricier diagnosis group than warranted", "T4"),
    ("payments per admission far above comparable hospitals", "T4"),
    ("service outside the plan coded so it gets paid", "T5"),
    ("uncovered treatment disguised as reimbursable", "T5"),
    ("unusually large number of claims and total payments", "T6"),
    ("provider with very high billing volume", "T6"),
    ("hospital stays kept longer than necessary", "T7"),
    ("admissions lasting many more days than peers", "T7"),
    ("one doctor appears on most of the claims", "T8"),
    ("single attending physician dominates the provider's bills", "T8"),
    ("providers passing the same patients between them", "T9"),
    ("ring of clinics sharing beneficiaries and referrals", "T9"),
    ("healthy patients but intensive expensive billing", "T10"),
    ("sickness of patients does not match what is charged", "T10"),
]


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class HybridRetriever:
    def __init__(self, path: Path = CORPUS, svd_dims: int = 8, k_rrf: int = 60, seed: int = 42):
        self.doc = json.loads(path.read_text())
        self.chunks = self.doc["chunks"]
        self.ids = [c["id"] for c in self.chunks]
        texts = [f"{c['title']}. {c['text']}" for c in self.chunks]
        # BM25 statistics
        self.toks = [tokens(t) for t in texts]
        self.avgdl = np.mean([len(t) for t in self.toks])
        vocab = {w for t in self.toks for w in t}
        n = len(self.toks)
        self.idf = {w: np.log(1 + (n - sum(w in set(t) for t in self.toks) + 0.5)
                              / (sum(w in set(t) for t in self.toks) + 0.5)) for w in vocab}
        # dense (LSA) index
        self.tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words="english")
        X = self.tfidf.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=min(svd_dims, X.shape[0] - 1), random_state=seed)
        V = self.svd.fit_transform(X)
        self.V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        self.k_rrf = k_rrf

    def _bm25(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        q = tokens(query)
        scores = np.zeros(len(self.toks))
        for i, t in enumerate(self.toks):
            for w in q:
                f = t.count(w)
                if f:
                    scores[i] += self.idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * len(t) / self.avgdl))
        return scores

    def _dense(self, query: str) -> np.ndarray:
        v = self.svd.transform(self.tfidf.transform([query]))[0]
        v = v / max(np.linalg.norm(v), 1e-12)
        return self.V @ v

    def search(self, query: str, k: int = 3, mode: str = "hybrid") -> list[dict]:
        bm, de = self._bm25(query), self._dense(query)
        if mode == "bm25":
            fused = bm
        elif mode == "dense":
            fused = de
        else:
            rank_bm = np.argsort(np.argsort(-bm)) + 1
            rank_de = np.argsort(np.argsort(-de)) + 1
            fused = 1 / (self.k_rrf + rank_bm) + 1 / (self.k_rrf + rank_de)
            fused = np.where(bm > 0, fused, 1 / (self.k_rrf + rank_de))  # BM25 counts only on a match
        order = np.argsort(-fused)[:k]
        return [{**self.chunks[i], "score": float(fused[i])} for i in order]

    def by_id(self, chunk_id: str) -> dict | None:
        return next((c for c in self.chunks if c["id"] == chunk_id), None)


def evaluate_retrieval(retriever: HybridRetriever | None = None) -> list[dict]:
    r = retriever or HybridRetriever()
    rows = []
    for mode in ["bm25", "dense", "hybrid"]:
        hits1 = hits3 = rr = 0.0
        for q, gold in GOLD_QUERIES:
            ranked = [c["id"] for c in r.search(q, k=len(r.ids), mode=mode)]
            pos = ranked.index(gold) + 1
            hits1 += pos <= 1
            hits3 += pos <= 3
            rr += 1 / pos
        n = len(GOLD_QUERIES)
        rows.append({"retriever": mode, "recall_at_1": hits1 / n, "recall_at_3": hits3 / n, "mrr": rr / n,
                     "queries": n})
    return rows
