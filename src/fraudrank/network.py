"""Provider network features built without label leakage.

Graph: providers are linked when they bill the same beneficiary; the edge weight
is the number of shared beneficiaries. Edges use claims only, which are known for
every provider at scoring time, so the structure is fit on all providers.

Labels enter only through personalised PageRank (PPR) seeds and neighbour
fraud shares. Rules that keep this leak-free:
* Validation and test providers are scored with seeds from the training fold only.
* A training provider must not see its own label, so training rows get
  out-of-fold values: seeds come from the other inner folds of the training set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold

from .config import INNER_SPLITS, SEED

NETWORK_COLS = ["net_degree", "net_strength", "net_ppr_lift", "net_nbr_fraud_share",
                "net_nbr_fraud_weight"]


class ProviderGraph:
    def __init__(self, edges: pd.DataFrame, providers: pd.Index):
        self.providers = providers
        pidx = pd.Series(np.arange(len(providers)), index=providers)
        bcodes, _ = pd.factorize(edges["BeneID"])
        rows = pidx.loc[edges["Provider"]].to_numpy()
        B = sp.csr_matrix((np.ones(len(edges)), (rows, bcodes)), shape=(len(providers), bcodes.max() + 1))
        A = (B @ B.T).tocsr()
        A.setdiag(0)
        A.eliminate_zeros()
        self.A = A
        self.degree = np.asarray((A > 0).sum(axis=1)).ravel()
        self.strength = np.asarray(A.sum(axis=1)).ravel()
        inv = np.where(self.strength > 0, 1.0 / np.maximum(self.strength, 1e-12), 0.0)
        self.P = (A @ sp.diags(inv)).tocsr()     # column-stochastic transition matrix

    def index_of(self, provs) -> np.ndarray:
        return self.providers.get_indexer(provs)

    def ppr(self, seed_idx: np.ndarray, alpha: float = 0.85, iters: int = 50) -> np.ndarray:
        n = self.A.shape[0]
        s = np.zeros(n)
        if len(seed_idx) == 0:
            return s
        s[seed_idx] = 1.0 / len(seed_idx)
        r = s.copy()
        for _ in range(iters):
            r = alpha * (self.P @ r) + (1 - alpha) * s
        return r

    def _label_features(self, known_pos: np.ndarray, known_all: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Features for `target` nodes given which nodes have a known label."""
        r_fraud = self.ppr(known_pos)
        r_all = self.ppr(known_all)
        # lift: how much more the fraud walk visits a node than a walk from all known
        # providers; dividing by r_all removes the pull of sheer size and degree
        lift = np.where(r_all[target] > 0, r_fraud[target] / np.maximum(r_all[target], 1e-15), 0.0)
        pos_mask = np.zeros(self.A.shape[0]); pos_mask[known_pos] = 1
        all_mask = np.zeros(self.A.shape[0]); all_mask[known_all] = 1
        A_t = self.A[target]
        w_pos = A_t @ pos_mask
        w_all = A_t @ all_mask
        share = np.where(w_all > 0, w_pos / np.maximum(w_all, 1e-12), 0.0)
        return np.column_stack([lift, share, np.log1p(w_pos)])

    def features(self, train_provs, train_y, score_provs) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (train features out-of-fold, score features seeded on all of train)."""
        tr = self.index_of(train_provs)
        y = np.asarray(train_y)
        sc = self.index_of(score_provs)

        def frame(idx, lab):
            return pd.DataFrame(np.column_stack([np.log1p(self.degree[idx]), np.log1p(self.strength[idx]), lab]),
                                columns=NETWORK_COLS)

        lab_score = self._label_features(tr[y == 1], tr, sc)
        lab_train = np.zeros((len(tr), 3))
        inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=SEED)
        for a, b in inner.split(tr, y):
            lab_train[b] = self._label_features(tr[a][y[a] == 1], tr[a], tr[b])
        f_tr = frame(tr, lab_train).set_index(pd.Index(train_provs))
        f_sc = frame(sc, lab_score).set_index(pd.Index(score_provs))
        return f_tr, f_sc


def homophily_test(graph: ProviderGraph, provs, y) -> dict:
    """Homophily test: observed share of cross-label edges vs the expected 2*l*f."""
    idx = graph.index_of(provs)
    y = np.asarray(y)
    sub = sp.triu(graph.A[idx][:, idx], k=1).tocoo()
    same_f = ((y[sub.row] == 1) & (y[sub.col] == 1)).sum()
    same_l = ((y[sub.row] == 0) & (y[sub.col] == 0)).sum()
    cross = len(sub.row) - same_f - same_l
    f = y.mean(); l = 1 - f
    expected = 2 * l * f
    observed = cross / max(1, len(sub.row))
    # one-sided binomial z-test for observed < expected
    n = len(sub.row)
    z = (observed - expected) / np.sqrt(expected * (1 - expected) / n)
    # Degree-adjusted expectation: big providers have many edges and are flagged more
    # often, so the naive 2lf overstates how many cross edges chance would produce.
    # Under a degree-preserving null, an edge end lands on a flagged node with
    # probability p = (edge ends on flagged nodes) / (all edge ends).
    deg = np.bincount(np.concatenate([sub.row, sub.col]), minlength=len(idx))
    p = deg[y == 1].sum() / max(1, deg.sum())
    expected_deg = 2 * p * (1 - p)
    z_deg = (observed - expected_deg) / np.sqrt(expected_deg * (1 - expected_deg) / n)
    return {"edges": int(n), "fraud_fraud": int(same_f), "legit_legit": int(same_l), "cross": int(cross),
            "fraud_share": float(f), "expected_cross_share": float(expected),
            "observed_cross_share": float(observed), "z": float(z),
            "edge_end_fraud_share": float(p), "expected_cross_share_degree_adj": float(expected_deg),
            "z_degree_adj": float(z_deg)}
