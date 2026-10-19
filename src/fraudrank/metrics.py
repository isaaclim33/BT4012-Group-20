"""Ranking metrics for an audit queue.

PR-AUC (average precision) is the primary metric because positives are 9.4% of
providers and ROC-AUC rewards correct ordering among the many negatives that an
investigator never sees. precision@k and recall@k describe a queue of fixed size k.
Every precision@k is reported beside its ceiling min(1, positives / k), so a
result such as 0.48 at k = 200 is read against the best achievable value.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .config import BOOTSTRAP, K_FRACTIONS, K_VALUES, SEED


def k_list(n: int) -> dict[str, int]:
    ks = {f"k{k}": k for k in K_VALUES if k < n}
    ks.update({f"top{int(f * 100)}pct": max(1, int(round(f * n))) for f in K_FRACTIONS})
    return ks


def precision_recall_at_k(y: np.ndarray, score: np.ndarray, k: int) -> tuple[float, float]:
    order = np.argsort(-score, kind="stable")[:k]
    tp = y[order].sum()
    return tp / k, tp / max(1, y.sum())


def evaluate(y, score, ks: dict[str, int] | None = None) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    ks = ks or k_list(len(y))
    out = {"pr_auc": average_precision_score(y, score), "roc_auc": roc_auc_score(y, score)}
    for name, k in ks.items():
        p, r = precision_recall_at_k(y, score, k)
        out[f"prec_{name}"] = p
        out[f"rec_{name}"] = r
        out[f"ceil_{name}"] = min(1.0, y.sum() / k)
    return out


def threshold_metrics(y, score, threshold: float) -> dict[str, float]:
    """Confusion-matrix metrics when every provider above `threshold` is flagged."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(score) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "f1": f1_score(y, pred), "fpr": fp / max(1, fp + tn), "fnr": fn / max(1, fn + tp)}


def bootstrap_ci(y, score, metric: str = "pr_auc", n: int = BOOTSTRAP, seed: int = SEED,
                 ks: dict[str, int] | None = None) -> tuple[float, float]:
    """95% percentile interval, resampling providers with stratification by label."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    ks = ks or k_list(len(y))
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(evaluate(y[idx], score[idx], ks)[metric])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_diff(y, score_a, score_b, metric: str = "pr_auc", n: int = BOOTSTRAP,
                          seed: int = SEED) -> tuple[float, float, float]:
    """Difference a - b on the same resamples; returns (diff, lo, hi)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    a, b = np.asarray(score_a, float), np.asarray(score_b, float)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    ks = k_list(len(y))
    diffs = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs.append(evaluate(y[idx], a[idx], ks)[metric] - evaluate(y[idx], b[idx], ks)[metric])
    full = evaluate(y, a, ks)[metric] - evaluate(y, b, ks)[metric]
    return float(full), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
