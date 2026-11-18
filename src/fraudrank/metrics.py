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
    if not 1 <= k <= len(y):
        raise ValueError("k must be between 1 and the number of providers")
    order = np.argsort(-score, kind="stable")[:k]
    tp = y[order].sum()
    return tp / k, tp / max(1, y.sum())


def evaluate(y, score, ks: dict[str, int] | None = None) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    ks = k_list(len(y)) if ks is None else ks
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
    ks = k_list(len(y)) if ks is None else ks
    vals = []
    for _ in range(n):
        idx = _bootstrap_indices(rng, pos, neg)
        vals.append(_metric(y[idx], score[idx], metric, ks))
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
        idx = _bootstrap_indices(rng, pos, neg)
        diffs.append(_metric(y[idx], a[idx], metric, ks) - _metric(y[idx], b[idx], metric, ks))
    full = evaluate(y, a, ks)[metric] - evaluate(y, b, ks)[metric]
    return float(full), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _bootstrap_indices(rng, pos, neg):
    """Preserve original provider order within score ties, never label order.

    Original rows must use the same canonical provider ordering as evaluation.
    Sorting sampled positions also keeps paired comparisons on identical draws
    without changing the random stream used for the published PR-AUC intervals.
    """
    return np.sort(np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]))


def _metric(y, score, metric, ks):
    # Bootstrap only the requested statistic, rather than recomputing every
    # ranking metric thousands of times.
    if metric == "pr_auc":
        return average_precision_score(y, score)
    if metric == "roc_auc":
        return roc_auc_score(y, score)
    for prefix, position in (("prec_", 0), ("rec_", 1)):
        if metric.startswith(prefix):
            return precision_recall_at_k(y, score, ks[metric[len(prefix):]])[position]
    return evaluate(y, score, ks)[metric]
