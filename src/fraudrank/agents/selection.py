"""Choose the configuration to carry forward with a paired fold rule, in code.

Every configuration in a loop is scored on the same folds, so two configurations
are compared fold by fold. The fold SD measures how much one model's score moves
between folds; the paired difference removes that shared movement, so it is the
right yardstick for "is A really better than B".

Rule: the configuration with the highest mean beats another one reliably when its
mean paired gain exceeds one paired standard error AND it wins most folds. Among
the configurations it does not beat reliably (itself included), the simplest is
chosen: logistic < random forest < LightGBM, then fewer features, then no
imbalance handling. Ties in simplicity go to the higher mean.
"""
from __future__ import annotations

import numpy as np

MODEL_RANK = {"logistic": 0, "random_forest": 1, "lightgbm": 2}


def simplicity(h: dict) -> tuple:
    return (MODEL_RANK[h["model"]], h["n_features"], h["imbalance"] != "none")


def paired(a: list[float], b: list[float]) -> dict:
    """Paired comparison of a against b on the same folds."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    wins = int((d > 0).sum())
    return {"mean_gain": round(float(d.mean()), 4), "paired_se": round(se, 4), "wins": wins, "n_folds": n,
            "reliably_better": bool(d.mean() > se and wins > n / 2)}


def choose_config(history: list[dict], fold_details: dict[int, list[float]]) -> dict:
    """Return the chosen run id and the paired comparisons behind the choice."""
    best = max(history, key=lambda h: h["pr_auc_mean"])
    comparisons = []
    for h in history:
        if h["run_id"] == best["run_id"]:
            continue
        comparisons.append({"best_run": best["run_id"], "vs_run": h["run_id"],
                            **paired(fold_details[best["run_id"]], fold_details[h["run_id"]])})
    beaten = {c["vs_run"] for c in comparisons if c["reliably_better"]}
    candidates = [h for h in history if h["run_id"] not in beaten]
    chosen = min(candidates, key=lambda h: (simplicity(h), -h["pr_auc_mean"]))
    return {"chosen": chosen["run_id"], "best_mean_run": best["run_id"],
            "kept_simpler": chosen["run_id"] != best["run_id"], "comparisons": comparisons}
