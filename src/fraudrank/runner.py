"""Shared machinery: load processed data, run a model over the 25 CV folds, and
score the held-out test set once."""
from __future__ import annotations

import json
from typing import Callable

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR
from .metrics import evaluate, k_list
from .network import NETWORK_COLS, ProviderGraph
from .splits import load_splits

_GRAPH = None


def load_all():
    df = pd.read_pickle(PROCESSED_DIR / "provider_features.pkl")
    families = json.loads((PROCESSED_DIR / "families.json").read_text())
    return df, families, load_splits()


def graph(df) -> ProviderGraph:
    global _GRAPH
    if _GRAPH is None:
        edges = pd.read_pickle(PROCESSED_DIR / "provider_bene_edges.pkl")
        _GRAPH = ProviderGraph(edges, df.index)
    return _GRAPH


def _xy(df, provs, cols, net_frame=None):
    X = df.loc[provs, cols]
    if net_frame is not None:
        X = X.join(net_frame)
    return X, df.loc[provs, "y"].to_numpy()


def cv_evaluate(make_model: Callable, cols: list[str], df: pd.DataFrame, splits: dict,
                use_network: bool = False, label_transform: Callable | None = None,
                folds: list | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Run over every CV fold. Returns per-fold metrics and the mean out-of-fold score.

    label_transform(y, fold_index) lets E7 corrupt training labels; validation
    labels always stay clean.
    """
    rows, oof = [], {}
    g = graph(df) if use_network else None
    for i, fold in enumerate(folds or splits["folds"]):
        tr, va = fold["train"], fold["val"]
        net_tr = net_va = None
        y_tr_raw = df.loc[tr, "y"].to_numpy()
        y_fit = label_transform(y_tr_raw, i) if label_transform else y_tr_raw
        if use_network:
            net_tr, net_va = g.features(tr, y_fit, va)
        X_tr, _ = _xy(df, tr, cols, net_tr)
        X_va, y_va = _xy(df, va, cols, net_va)
        model = make_model().fit(X_tr, y_fit)
        s = model.score(X_va)
        m = evaluate(y_va, s, k_list(len(y_va)))
        m["fold"] = i
        rows.append(m)
        for p, v in zip(va, s):
            oof.setdefault(p, []).append(v)
    res = pd.DataFrame(rows)
    oof_mean = pd.Series({p: float(np.mean(v)) for p, v in oof.items()})
    return res, oof_mean


def test_evaluate(make_model: Callable, cols: list[str], df: pd.DataFrame, splits: dict,
                  use_network: bool = False):
    """Fit on the whole development set, score the test set once."""
    dev, test = splits["dev"], splits["test"]
    net_dev = net_test = None
    if use_network:
        net_dev, net_test = graph(df).features(dev, df.loc[dev, "y"].to_numpy(), test)
    X_dev, y_dev = _xy(df, dev, cols, net_dev)
    X_te, y_te = _xy(df, test, cols, net_test)
    model = make_model().fit(X_dev, y_dev)
    s = pd.Series(model.score(X_te), index=test)
    return model, s, y_te, X_te


def summarise(res: pd.DataFrame, name: str) -> dict:
    out = {"model": name}
    for c in res.columns:
        if c == "fold":
            continue
        out[f"{c}_mean"] = res[c].mean()
        out[f"{c}_sd"] = res[c].std()
    return out


def network_cols() -> list[str]:
    return list(NETWORK_COLS)
