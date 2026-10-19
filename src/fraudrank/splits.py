"""Frozen provider-level splits.

One stratified 80/20 split separates a development set from a test set that is
scored once, after every modelling choice is fixed. Repeated stratified K-fold on
the development set gives 25 train/validation pairs for model selection and for
mean +/- SD reporting. A time-based split is not possible: the label is a single
flag per provider for the whole 13-month window.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from .config import CV_REPEATS, CV_SPLITS, PROCESSED_DIR, SEED, TEST_SIZE

SPLIT_FILE = PROCESSED_DIR / "splits.json"


def make_splits(df: pd.DataFrame) -> dict:
    providers = df.index.to_numpy()
    dev, test = train_test_split(providers, test_size=TEST_SIZE, stratify=df["y"], random_state=SEED)
    dev = np.sort(dev)
    y_dev = df.loc[dev, "y"].to_numpy()
    rskf = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=SEED)
    folds = [{"train": dev[tr].tolist(), "val": dev[va].tolist()}
             for tr, va in rskf.split(dev, y_dev)]
    splits = {"dev": dev.tolist(), "test": np.sort(test).tolist(), "folds": folds}
    SPLIT_FILE.write_text(json.dumps(splits))
    return splits


def load_splits() -> dict:
    return json.loads(SPLIT_FILE.read_text())
