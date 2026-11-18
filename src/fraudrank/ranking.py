"""One label-independent ordering for provider scores, queues and seed votes."""
import numpy as np
import pandas as pd


def ordered_scores(scores: pd.Series) -> pd.Series:
    if not scores.index.is_unique or scores.isna().any() or not np.isfinite(scores).all():
        raise ValueError("Provider scores require unique IDs and finite values")
    # Stable score sorting preserves ascending provider ID within ties.
    return scores.sort_index().sort_values(ascending=False, kind="stable")


def ranks(scores: pd.Series) -> pd.Series:
    ordered = ordered_scores(scores)
    return pd.Series(range(1, len(ordered) + 1), index=ordered.index).reindex(scores.index)
