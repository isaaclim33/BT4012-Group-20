"""Code that carries out agent decisions. Agents never execute anything themselves."""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from .. import models as M
from ..config import PROCESSED_DIR, SEED
from .schemas import ClaimAggFeature, ExperimentProposal, FeatureProposal, RatioFeature

_CLAIMS = None


def _claims() -> pd.DataFrame:
    global _CLAIMS
    if _CLAIMS is None:
        _CLAIMS = pd.read_pickle(PROCESSED_DIR / "claims_enriched.pkl")
    return _CLAIMS


def compute_agent_features(proposal: FeatureProposal, base: pd.DataFrame) -> pd.DataFrame:
    """Turn a validated proposal into columns. Unknown names never reach this point."""
    c = _claims()
    out = pd.DataFrame(index=base.index)
    for f in proposal.claim_aggregates:
        sub = c if f.scope == "all" else c[c.is_ip == (1 if f.scope == "ip" else 0)]
        g = sub.groupby("Provider")[f.column]
        if f.agg == "share_positive":
            col = g.apply(lambda s: (s > 0).mean())
        else:
            col = getattr(g, f.agg)()
        out[f.name] = col.reindex(base.index)
    for f in proposal.ratios:
        den = base[f.denominator].to_numpy()
        out[f.name] = np.where(den > 0, base[f.numerator].to_numpy() / np.where(den > 0, den, 1), 0.0)
    return out.fillna(0).astype(float)


def duplicate_gate(feats: pd.DataFrame, base: pd.DataFrame, existing: list[str], providers: list[str],
                   max_corr: float = 0.95, max_dominant: float = 0.995) -> tuple[list[str], dict[str, str]]:
    """Keep a proposed column only if it adds information: no single value on more than max_dominant
    of providers, and |rank correlation| <= max_corr with every existing or already-kept column.
    Correlations skip providers where both columns are 0: outpatient-only providers are 0 on every
    inpatient feature, and that shared block would make unrelated features look identical."""
    ref = base.loc[providers, existing]
    kept, rejected = [], {}
    with np.errstate(all="ignore"):
        for f in feats.columns:
            s = feats.loc[providers, f]
            if s.value_counts(normalize=True).iloc[0] > max_dominant:
                rejected[f] = "near-constant"
                continue
            for c in list(ref.columns):
                m = (s != 0) | (ref[c] != 0)
                r = abs(s[m].rank().corr(ref.loc[m, c].rank())) if m.sum() >= 50 else 0.0
                if r > max_corr:
                    rejected[f] = f"duplicates {c} (rank correlation {r:.3f})"
                    break
            else:
                kept.append(f)
                ref = ref.assign(**{f: s})
    return kept, rejected


def describe_feature(f: ClaimAggFeature | RatioFeature) -> str:
    if isinstance(f, ClaimAggFeature):
        return f"{f.name}: {f.agg} of claim column {f.column} over {f.scope} claims, per provider"
    return f"{f.name}: provider {f.numerator} divided by provider {f.denominator}"


def make_model(p: ExperimentProposal):
    params = {k: v for k, v in p.params.model_dump().items() if v is not None}
    sampler = SMOTE(random_state=SEED) if p.imbalance == "smote" else None
    balanced = p.imbalance == "class_weight"
    if p.model == "logistic":
        return lambda: M.logistic(C=params.get("C", 0.1), balanced=balanced, sampler=sampler)
    if p.model == "random_forest":
        return lambda: M.random_forest(params.get("min_samples_leaf", 3),
                                       params.get("max_features", "sqrt"), balanced, sampler)
    lgb_keys = ["num_leaves", "learning_rate", "n_estimators", "min_child_samples", "colsample_bytree"]
    return lambda: M.lightgbm({k: params[k] for k in lgb_keys if k in params}, balanced, sampler)


def feature_set(p: ExperimentProposal, families: dict, agent_cols: list[str]) -> list[str]:
    all_cols = [c for cols in families.values() for c in cols]
    no_size = [c for f, cols in families.items() if f != "size" for c in cols]
    base = no_size if p.feature_set.startswith("no_size") else all_cols
    return base + (agent_cols if p.feature_set.endswith("+agent") else [])
