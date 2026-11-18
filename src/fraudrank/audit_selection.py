"""Two-gate audit selection: who enters this cycle's review queue, and with which action.

Gate 1, rule check: billing anomalies that need verification goes straight to review, whatever
the model says. A provider is referred if it has any matching duplicate-like claim, or at
least two inpatient stays that start before the same patient's earlier stay ended.
Gate 2, risk ranking: the model ranks everyone gate 1 did not refer; the top of that
ranking joins the queue. Agreement among size-free models retrained with different
seeds decides full audit or desk review, so the action does not hinge on one seed.
The rule decides whether a provider is reviewed; the model decides when: the queue
is ordered by risk score for both gates.

Everything here is code: no LLM changes who is queued or which action they get.
"""
from __future__ import annotations

import pandas as pd

from . import data as D
from .ranking import ordered_scores, ranks
from .config import (RISK_QUEUE_SIZE, RULE_DUPLICATE_MIN, RULE_OVERLAP_MIN, VOTE_AUDIT, VOTE_BORDERLINE,
                     VOTE_SEEDS)

DUP_KEY = ["Provider", "BeneID", "ClaimStartDt", "InscClaimAmtReimbursed", "ClmDiagnosisCode_1"]


def overlap_pairs(claims: pd.DataFrame):
    """Yield one backing pair per stay overlapping any earlier stay.

    Keep the earlier stay with the latest discharge. Sorting by claim ID also
    makes simultaneous admissions independent of input row order.
    """
    ip = claims.dropna(subset=["AdmissionDt", "DischargeDt"]).sort_values(
        ["Provider", "BeneID", "AdmissionDt", "ClaimID"])
    for _, group in ip.groupby(["Provider", "BeneID"], sort=False):
        previous = None
        for row in group.itertuples(index=False):
            if previous is not None and row.AdmissionDt < previous.DischargeDt:
                yield previous, row
            if previous is None or row.DischargeDt > previous.DischargeDt:
                previous = row


def billing_checks(index: pd.Index, claims: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per provider: claims repeating an earlier claim, and overlapping inpatient stays."""
    claims = D.load_claims() if claims is None else claims
    dup = claims[claims.duplicated(DUP_KEY, keep="first")].groupby("Provider").size()
    overlap = pd.Series([row.Provider for _, row in overlap_pairs(claims)], dtype=str).value_counts()
    return pd.DataFrame({"duplicate_claims": dup, "overlapping_stays": overlap}).reindex(index).fillna(0).astype(int)


EXHIBIT_COLS = ["ClaimID", "BeneID", "ClaimStartDt", "ClaimEndDt", "InscClaimAmtReimbursed",
                "ClmDiagnosisCode_1", "AdmissionDt", "DischargeDt"]


def billing_exhibits(providers: list[str], claims: pd.DataFrame | None = None) -> dict[str, dict]:
    """Return one-line summaries and backing claim rows for each gate-1 finding."""
    claims = D.load_claims() if claims is None else claims
    hit = claims[claims.Provider.isin(set(providers))]
    out: dict[str, dict] = {}
    for p in providers:
        pc = hit[hit.Provider == p]
        entry: dict[str, list] = {"duplicates": [], "overlaps": []}

        marked = pc[pc.duplicated(DUP_KEY, keep=False)]
        for _, grp in marked.groupby(DUP_KEY[1:], dropna=False):
            rows = grp[EXHIBIT_COLS].sort_values("ClaimID")
            amt = int(rows.iloc[0].InscClaimAmtReimbursed)
            bene = str(rows.iloc[0].BeneID)
            date = str(rows.iloc[0].ClaimStartDt)[:10]
            ends = pd.to_datetime(rows["ClaimEndDt"])
            gap = int((ends.max() - ends.min()).days)
            ids = " and ".join(str(r.ClaimID) for _, r in rows.iterrows())
            times = "twice" if len(rows) == 2 else f"{len(rows)} times"
            summary = f"${amt:,} billed {times} for {bene} on {date} ({ids})"
            if gap:
                summary += f" — end dates differ by {gap} day{'s' if gap != 1 else ''}"
            entry["duplicates"].append({
                "summary": summary,
                "claims": [{c: _fmt(rows.iloc[i][c]) for c in EXHIBIT_COLS} for i in range(len(rows))],
                "end_date_gap_days": gap,
            })

        for prev, row in overlap_pairs(pc):
            overlap_days = (prev.DischargeDt - row.AdmissionDt).days
            summary = (f"{row.BeneID} admitted {str(row.AdmissionDt)[:10]}, "
                       f"{overlap_days} day{'s' if overlap_days != 1 else ''} before previous discharge "
                       f"({str(prev.DischargeDt)[:10]}) — "
                       f"${int(row.InscClaimAmtReimbursed):,} overlapping ${int(prev.InscClaimAmtReimbursed):,}")
            entry["overlaps"].append({
                "summary": summary,
                "patient": str(row.BeneID),
                "earlier_stay": {c: _fmt(getattr(prev, c)) for c in EXHIBIT_COLS},
                "later_stay": {c: _fmt(getattr(row, c)) for c in EXHIBIT_COLS},
            })
        out[p] = entry
    return out


def _fmt(v) -> str | float | int | None:
    if pd.isna(v):
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()[:10]
    if hasattr(v, "item"):
        return v.item()
    return v


def rule_check(checks: pd.DataFrame) -> pd.Series:
    """Gate 1 reason per provider ('' when no rule fires)."""
    def reason(row):
        dup, overlap = int(row.duplicate_claims), int(row.overlapping_stays)
        parts = []
        if dup >= RULE_DUPLICATE_MIN:
            parts.append(f"{dup} duplicate claim" + ("s" if dup != 1 else ""))
        if overlap >= RULE_OVERLAP_MIN:
            parts.append(f"{overlap} overlapping hospital stays")
        return "; ".join(parts)
    return checks.apply(reason, axis=1)


def rule_check_dev(checks: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """How often each gate-1 rule fires on development providers, and the share previously flagged."""
    rules = {"duplicate claims >= %d" % RULE_DUPLICATE_MIN: checks.duplicate_claims >= RULE_DUPLICATE_MIN,
             "overlapping stays >= %d" % RULE_OVERLAP_MIN: checks.overlapping_stays >= RULE_OVERLAP_MIN}
    rules["either (gate 1)"] = rules[next(iter(rules))] | list(rules.values())[1]
    return pd.DataFrame([{"rule": name, "providers": int(f.sum()), "share_of_providers": f.mean(),
                          "previously_flagged": int(y[f].sum()), "flagged_rate": y[f].mean() if f.any() else None,
                          "base_rate": y.mean()} for name, f in rules.items()])


def seed_votes(size_free_scores: dict[int, pd.Series]) -> pd.Series:
    """Number of retrained size-free models that put each provider in their top RISK_QUEUE_SIZE."""
    if not size_free_scores:
        raise ValueError("At least one seed score series is required")
    reference = next(iter(size_free_scores.values())).index
    if any(set(s.index) != set(reference) for s in size_free_scores.values()):
        raise ValueError("Seed scores must contain the same provider IDs")
    tops = [ranks(s) <= RISK_QUEUE_SIZE for s in size_free_scores.values()]
    return pd.concat(tops, axis=1).sum(axis=1).astype(int)


def two_gate_selection(score: pd.Series, checks: pd.DataFrame, votes: pd.Series) -> pd.DataFrame:
    """The review queue, ordered by risk score (highest first), then provider ID, so the order never
    depends on load order. Gate 1 guarantees a place and a full audit; it does not move a provider up."""
    score = ordered_scores(score)
    reason = rule_check(checks.reindex(score.index))
    rank = ranks(score)
    gate1 = reason[reason != ""].index
    rest = score.drop(gate1)
    gate2 = ordered_scores(rest).index[:RISK_QUEUE_SIZE]
    rows = []
    for p in list(gate1) + list(gate2):
        v = int(votes[p])
        if p in gate1:
            gate, action, note = "1_rule_check", "priority_audit", reason[p]
        elif v >= VOTE_AUDIT:
            gate, action, note = "2_risk_ranking", "priority_audit", ""
        elif v >= VOTE_BORDERLINE:
            gate, action, note = "2_risk_ranking", "desk_review", "borderline"
        else:
            gate, action, note = "2_risk_ranking", "desk_review", ""
        rows.append({"provider": p, "gate": gate, "action": action, "reason": note,
                     "risk_score": float(score[p]), "risk_rank": int(rank[p]), "votes": v, "of_models": len(VOTE_SEEDS),
                     "duplicate_claims": int(checks.loc[p, "duplicate_claims"]),
                     "overlapping_stays": int(checks.loc[p, "overlapping_stays"])})
    q = pd.DataFrame(rows).sort_values(["risk_score", "provider"], ascending=[False, True])
    return q.assign(position=range(1, len(q) + 1)).reset_index(drop=True)


def queue_evaluation(score: pd.Series, y: pd.Series, gate1: pd.Series, queue: pd.DataFrame) -> pd.DataFrame:
    """Ranking quality with and without the gates on the same providers (paired bootstrap), and the
    queue against the model's own top of the same size."""
    from .metrics import bootstrap_ci, evaluate, paired_bootstrap_diff
    y_, s_ = y.reindex(score.index).to_numpy(), score.to_numpy()
    first = (score + gate1.reindex(score.index).astype(float) * 10).to_numpy()   # gate 1 ahead of everyone
    rows = []
    for name, sc in [("model only", s_), ("two-gate, queue by risk score", s_),
                     ("two-gate, gate 1 first (not used)", first)]:
        m = evaluate(y_, sc)
        lo, hi = bootstrap_ci(y_, sc, "pr_auc")
        d, dlo, dhi = paired_bootstrap_diff(y_, sc, s_, "pr_auc")
        rows.append({"ranking": name, "pr_auc": m["pr_auc"], "pr_auc_lo": lo, "pr_auc_hi": hi,
                     "diff_vs_model": d, "diff_lo": dlo, "diff_hi": dhi})
    rest = ~gate1.reindex(score.index).to_numpy()
    m = evaluate(y_[rest], s_[rest])
    lo, hi = bootstrap_ci(y_[rest], s_[rest], "pr_auc")
    rows.append({"ranking": "model only, providers gate 1 did not refer", "pr_auc": m["pr_auc"],
                 "pr_auc_lo": lo, "pr_auc_hi": hi})
    top = ordered_scores(score).index[:len(queue)]
    rows.append({"ranking": f"model top {len(queue)}: previously flagged", "flagged": int(y[top].sum())})
    rows.append({"ranking": f"review queue ({len(queue)}): previously flagged", "flagged": int(y[queue.provider].sum())})
    return pd.DataFrame(rows)
