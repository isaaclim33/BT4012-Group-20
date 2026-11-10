"""Evidence tools for the Case-file agent and deterministic checks on its briefs.

Structured evidence (feature values, peer percentiles, SHAP drivers) comes from
code. Unstructured evidence (playbook sections) comes from the hybrid retriever.
Every brief is then checked without any LLM:
* numeric faithfulness: each driver's value, percentile and SHAP match the tool output;
* citation existence: each cited chunk was actually retrieved for this provider;
* citation support: the cited section lists at least one of the brief's driver features.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .rag import HybridRetriever
from .schemas import CaseBrief

FEATURE_TEXT = {
    "total_reimb": "very high total reimbursement", "ip_total_reimb": "high inpatient reimbursement total",
    "op_total_reimb": "high outpatient reimbursement total", "n_claims": "very many claims",
    "n_ip_claims": "many inpatient claims", "n_bene": "many beneficiaries",
    "reimb_per_claim": "high payment per claim", "reimb_per_bene": "high payment per beneficiary",
    "ip_reimb_mean": "high average inpatient payment", "ip_reimb_max": "very high maximum inpatient payment",
    "los_mean": "long average hospital stay", "los_max": "very long maximum stay",
    "los_long_share": "many stays of ten days or more", "claims_per_bene": "many claims per patient",
    "claims_per_attending": "many claims per attending physician",
    "top_attending_share": "one attending physician on most claims", "att_is_op_share":
    "attending and operating physician often the same", "dup_same_share": "duplicate claims within provider",
    "dup_any_share": "duplicate claims across providers", "diag_per_claim": "many diagnosis codes per claim",
    "ip_surgery_share": "procedures on most inpatient stays", "dgc_per_ip_claim":
    "many diagnosis-related groups per inpatient claim", "n_chronic_mean": "patients with many chronic conditions",
    "has_ip": "bills inpatient care", "ip_claim_share": "high share of inpatient claims",
}


def peer_percentiles(df: pd.DataFrame, peers: list[str], provider: str, cols: list[str]) -> dict[str, float]:
    """Percentile of the provider's value among development providers of the same service type."""
    ptype = (df.loc[provider, "has_ip"], df.loc[provider, "has_op"])
    peer = df.loc[peers]
    peer = peer[(peer.has_ip == ptype[0]) & (peer.has_op == ptype[1])]
    return {c: float((peer[c] <= df.loc[provider, c]).mean() * 100) for c in cols}


def build_evidence(provider: str, rank: int, score: float, X: pd.DataFrame, shap: pd.DataFrame,
                   df: pd.DataFrame, dev: list[str], retriever: HybridRetriever | None, n_drivers: int = 5) -> dict:
    s = shap.loc[provider]
    top = s.reindex(s.abs().sort_values(ascending=False).index)[:n_drivers]
    pct = peer_percentiles(df, dev, provider, list(top.index))
    drivers = [{"feature": f, "description": FEATURE_TEXT.get(f, f), "value": round(float(X.loc[provider, f]), 4),
                "peer_percentile": round(pct[f], 1), "shap": round(float(top[f]), 4)} for f in top.index]
    ptype = {(1, 1): "inpatient and outpatient", (1, 0): "inpatient only", (0, 1): "outpatient only"}[
        (int(df.loc[provider, "has_ip"]), int(df.loc[provider, "has_op"]))]
    ev = {"provider": provider, "risk_rank": rank, "risk_score": round(float(score), 4), "provider_type": ptype,
          "claims": int(df.loc[provider, "n_claims"]), "total_reimbursement": float(df.loc[provider, "total_reimb"]),
          "drivers": drivers}
    if retriever is not None:
        seen, chunks = set(), []
        for d in drivers[:3]:                       # one focused query per driver
            if d["shap"] <= 0:
                continue
            for c in retriever.search(f"provider with {d['description']}", k=2):
                if c["id"] not in seen:
                    seen.add(c["id"])
                    chunks.append({"chunk_id": c["id"], "title": c["title"], "text": c["text"],
                                   "signals": c["signals"]})
        ev["playbook"] = chunks
    return ev


CLAIMS = ["committed fraud", "is fraudulent", "confirmed fraud", "proven fraud", "engaged in fraud"]
NEGATIONS = re.compile(r"\b(not|no|never|without|rather than|nor|cannot be|isn't|is not)\b[^.]{0,40}$")


def asserts_fraud(summary: str) -> bool:
    """True if the summary states fraud as fact. A phrase preceded in the same sentence by a
    negation ("is a suspicion marker, not confirmed fraud") is a correct hedge, not a claim."""
    low = summary.lower()
    for phrase in CLAIMS:
        for m in re.finditer(re.escape(phrase), low):
            before = low[:m.start()].split(".")[-1]
            if not NEGATIONS.search(before):
                return True
    return False


def check_brief(brief: CaseBrief, evidence: dict) -> dict:
    tool = {d["feature"]: d for d in evidence["drivers"]}
    num_ok = num_total = 0
    for d in brief.drivers:
        num_total += 1
        t = tool.get(d.feature)
        ok = (t is not None
              and np.isclose(d.value, t["value"], rtol=0.01, atol=1e-3)
              and abs(d.peer_percentile - t["peer_percentile"]) <= 0.5
              and np.isclose(d.shap, t["shap"], rtol=0.01, atol=1e-3))
        num_ok += bool(ok)
    retrieved = {c["chunk_id"]: c for c in evidence.get("playbook", [])}
    drivers = {d.feature for d in brief.drivers}
    cite_exist = cite_support = 0
    for cit in brief.typologies:
        c = retrieved.get(cit.chunk_id)
        cite_exist += c is not None
        cite_support += c is not None and bool(drivers & set(c["signals"]))
    n_cit = len(brief.typologies)
    return {"provider": brief.provider, "drivers": num_total, "drivers_faithful": num_ok,
            "citations": n_cit, "citations_exist": cite_exist, "citations_supported": cite_support,
            "rank_matches": brief.risk_rank == evidence["risk_rank"],
            "says_fraud_confirmed": asserts_fraud(brief.summary),
            "action": brief.recommended_action}
