"""Data behind the review dashboard, computed by code for every scored provider.

The dashboard covers all 1,082 held-out providers. Every number on it comes from
here; the Dashboard agent only adds plain-language text for the top of the queue.
The queue and each provider's action come from the two-gate audit selection
(audit_selection.py, saved to audit_queue.csv); the Case-file agent's own suggested
action is kept for comparison but never decides anything.

Peer group = development-set providers with the same type of care (inpatient and
outpatient / inpatient only / outpatient only), so a hospital is compared with
hospitals, not with small clinics. Known labels are deliberately left out: in
deployment the reviewer would not know whether a provider is flagged.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import BATCH_SIZE, CACHE_DIR, RISK_QUEUE_SIZE, TABLES_DIR, VOTE_AUDIT
from .rag import HybridRetriever

CARE = {(1, 1): "Hospital and outpatient", (1, 0): "Hospital only", (0, 1): "Outpatient only"}

# Metrics shown to reviewers: (column, label, unit, log scale for charts, what it means)
METRICS = [
    ("total_reimb", "Total payments", "usd", True, "Everything Medicare paid this provider over the period."),
    ("n_claims", "Claims", "count", True, "Number of claims submitted."),
    ("n_bene", "Patients", "count", True, "Different patients billed for."),
    ("reimb_per_claim", "Payment per claim", "usd", True, "Average payment for one claim."),
    ("reimb_per_bene", "Payment per patient", "usd", True, "Average total paid per patient."),
    ("claims_per_bene", "Claims per patient", "ratio", False, "How often the same patient is billed."),
    ("ip_claim_share", "Hospital-stay claims", "share", False, "Share of claims that are hospital stays."),
    ("los_mean", "Average hospital stay", "days", False, "Average length of a hospital stay."),
    ("los_max", "Longest hospital stay", "days", False, "The single longest stay billed."),
    ("los_long_share", "Stays of 10+ days", "share", False, "Share of hospital stays lasting 10 days or more."),
    ("diag_per_claim", "Diagnoses per claim", "ratio", False, "How many diagnosis codes each claim carries."),
    ("distinct_diag1_per_claim", "Variety of main diagnoses", "share", False,
     "Different main diagnoses relative to the number of claims."),
    ("n_attending", "Doctors billing", "count", True, "Different attending doctors on the claims."),
    ("top_attending_share", "Claims under busiest doctor", "share", False,
     "Share of claims carrying the most frequent attending doctor."),
    ("dup_same_share", "Possible duplicate claims", "share", False,
     "Claims matching another of this provider's claims on patient, date, diagnosis and amount."),
    ("n_chronic_mean", "Chronic conditions per patient", "ratio", False,
     "Average number of long-term conditions among the provider's patients."),
]


# Neutral name and unit for every model feature, so a factor reads correctly whether its
# value is high or low.
FEATURE_NAMES = {
    "n_claims": ("Claims", "count"), "n_ip_claims": ("Hospital-stay claims", "count"),
    "n_op_claims": ("Outpatient claims", "count"), "n_bene": ("Patients", "count"),
    "total_reimb": ("Total payments", "usd"), "ip_total_reimb": ("Hospital-stay payments", "usd"),
    "op_total_reimb": ("Outpatient payments", "usd"), "n_attending": ("Doctors billing", "count"),
    "n_operating": ("Operating doctors", "count"), "n_other_phys": ("Other doctors named", "count"),
    "n_states": ("States patients come from", "count"), "n_dgc": ("Diagnosis groups used", "count"),
    "n_admit_diag": ("Different admission reasons", "count"), "has_ip": ("Bills hospital stays", "flag"),
    "has_op": ("Bills outpatient care", "flag"), "ip_claim_share": ("Hospital-stay claims", "share"),
    "ip_reimb_share": ("Payments from hospital stays", "share"), "reimb_per_claim": ("Payment per claim", "usd"),
    "reimb_per_bene": ("Payment per patient", "usd"), "ip_reimb_mean": ("Average hospital-stay payment", "usd"),
    "ip_reimb_max": ("Largest hospital-stay payment", "usd"), "op_reimb_mean": ("Average outpatient payment", "usd"),
    "op_reimb_median": ("Typical outpatient payment", "usd"), "op_reimb_max": ("Largest outpatient payment", "usd"),
    "op_deductible_mean": ("Average outpatient deductible", "usd"), "zero_reimb_share": ("Claims paid nothing", "share"),
    "los_mean": ("Average hospital stay", "days"), "los_max": ("Longest hospital stay", "days"),
    "los_long_share": ("Stays of 10+ days", "share"), "ip_claim_days_mean": ("Average hospital claim period", "days"),
    "op_claim_days_mean": ("Average outpatient claim period", "days"), "claims_per_bene": ("Claims per patient", "ratio"),
    "diag_per_claim": ("Diagnoses per claim", "ratio"), "ip_proc_per_claim": ("Procedures per hospital stay", "ratio"),
    "ip_surgery_share": ("Hospital stays with a procedure", "share"),
    "dgc_per_ip_claim": ("Diagnosis groups per hospital stay", "ratio"),
    "admit_diag_per_claim": ("Admission reasons per claim", "ratio"),
    "distinct_diag1_per_claim": ("Variety of main diagnoses", "share"),
    "claims_per_attending": ("Claims per doctor", "ratio"), "top_attending_share": ("Claims under busiest doctor", "share"),
    "operating_present_share": ("Claims naming a surgeon", "share"),
    "other_present_share": ("Claims naming an extra doctor", "share"),
    "att_is_op_share": ("Same doctor attending and operating", "share"),
    "dup_any_share": ("Claims matching another provider's", "share"),
    "dup_same_share": ("Possible duplicate claims", "share"), "age_mean": ("Average patient age", "years"),
    "dead_share": ("Patients who died in the period", "share"), "n_chronic_mean": ("Chronic conditions per patient", "ratio"),
    "renal_share": ("Patients with kidney failure", "share"),
    "bene_ip_annual_mean": ("Patients' yearly hospital costs", "usd"),
    "bene_op_annual_mean": ("Patients' yearly outpatient costs", "usd"),
    "states_per_bene": ("States per patient", "ratio"),
    "share_alzheimer": ("Patients with Alzheimer's", "share"), "share_heartfailure": ("Patients with heart failure", "share"),
    "share_kidneydisease": ("Patients with kidney disease", "share"), "share_cancer": ("Patients with cancer", "share"),
    "share_obstrpulmonary": ("Patients with lung disease", "share"), "share_depression": ("Patients with depression", "share"),
    "share_diabetes": ("Patients with diabetes", "share"), "share_ischemicheart": ("Patients with heart disease", "share"),
    "share_osteoporasis": ("Patients with osteoporosis", "share"),
    "share_rheumatoidarthritis": ("Patients with rheumatoid arthritis", "share"),
    "share_stroke": ("Patients who had a stroke", "share"),
}
SIZE_FEATURES = {"n_claims", "n_ip_claims", "n_op_claims", "n_bene", "total_reimb", "ip_total_reimb",
                 "op_total_reimb", "n_attending", "n_operating", "n_other_phys", "n_states", "n_dgc", "n_admit_diag"}


def _pattern_text(chunk: dict) -> str:
    """Playbook text without its leading 'T6 Title.' label, which the dashboard shows as a heading."""
    text = chunk["text"]
    head = f"{chunk['id']} "
    if text.startswith(head):
        text = text[len(head):]
        text = text.split(". ", 1)[1] if ". " in text else text
    return text


def _care(row) -> str:
    return CARE[(int(row["has_ip"]), int(row["has_op"]))]


def _hist(values: np.ndarray, log: bool, bins: int = 28) -> dict:
    v = values[np.isfinite(values)]
    if log:
        v = v[v > 0]
        if len(v) == 0:
            return {"edges": [], "counts": [], "log": True}
        edges = np.logspace(np.log10(v.min()), np.log10(v.max() * 1.0001), bins + 1)
    else:
        lo, hi = float(np.min(v)), float(np.max(v))
        edges = np.linspace(lo, hi if hi > lo else lo + 1, bins + 1)
    counts, _ = np.histogram(v, edges)
    return {"edges": [round(float(e), 6) for e in edges], "counts": counts.tolist(), "log": log}


def build(df: pd.DataFrame, splits: dict) -> dict:
    scores = pd.read_pickle(CACHE_DIR / "test_scores.pkl")
    shap = pd.read_pickle(CACHE_DIR / "shap_test_lightgbm.pkl")
    shap_ns = pd.read_pickle(CACHE_DIR / "shap_test_lgbm_no_size.pkl")   # size-free model
    test = list(scores.index)
    dev = df.loc[splits["dev"]]
    dev = dev.assign(care=dev.apply(_care, axis=1))

    rank = scores["lightgbm"].rank(ascending=False, method="first").astype(int)
    size_free_rank = scores["lgbm:no_size"].rank(ascending=False, method="first").astype(int)
    order = rank.sort_values().index
    aq = pd.read_csv(TABLES_DIR / "agents" / "audit_queue.csv").fillna({"reason": ""}).set_index("provider")

    # peer distributions and quantiles per care type
    peers = {}
    for care, g in dev.groupby("care"):
        peers[care] = {"n": int(len(g)), "metrics": {}}
        for col, _, _, log, _ in METRICS:
            v = g[col].to_numpy(dtype=float)
            peers[care]["metrics"][col] = {
                "q": [round(float(x), 4) for x in np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])],
                "hist": _hist(v, log)}

    briefs = {}
    bpath = TABLES_DIR / "agents" / "briefs.json"
    if bpath.exists():
        for b in json.loads(bpath.read_text()):
            if b["condition"] == "with retrieval":
                briefs[b["brief"]["provider"]] = b
    retr = HybridRetriever()

    providers = []
    for p in order:
        row = df.loc[p]
        care = _care(row)
        g = dev[dev.care == care]
        # "higher than X% of similar providers": strictly lower peers only, so a value tied
        # at a cap (e.g. the 35-day longest stay) is not reported as higher than everyone
        vals = [round(float(row[col]), 4) for col, *_ in METRICS]
        pcts = [round(float((g[col] < row[col]).mean() * 100), 1) for col, *_ in METRICS]
        s = shap.loc[p]
        top = s.reindex(s.abs().sort_values(ascending=False).index)[:5]
        # compact rows [feature, raises rank (1/0), value, % of similar providers below]
        drivers = [[f, int(v > 0), round(float(row[f]), 4), round(float((g[f] < row[f]).mean() * 100), 1)]
                   for f, v in top.items()]
        # what stands out once size is set aside: the size-free model's strongest upward factors
        sn = shap_ns.loc[p]
        beyond = [[f, round(float(row[f]), 4), round(float((g[f] < row[f]).mean() * 100), 1)]
                  for f, v in sn.sort_values(ascending=False).items() if v > 0][:3]
        in_queue = p in aq.index
        entry = {"id": p, "rank": int(rank[p]), "size_free_rank": int(size_free_rank[p]),
                 "care": care, "in_queue": in_queue, "unusual_without_size": None,
                 "v": vals, "pct": pcts, "drivers": drivers, "beyond": beyond}
        if in_queue:
            q = aq.loc[p]
            entry.update(action=q.action, gate=int(q.gate[0]), position=int(q.position), votes=int(q.votes),
                         of_models=int(q.of_models), unusual_without_size=bool(q.votes >= VOTE_AUDIT),
                         rule_reason=q.reason if q.gate.startswith("1") else "",
                         recommendation_check="borderline" if q.reason == "borderline" else None,
                         duplicate_claims=int(q.duplicate_claims), overlapping_stays=int(q.overlapping_stays),
                         lane=("borderline" if q.reason == "borderline" else
                               "audit" if q.action == "priority_audit" else "desk"),
                         batch=(int(q.position) - 1) // BATCH_SIZE + 1,
                         group=(" and ".join((["Duplicate claims"] if q.duplicate_claims and q.gate.startswith("1") else [])
                                             + (["Overlapping stays"] if q.overlapping_stays >= 2 and q.gate.startswith("1")
                                                else [])) if q.gate.startswith("1") else
                                "Possibly unusual billing" if q.reason == "borderline" else
                                "Unusual billing" if q.votes >= VOTE_AUDIT else "Mainly provider size"))
            # why the case is here, in the investigator's words: the rule finding, or what drives the rank
            entry["reason_label"] = entry["rule_reason"] if entry["gate"] == 1 else entry["group"]
        if p in briefs:
            br = briefs[p]["brief"]
            entry["agent_suggested_action"] = br["recommended_action"]
            entry["patterns"] = [{"id": t["chunk_id"], "title": retr.by_id(t["chunk_id"])["title"],
                                  "text": _pattern_text(retr.by_id(t["chunk_id"]))}
                                 for t in br["typologies"] if retr.by_id(t["chunk_id"])]
            entry["pattern_group"] = entry["patterns"][0]["title"] if entry["patterns"] else "No pattern matched"
            entry["case_summary"] = br["summary"]
            entry["case_uncertainty"] = br["uncertainty"]
        providers.append(entry)

    test_res = pd.read_csv(TABLES_DIR / "test_results.csv").set_index("model")
    queue = sorted((p for p in providers if p["in_queue"]), key=lambda p: p["position"])
    overview = {
        "providers_scored": len(test), "queue_size": len(queue), "risk_queue_size": RISK_QUEUE_SIZE,
        "gate1_rule_check": sum(p["gate"] == 1 for p in queue), "gate2_risk_ranking": sum(p["gate"] == 2 for p in queue),
        "gate1_reasons": {"duplicate claims": sum(p["gate"] == 1 and p["duplicate_claims"] > 0 for p in queue),
                          "overlapping hospital stays": sum(p["gate"] == 1 and p["overlapping_stays"] >= 2 for p in queue)},
        "vote_models": int(aq.of_models.iloc[0]), "vote_audit": VOTE_AUDIT, "batch_size": BATCH_SIZE,
        "actions": {a: sum(p.get("action") == a for p in queue) for a in ("priority_audit", "desk_review", "monitor")},
        "queue_unusual_without_size": sum(bool(p["unusual_without_size"]) for p in queue),
        "queue_care": pd.Series([p["care"] for p in queue]).value_counts().to_dict(),
        "track_record": {
            "flagged_in_test": int(scores["y"].sum()),
            "top100_previously_flagged": int(round(test_res.loc["lightgbm", "prec_k100"] * 100)),
            "top100_payments_alone": int(round(test_res.loc["size_only:total_reimb", "prec_k100"] * 100)),
        },
        "period": {"from": "2008-11-27", "to": "2009-12-31"},
        "peer_counts": {c: peers[c]["n"] for c in peers},
    }
    used = sorted({d[0] for p in providers for d in p["drivers"]} | {b[0] for p in providers for b in p["beyond"]})
    metric_keys = {c for c, *_ in METRICS}
    for care, g in dev.groupby("care"):
        peers[care]["driver_q"] = {f: [round(float(x), 4) for x in np.quantile(g[f].to_numpy(dtype=float),
                                                                              [0.1, 0.25, 0.5, 0.75, 0.9])]
                                   for f in used if f not in metric_keys}
    overview["recommendations_to_check"] = sum(bool(p.get("recommendation_check")) for p in queue)
    overview["lanes"] = {k: sum(p["lane"] == k for p in queue) for k in ("audit", "borderline", "desk")}
    overview["batches"] = max(p["batch"] for p in queue)
    overview["agent_agrees"] = sum(p.get("agent_suggested_action") == p["action"] for p in queue)
    overview["actions_unusual_beyond_size"] = {
        a: sum(p.get("action") == a and bool(p["unusual_without_size"]) for p in queue)
        for a in ("priority_audit", "desk_review", "monitor")}
    return {"overview": overview, "metrics": [{"key": c, "label": l, "unit": u, "log": lg, "help": h}
                                              for c, l, u, lg, h in METRICS],
            "driver_names": {f: {"label": FEATURE_NAMES[f][0], "unit": FEATURE_NAMES[f][1],
                                 "size": f in SIZE_FEATURES,
                                 "log": FEATURE_NAMES[f][1] in ("usd", "count")} for f in used},
            "peers": peers, "providers": providers}
