"""Dashboard agent: evidence for each page, checks on what the agent writes, and assembly.

Pipeline: Case-file briefs (BRIEFS) + code-computed dashboard data -> Dashboard agent
(one overview call + one call per queued provider) -> checks -> template.
Text that fails a check is dropped and the page falls back to standard system
wording, so unchecked text never reaches a reviewer.
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

from .schemas import OVERVIEW_SLOTS, PAGE_SLOTS, REQUIRED_CAVEATS
from .text_checks import check_text

ACTION_TEXT = {"priority_audit": "Full audit", "desk_review": "Desk review", "monitor": "Monitor"}


# ---------------------------------------------------------------- formatting (mirrors the template's JS)
# JavaScript rounds halves up (Math.round, toFixed); Python's round() rounds halves to even.
# These helpers copy JavaScript so the agent sees exactly the numbers the page shows.
def js_round(x: float) -> int:
    return math.floor(x + 0.5)


def js_fixed(x: float, digits: int) -> str:
    return str(Decimal(x).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def money(v: float) -> str:
    if v >= 1e6:
        return f"${js_fixed(v / 1e6, 1 if v >= 1e7 else 2)}M"
    if v >= 1e4:
        return f"${js_round(v / 1e3)}K"
    return f"${js_round(v):,}"


def fmt(v: float, unit: str) -> str:
    if unit == "usd":
        return money(v)
    if unit == "count":
        return f"{js_round(v):,}"
    if unit == "share":
        pct = v * 100
        return (js_fixed(pct, 1) if 0 < pct < 10 else str(js_round(pct))) + "%"
    if unit == "days":
        n = js_round(v) if abs(v - js_round(v)) < .05 else js_fixed(v, 1)
        return f"{n} day" if abs(v - 1) < 1e-9 else f"{n} days"
    if unit == "years":
        return f"{js_fixed(v, 1)} years"
    if unit == "flag":
        return "Yes" if v >= .5 else "No"
    return js_fixed(v, 2)


def peer_word(care: str) -> str:
    return "similar outpatient providers" if care == "Outpatient only" else "similar hospitals"


def compare(pct: float, care: str) -> str:
    if pct >= 99.5:
        return f"higher than almost all {peer_word(care)}"
    if pct <= .5:
        return f"lower than almost all {peer_word(care)}"
    if pct >= 50:
        return f"higher than {math.floor(pct)}% of {peer_word(care)}"
    return f"lower than {math.floor(100 - pct)}% of {peer_word(care)}"


# ---------------------------------------------------------------- evidence
def nice_date(iso: str) -> str:
    """'2008-11-27' -> '27 Nov 2008', as the page shows it."""
    from datetime import date
    d = date.fromisoformat(str(iso)[:10])
    return f"{d.day} {d.strftime('%b')} {d.year}"


def claims_period(data: dict) -> str:
    p = data["overview"]["period"]
    return f"{nice_date(p['from'])} to {nice_date(p['to'])}"


def overview_facts(data: dict) -> dict:
    o = data["overview"]
    return {
        "providers_ranked": o["providers_scored"], "queue_size": o["queue_size"],
        "providers_outside_queue": o["providers_scored"] - o["queue_size"],
        "recommendations": {ACTION_TEXT[a]: n for a, n in o["actions"].items() if n},
        "of_which_unusual_beyond_size": {ACTION_TEXT[a]: n for a, n in o["actions_unusual_beyond_size"].items()
                                         if o["actions"][a]},
        "queue_unusual_beyond_size": o["queue_unusual_without_size"],
        "how_the_queue_is_built": {
            "rule_check": o["gate1_rule_check"], "rule_check_reasons": o["gate1_reasons"],
            "rule_check_meaning": "billing anomalies that need verification (a claim repeating an earlier one, or hospital stays "
                                  "starting before the same patient's earlier stay ended) is referred whatever its rank",
            "risk_ranking": o["gate2_risk_ranking"],
            "risk_ranking_meaning": f"the {o['risk_queue_size']} highest-ranked of the remaining providers; a full "
                                    f"audit when at least {o['vote_audit']} of {o['vote_models']} models agree it is "
                                    "unusual beyond its size, otherwise a desk review"},
        "borderline_desk_reviews": o["recommendations_to_check"],
        "borderline_meaning": f"desk review because only some of the {o['vote_models']} models rank the provider "
                              "unusual beyond its size; a person decides whether a full audit is needed",
        "queue_types_of_care": o["queue_care"],
        "track_record": {"top_100_previously_flagged": o["track_record"]["top100_previously_flagged"],
                         "top_100_when_sorted_by_payments_alone": o["track_record"]["top100_payments_alone"],
                         "test_providers_previously_flagged": o["track_record"]["flagged_in_test"]},
        "claims_period": claims_period(data),
        "label_meaning": "past flags mean previously suspected, not proven; how they were assigned is undocumented",
    }


def provider_evidence(p: dict, data: dict) -> dict:
    names = data["driver_names"]
    N = data["overview"]["providers_scored"]
    mi = {m["key"]: i for i, m in enumerate(data["metrics"])}

    def factor(key, value, pct, raises=True):
        n = names[key]
        return {"feature": key, "name": n["label"], "value": fmt(value, n["unit"]),
                "comparison": compare(pct, p["care"]), "effect": "pushes rank up" if raises else "pulls rank down",
                "reflects_size": n["size"]}

    key_measures = {m["label"]: f'{fmt(p["v"][mi[m["key"]]], m["unit"])}, {compare(p["pct"][mi[m["key"]]], p["care"])}'
                    for m in data["metrics"] if m["key"] in ("total_reimb", "n_claims", "n_bene", "reimb_per_claim",
                                                             "claims_per_bene", "los_mean")}
    return {
        "provider": p["id"], "rank": f'{p["rank"]} of {N:,}', "type_of_care": p["care"],
        "claims_period": claims_period(data),
        "recommendation": ACTION_TEXT.get(p.get("action"), "Not in queue"),
        "rank_if_size_ignored": f'{p["size_free_rank"]} of {N:,}',
        "unusual_beyond_size": bool(p.get("unusual_without_size")),
        "selected_by": f'rule check: {p["rule_reason"]}' if p.get("gate") == 1 else "risk ranking",
        "model_agreement": f'{p["votes"]} of {p["of_models"]} models rank it unusual beyond its size'
                           if p.get("in_queue") else None,
        "recommendation_check": (f'borderline: only {p["votes"]} of {p["of_models"]} models rank it unusual beyond '
                                 "its size, so a person decides whether a full audit is needed"
                                 if p.get("recommendation_check") == "borderline" else None),
        "key_measures": key_measures,
        "beyond_size": [factor(b[0], b[1], b[2]) for b in p["beyond"]],
        "overall_factors": [factor(d[0], d[2], d[3], bool(d[1])) for d in p["drivers"]],
        "patterns": [{"pattern_id": t["id"], "title": t["title"], "description": t["text"]} for t in p.get("patterns", [])],
        "case_analyst_notes": {"summary": p.get("case_summary", ""), "uncertainty": p.get("case_uncertainty", "")},
    }


# ---------------------------------------------------------------- checks
def words(text: str) -> int:
    return len(str(text).split())


def too_long(out: dict, slots: dict) -> list[str]:
    """Slots whose text is over its word limit, as 'field' or 'field[i]' (i = list position or key)."""
    over = []
    for path, (_, limit) in slots.items():
        field, _, sub = path.partition(".")
        val = out.get(field)
        if isinstance(val, str):
            if words(val) > limit:
                over.append(field)
        elif isinstance(val, list):
            for i, item in enumerate(val):
                text = item.get(sub, "") if sub else item
                if words(text) > limit:
                    key = item.get("feature") or item.get("pattern_id") or item.get("action") if sub else None
                    over.append(f"{field}[{key if key is not None else i}]")
    return over


def check_overview(out: dict, facts: dict) -> dict:
    res = check_text(out, facts)
    caveats = {c["caveat_id"] for c in out.get("caveats", [])}
    allowed_actions = {a for a, n in {"priority_audit": facts["recommendations"].get("Full audit", 0),
                                      "desk_review": facts["recommendations"].get("Desk review", 0),
                                      "monitor": facts["recommendations"].get("Monitor", 0)}.items() if n}
    res["missing_caveats"] = sorted(set(REQUIRED_CAVEATS) - caveats)
    res["actions_not_in_queue"] = sorted({a["action"] for a in out.get("actions", [])} - allowed_actions)
    res["too_long"] = too_long(out, OVERVIEW_SLOTS)      # not a failure: only those slots fall back
    # a note on a recommendation nobody in the queue has is dropped on its own (see assemble), not the whole page
    res["passed"] = (not res["numbers_unsupported"] and not res["jargon_terms"] and not res["says_proven_fraud"]
                     and not res["missing_caveats"])
    return res


def exhibit_lines(p: dict) -> list[str]:
    """The billing exhibits a rule-check page shows (code-written), so text may quote their amounts and dates."""
    return [x["summary"] for kind in ("duplicates", "overlaps") for x in p.get("billing_exhibits", {}).get(kind, [])]


def check_page(out: dict, ev: dict, exhibits: list[str] | None = None) -> dict:
    # case-analyst notes are background, not a source the page may quote numbers from
    source = {k: v for k, v in ev.items() if k != "case_analyst_notes"}
    if exhibits:
        source["billing_exhibits"] = exhibits
    res = check_text(out, source)
    beyond_keys = {f["feature"] for f in ev["beyond_size"]}
    overall_keys = {f["feature"] for f in ev["overall_factors"]}
    pattern_ids = {t["pattern_id"] for t in ev["patterns"]}
    res["wrong_provider"] = out.get("provider") != ev["provider"]
    res["too_long"] = too_long(out, PAGE_SLOTS)          # not a failure: only those slots fall back
    res["unknown_factors"] = sorted(({f["feature"] for f in out.get("beyond", [])} - beyond_keys)
                                    | ({f["feature"] for f in out.get("reasons", [])} - overall_keys))
    res["unknown_patterns"] = sorted({t["pattern_id"] for t in out.get("patterns", [])} - pattern_ids)
    res["factors_explained"] = (len({f["feature"] for f in out.get("beyond", [])} & beyond_keys)
                                + len({f["feature"] for f in out.get("reasons", [])} & overall_keys))
    res["factors_expected"] = len(beyond_keys) + len(overall_keys)
    res["mentions_double_check"] = bool(ev["recommendation_check"]) and any(
        w in out.get("summary", "").lower() for w in ("borderline", "double-check", "double check", "second look",
                                                      "recheck", "check this recommendation", "reconsider"))
    res["passed"] = (not res["numbers_unsupported"] and not res["jargon_terms"] and not res["says_proven_fraud"]
                     and not res["wrong_provider"] and not res["unknown_factors"] and not res["unknown_patterns"])
    return res


# ---------------------------------------------------------------- assembly
def _fits(text: str, slots: dict, path: str) -> bool:
    return words(text) <= slots[path][1]


def assemble(overview: dict | None, ov_check: dict | None, pages: dict, checks: dict) -> tuple[dict, list[str]]:
    """Template-ready text.

    A part that fails a content check (numbers, jargon, proven-fraud wording, wrong provider, unknown factor)
    is left out entirely. Text that only runs over its slot's word limit, and a note on a recommendation no
    queued provider has, are dropped one by one. Either way the template shows standard system wording there.
    """
    failed = []
    ai = {"overview": None, "providers": {}}
    if overview is not None and ov_check and ov_check["passed"]:
        S, o = OVERVIEW_SLOTS, overview
        stray = set(ov_check.get("actions_not_in_queue", []))
        acts = [a for a in o["actions"] if a["action"] not in stray]
        acts = acts if acts and all(_fits(a["why"], S, "actions.why") for a in acts) else None
        watch = [w for w in o["watch_out_for"] if _fits(w, S, "watch_out_for")]
        cav = o["caveats"] if all(_fits(c["text"], S, "caveats.text") for c in o["caveats"]) else None
        ai["overview"] = {
            "headline": o["headline"] if _fits(o["headline"], S, "headline") else None,
            "summary": o["summary"] if _fits(o["summary"], S, "summary") else None,
            "actions": acts,
            "how_to_read": o["how_to_read"] if _fits(o["how_to_read"], S, "how_to_read") else None,
            "watch_out_for": watch or None,
            "caveats": cav,
            "glossary": [g for g in o["glossary"] if _fits(g["meaning"], S, "glossary.meaning")]}
    elif overview is not None:
        failed.append("home page")
    S = PAGE_SLOTS
    for pid, page in pages.items():
        if not checks[pid]["passed"]:
            failed.append(pid)
            continue
        checks_ = [c for c in page["what_to_check"] if _fits(c, S, "what_to_check")]
        cautions = [c for c in page["cautions"] if _fits(c, S, "cautions")]
        ai["providers"][pid] = {
            "summary": page["summary"] if _fits(page["summary"], S, "summary") else None,
            "size_note": page["size_note"] if _fits(page["size_note"], S, "size_note") else None,
            "beyond": {f["feature"]: f["explanation"] for f in page["beyond"]
                       if _fits(f["explanation"], S, "beyond.explanation")},
            "reasons": {f["feature"]: f["explanation"] for f in page["reasons"]
                        if _fits(f["explanation"], S, "reasons.explanation")},
            "what_to_check": checks_ or None,
            "patterns": {t["pattern_id"]: t["explanation"] for t in page["patterns"]
                         if _fits(t["explanation"], S, "patterns.explanation")},
            "cautions": cautions or None}
    return ai, failed
