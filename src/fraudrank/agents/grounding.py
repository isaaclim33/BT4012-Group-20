"""Grounding checks: everything the dashboard states about a case must trace to that case's evidence.

Case evidence = the dict a provider page is written from (dashboard_agent.provider_evidence) plus the
billing-check counts behind the review queue. Two kinds of stated information are checked here:

* evidence lines: the one-line facts in the queue ('1 duplicate claim', '$4.2M paid'). Each item is
  built from the case evidence, then re-derived independently and its numbers checked; an item that
  fails is not shown;
* AI text after assembly (overview and provider pages): every number must appear in the evidence the
  text describes, written as digits (a number spelled out in words cannot be checked, so it fails), with
  no technical terms and no claim that fraud is proven. Text that fails is removed: one item of a list,
  or one entry of a per-factor field, at a time; elsewhere the whole field. The page shows standard code
  wording where nothing is left.

Every check is written to dashboard_grounding.csv. Code-computed numbers and charts need no check:
they are read from the same data as the evidence.
"""
from __future__ import annotations

from .dashboard_agent import overview_facts, provider_evidence
from .text_checks import allowed_numbers, check_text, numbers_in


def case_evidence(p: dict, data: dict) -> dict:
    ev = provider_evidence(p, data)
    ev["billing_checks"] = {"duplicate_claims": p.get("duplicate_claims", 0),
                            "overlapping_hospital_stays": p.get("overlapping_stays", 0)}
    return ev


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def evidence_items(ev: dict) -> list[dict]:
    """Facts for the one-line evidence: billing checks, total paid, and what stands out beyond size."""
    items = []
    b = ev["billing_checks"]
    if b["duplicate_claims"]:
        items.append({"kind": "duplicate_claims", "text": _plural(b["duplicate_claims"], "duplicate claim")})
    if b["overlapping_hospital_stays"]:
        items.append({"kind": "overlapping_stays",
                      "text": _plural(b["overlapping_hospital_stays"], "overlapping hospital stay")})
    items.append({"kind": "total_paid", "text": ev["key_measures"]["Total payments"].split(",")[0] + " paid"})
    if ev["beyond_size"]:
        f = ev["beyond_size"][0]
        items.append({"kind": "beyond_size", "feature": f["feature"], "text": f"{f['name']} {f['value']}"})
    return items


def _rederive(item: dict, ev: dict) -> str | None:
    """The text an item must have, recomputed from the evidence by a separate route."""
    b = ev["billing_checks"]
    if item["kind"] == "duplicate_claims":
        return _plural(b["duplicate_claims"], "duplicate claim") if b["duplicate_claims"] else None
    if item["kind"] == "overlapping_stays":
        n = b["overlapping_hospital_stays"]
        return _plural(n, "overlapping hospital stay") if n else None
    if item["kind"] == "total_paid":
        return ev["key_measures"]["Total payments"].split(",")[0] + " paid"
    if item["kind"] == "beyond_size":
        f = next((f for f in ev["beyond_size"] if f["feature"] == item.get("feature")), None)
        return f"{f['name']} {f['value']}" if f else None
    return None


def check_item(item: dict, ev: dict) -> dict:
    allowed = allowed_numbers(ev)
    unsupported = [n for n in numbers_in(item["text"]) if round(n, 4) not in allowed]
    expected = _rederive(item, ev)
    return {"kind": item["kind"], "text": item["text"], "matches_evidence": item["text"] == expected,
            "numbers_unsupported": len(unsupported), "passed": item["text"] == expected and not unsupported}


def _row(pid: str, field: str, value, source: dict) -> dict:
    chk = check_text({field: value}, source)
    return {"provider": pid, "part": "ai text", "field": field, "numbers_stated": chk["numbers_stated"],
            "numbers_unsupported": chk["numbers_unsupported"], "unsupported_values": chk["unsupported_values"],
            "spelled_numbers": chk["spelled_numbers"], "jargon_terms": chk["jargon_terms"],
            "says_proven_fraud": chk["says_proven_fraud"],
            "passed": not chk["numbers_unsupported"] and not chk["spelled_numbers"] and not chk["jargon_terms"]
            and not chk["says_proven_fraud"]}


def _gate(pid: str, fields: dict, source: dict) -> list[dict]:
    """Check every displayed AI field in place and remove what fails. Lists of sentences and per-factor
    entries are checked one at a time, so one bad item does not take the rest with it."""
    rows = []
    for name, value in list(fields.items()):
        if value is None or value == [] or value == {}:
            continue
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            items = [_row(pid, f"{name}[{i}]", v, source) for i, v in enumerate(value)]
            rows += items
            fields[name] = [v for v, r in zip(value, items) if r["passed"]] or None
        elif isinstance(value, dict):
            items = {k: _row(pid, f"{name}[{k}]", v, source) for k, v in value.items()}
            rows += items.values()
            fields[name] = {k: v for k, v in value.items() if items[k]["passed"]}
        else:
            r = _row(pid, name, value, source)
            rows.append(r)
            if not r["passed"]:
                fields[name] = None
    return rows


def ground(data: dict, ai: dict | None) -> tuple[dict, list[dict]]:
    """Attach checked evidence lines to queued providers and remove any AI field that fails.
    Returns the cleaned AI text and one row per check."""
    rows = []
    for p in data["providers"]:
        if not p["in_queue"]:
            continue
        ev = case_evidence(p, data)
        kept = []
        for item in evidence_items(ev):
            chk = check_item(item, ev)
            rows.append({"provider": p["id"], "part": "evidence line", "field": item["kind"], **chk})
            if chk["passed"]:
                kept.append(item["text"])
        p["evidence_line"] = kept
        page = (ai or {}).get("providers", {}).get(p["id"])
        if page:
            rows += _gate(p["id"], page, {k: v for k, v in ev.items() if k != "case_analyst_notes"})
    ov = (ai or {}).get("overview")
    if ov:
        rows += _gate("overview", ov, overview_facts(data))
    return ai, rows
