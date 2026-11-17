"""Plumbing test for the agent layer with a scripted fake LLM.

It checks that every agent's output parses, that the executor runs what agents
propose, and that the checks score briefs correctly. The fake responses are
fixtures for testing code paths only; they are never written to results/.
Run: python tests/test_agent_plumbing.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from fraudrank import runner  # noqa: E402
from fraudrank.agents import agents as A  # noqa: E402
from fraudrank.agents import executor as X  # noqa: E402
from fraudrank.agents import leak_bench  # noqa: E402
from fraudrank.agents.case_file import build_evidence, check_brief  # noqa: E402
from fraudrank.agents.llm import LLMClient  # noqa: E402
from fraudrank.agents.rag import HybridRetriever  # noqa: E402

FIX = {
    "cleaning_plan": {"steps": [
        {"column": "ChronicCond_Alzheimer", "action": "recode_binary", "detail": "1->1, 2->0", "reason": "coded 1/2"},
        {"column": "DeductibleAmtPaid", "action": "drop_column", "detail": "inpatient constant 1068", "reason": "no variance"},
        {"column": "OperatingPhysician", "action": "impute", "detail": "mode", "reason": "missing"}]},
    "feature_proposal": {"claim_aggregates": [
        {"name": "agent_los_std", "column": "los", "agg": "std", "scope": "ip", "rationale": "stay variability"}],
        "ratios": [{"name": "agent_ip_reimb_per_bene", "numerator": "ip_total_reimb", "denominator": "n_bene",
                    "rationale": "inpatient money per patient"}]},
    "leakage_review": {"verdicts": [{"feature": "agent_los_std", "verdict": "ok", "reason": "claims only"},
                                    {"feature": "agent_ip_reimb_per_bene", "verdict": "ok", "reason": "claims only"}],
                       "split_ok": True, "split_comment": "fine"},
    "experiment_batch": {"proposals": [
        {"model": "lightgbm", "feature_set": "all+agent", "imbalance": "class_weight",
         "params": {"num_leaves": 15, "n_estimators": 200}, "rationale": "a"},
        {"model": "logistic", "feature_set": "no_size", "imbalance": "class_weight", "params": {"C": 0.3},
         "rationale": "b"},
        {"model": "lightgbm", "feature_set": "no_size+agent", "imbalance": "none", "params": {}, "rationale": "c"}]},
    "critic_verdict": {"stop": True, "concerns": ["none"], "next_experiments": []},
}
FIX["combined_proposal"] = {**FIX["feature_proposal"], **FIX["experiment_batch"]}
FIX["followup_batch"] = {"proposals": FIX["experiment_batch"]["proposals"][:1]}


TITLES = {"CleaningPlan": "cleaning_plan", "FeatureProposal": "feature_proposal", "LeakageReview": "leakage_review",
          "ExperimentBatch": "experiment_batch", "CriticVerdict": "critic_verdict", "CaseBrief": "case_brief",
          "DashboardOverview": "dashboard_overview", "ProviderPage": "provider_page",
          "CombinedProposal": "combined_proposal", "FollowUpBatch": "followup_batch"}
CAVEATS = [{"caveat_id": c, "text": f"Rehearsal caution about {c.replace('_', ' ')}."}
           for c in ("suspicion_not_proof", "size_effect", "label_origin", "benchmark_data", "human_decision")]


def fake_overview(f):
    rec = f["recommendations"]
    return {"headline": f"Rehearsal: {f['queue_unusual_beyond_size']} of the {f['queue_size']} providers in the queue "
                        "bill unusually even when their size is set aside.",
            "summary": "Rehearsal: start with the full audits, then look again at the desk reviews flagged for a "
                       "second look.",
            "actions": [{"action": "priority_audit", "why": f"Rehearsal: all {rec.get('Full audit', 0)} bill unusually."}]
                       + ([{"action": "desk_review", "why": "Rehearsal: some of these also bill unusually."}]
                          if rec.get("Desk review") else []),                 # only when the queue has desk reviews
            "how_to_read": "Rehearsal: rank 1 looks most like providers flagged before. A rank is a reason to look. "
                           f"The other {f['providers_outside_queue']:,} are outside the queue.",
            "watch_out_for": ["Rehearsal: large hospitals rank high.", "Rehearsal: sicker patients cost more."],
            "caveats": CAVEATS, "glossary": [{"term": "Desk review", "meaning": "Rehearsal: a records check."}]}


def fake_page(ev):
    return {"provider": ev["provider"],
            "summary": f"Rehearsal: ranked {ev['rank']}; it stays high when size is set aside."
                       + (" Double-check the recommendation." if ev["recommendation_check"] else ""),
            "size_note": "Rehearsal: how it bills stands out, not only its size.",
            "beyond": [{"feature": f["feature"], "explanation": f"Rehearsal: {f['name']} is {f['value']}, {f['comparison']}."}
                       for f in ev["beyond_size"]],
            "reasons": [{"feature": f["feature"], "explanation": f"Rehearsal: {f['name']} {f['effect']}."}
                        for f in ev["overall_factors"]],
            "what_to_check": ["Rehearsal: pull the longest stays.", "Rehearsal: compare with similar hospitals.",
                              "Rehearsal: sample claims."],
            "patterns": [{"pattern_id": t["pattern_id"], "explanation": "Rehearsal: matches this pattern."}
                         for t in ev["patterns"]],
            "cautions": ["Rehearsal: a high rank is not proof.", f"Rehearsal: data covers {ev['claims_period']}."]}


def fake_raw(self, system, messages, schema, effort=None):
    tool_name = TITLES[schema["title"]]
    if tool_name in ("dashboard_overview", "provider_page"):
        import json
        data = json.loads(messages[0]["content"].split("<data>\n")[1].split("\n</data>")[0])
        return fake_overview(data) if tool_name == "dashboard_overview" else fake_page(data)
    if tool_name == "case_brief":
        import json
        ev = json.loads(messages[0]["content"].split("<data>\n")[1].split("\n</data>")[0])
        d = ev["drivers"][0]
        cites = [{"chunk_id": ev["playbook"][0]["chunk_id"], "why": "x"}] if ev.get("playbook") else []
        return {"provider": ev["provider"], "risk_rank": ev["risk_rank"], "summary": "Flagged as suspicious.",
                "drivers": [{k: d[k] for k in ("feature", "value", "peer_percentile", "shap")}],
                "typologies": cites, "recommended_action": "priority_audit",
                "action_reason": "Payments stand out.", "uncertainty": "label is suspicion"}
    return FIX[tool_name]


LLMClient._raw = fake_raw
llm = LLMClient(mode="replay")
df, FAM, SPLITS = runner.load_all()

plan = A.cleaning_agent(llm, "profile", ["ChronicCond_Alzheimer"])
assert plan.steps[0].action == "recode_binary"
prop = A.feature_agent(llm, FAM)
feats = X.compute_agent_features(prop, df)
assert list(feats.columns) == ["agent_los_std", "agent_ip_reimb_per_bene"] and feats.notna().all().all()
ALL_COLS = [c for cols in FAM.values() for c in cols]
probe = feats.assign(copy_of_los_max=df["los_max"], copy_again=df["los_max"] * 2, constant=1.0)
kept, dup = X.duplicate_gate(probe, df, ALL_COLS, SPLITS["dev"])
assert set(dup) == {"copy_of_los_max", "copy_again", "constant"} and "agent_ip_reimb_per_bene" in kept, dup
print(f"duplicate gate: ok (kept {kept}, rejected {sorted(dup)})")

# the Feature agent's prompt without definitions is unchanged, so the cached runs replay; with
# definitions it states the rule and sees every existing feature's definition
from fraudrank.agents.feature_brief import feature_brief  # noqa: E402

seen = []
LLMClient._raw = lambda self, system, messages, schema, effort=None: (seen.append((system, messages)),
                                                                      fake_raw(self, system, messages, schema, effort))[1]
A.feature_agent(llm, FAM, "(Independent run 1.)")
A.feature_agent(llm, feature_brief(FAM), "(Independent run 1.)", definitions=True)
LLMClient._raw = fake_raw
assert seen[0][0] == ("You are a fraud analytics feature engineer. " + A.TASK + " Existing provider features are grouped "
                      "into families below. Propose new provider-level features that describe HOW a provider bills rather "
                      "than HOW MUCH, because claim volume already explains most of the label. Use only the claim columns "
                      "and provider bases the schema allows. Never use the label. (Independent run 1.)")
assert A.DEFINITIONS_RULE in seen[1][0] and "n_claims / n_bene" in seen[1][1][0]["content"]
cp = A.combined_agent(llm, feature_brief(FAM), {"models_available": ["lightgbm"]})
assert len(cp.proposals) == 3 and len(cp.claim_aggregates + cp.ratios) == 2
assert len(A.combined_followup(llm, {}).proposals) == 1
print("feature definitions and one-agent design: ok (old prompt unchanged)")
batch = A.experiment_agent(llm, {})
for cfg in batch.proposals:
    cols = X.feature_set(cfg, FAM, list(feats.columns))
    res, _ = runner.cv_evaluate(X.make_model(cfg), cols, df.join(feats), SPLITS, folds=SPLITS["folds"][:2])
    print(f"executed {cfg.model}/{cfg.feature_set}: PR-AUC {res.pr_auc.mean():.3f}")
specs, truth = leak_bench.benchmark()
assert len(specs) == 15 and sum(truth.values()) == 5

# ------------------------------------------------------------------ tool tests
from fraudrank.agents.tools import get_fold_detail

fold_details = {1: [0.72, 0.73, 0.71, 0.74, 0.72], 2: [0.68, 0.75, 0.66, 0.78, 0.69]}
detail = get_fold_detail({"run_id": 1}, {"fold_details": fold_details})
assert detail["n_folds"] == 5 and "error" not in detail and len(detail["pr_auc_per_fold"]) == 5
detail_missing = get_fold_detail({"run_id": 99}, {"fold_details": fold_details})
assert "error" in detail_missing
print("fold detail tool: ok")

# ------------------------------------------------------------------ selection rule
from fraudrank.agents.selection import choose_config

hist = [{"run_id": 1, "model": "logistic", "imbalance": "none", "n_features": 63, "pr_auc_mean": 0.70},
        {"run_id": 2, "model": "random_forest", "imbalance": "none", "n_features": 63, "pr_auc_mean": 0.718},
        {"run_id": 3, "model": "lightgbm", "imbalance": "none", "n_features": 63, "pr_auc_mean": 0.722}]
folds = {1: [0.66, 0.70, 0.71, 0.70, 0.73],           # beaten on every fold
         2: [0.70, 0.73, 0.69, 0.75, 0.72],           # mixed: lightgbm is not reliably better
         3: [0.72, 0.72, 0.73, 0.73, 0.71]}
sel = choose_config(hist, folds)
assert sel["best_mean_run"] == 3 and sel["chosen"] == 2 and sel["kept_simpler"], sel
folds[2] = [0.69, 0.70, 0.71, 0.70, 0.69]             # now lightgbm wins every fold clearly
assert choose_config(hist, folds)["chosen"] == 3
assert choose_config(hist[:1], {1: folds[1]})["chosen"] == 1
print("selection rule: ok")

from fraudrank.agents.selection import fold_test  # noqa: E402

base_f = [0.70, 0.72, 0.74, 0.71, 0.73, 0.69, 0.75, 0.72]
fr = pd.DataFrame([{"pipeline": p, "run": r, "fold": i, "pr_auc": v - shift + 0.001 * r}
                   for p, shift in (("one", 0.0), ("three", 0.0)) for r in (1, 2, 3) for i, v in enumerate(base_f)])
fr.loc[(fr.pipeline == "one") & (fr.fold % 2 == 0), "pr_auc"] -= 0.002     # noise: small, mixed in sign
fr.loc[(fr.pipeline == "one") & (fr.fold % 2 == 1), "pr_auc"] += 0.001
assert fold_test(fr, "one", "three")["decision"] == "fold into one agent"
fr.loc[fr.pipeline == "one", "pr_auc"] -= 0.02                              # clearly worse on every fold
ft = fold_test(fr, "one", "three")
assert ft["decision"] == "keep separate agents" and ft["folds"] == len(base_f), ft
print("fold test rule: ok")

# ------------------------------------------------------------------ two-gate audit selection
from fraudrank import audit_selection as AS  # noqa: E402
from fraudrank.config import RISK_QUEUE_SIZE  # noqa: E402

ids = [f"P{i:03d}" for i in range(RISK_QUEUE_SIZE + 60)]
score = pd.Series([1 - i / 1000 for i in range(len(ids))], index=ids)
score["P011"] = score["P010"]                                     # a tie, broken by provider ID
chk = pd.DataFrame(0, index=ids, columns=["duplicate_claims", "overlapping_stays"])
low = ids[-1]
chk.loc[low, "duplicate_claims"] = 1                              # lowest rank, but clearly wrong billing
chk.loc["P002", "overlapping_stays"] = 2
chk.loc["P003", "overlapping_stays"] = 1                          # one overlap: a warning, not a referral
votes = pd.Series(10, index=ids)
votes.loc[["P004", "P005"]] = [5, 0]
q = AS.two_gate_selection(score, chk, votes)
qi = q.set_index("provider")
assert set(q[q.gate == "1_rule_check"].provider) == {"P002", low} and qi.loc[low, "action"] == "priority_audit"
assert (q.gate == "2_risk_ranking").sum() == RISK_QUEUE_SIZE and "P003" in set(q.provider)
assert qi.loc["P004", "reason"] == "borderline" and (qi.loc["P005", "action"], qi.loc["P005", "reason"]) == ("desk_review", "")
assert q.provider.iloc[-1] == low and q.risk_score.is_monotonic_decreasing       # the rule does not jump the queue
assert list(q.provider).index("P010") < list(q.provider).index("P011")              # tie broken by provider ID
assert q.equals(AS.two_gate_selection(score.sample(frac=1, random_state=1), chk, votes))   # order-independent
saved, seeds = ROOT / "results/tables/agents/audit_queue.csv", ROOT / "results/cache/size_free_seed_scores.pkl"
if saved.exists() and seeds.exists():                                               # real data: rebuild == saved
    real = AS.two_gate_selection(pd.read_pickle(ROOT / "results/cache/test_scores.pkl")["lightgbm"],
                                 AS.billing_checks(df.index), AS.seed_votes(dict(pd.read_pickle(seeds).items())))
    pd.testing.assert_frame_equal(real.fillna({"reason": ""}), pd.read_csv(saved).fillna({"reason": ""}),
                                  check_dtype=False, rtol=1e-9)
    print("two-gate audit selection: ok (real queue rebuilt identically)")
else:
    print("two-gate audit selection: ok")

cache = ROOT / "results" / "cache"
if (cache / "shap_test_lightgbm.pkl").exists():
    shap = pd.read_pickle(cache / "shap_test_lightgbm.pkl")
    Xte = pd.read_pickle(cache / "X_test_lightgbm.pkl")
    prov = shap.index[0]
    ev = build_evidence(prov, 1, 0.9, Xte, shap, df, SPLITS["dev"], HybridRetriever())
    chk = check_brief(A.case_file_agent(llm, ev, True), ev)
    assert chk["drivers_faithful"] == 1 and chk["citations_exist"] == 1, chk
    print("case-file check:", chk)
    # completeness: the top 3 drivers, and for a rule-check referral the billing finding in the summary
    from fraudrank.agents.case_file import completeness, fill_gaps
    ev1 = dict(ev, queued_because="rule check: 1 duplicate claim",
               billing_exhibits={"duplicates": ["$12,000 billed twice for BENE1 on 2009-01-17 (CLM1 and CLM2)"],
                                 "overlaps": []})
    b1 = A.case_file_agent(llm, ev1, True)                    # fake: one driver, summary without the finding
    miss = completeness(b1, ev1)
    assert sum(m.startswith("driver") for m in miss) == 2 and any("duplicate finding" in m for m in miss), miss
    filled = fill_gaps(b1, ev1)
    assert completeness(filled, ev1) == [] and [d.feature for d in filled.drivers][:3] != [] and \
        {d["feature"] for d in ev1["drivers"][:3]} <= {d.feature for d in filled.drivers}
    assert check_brief(filled, ev1)["drivers_faithful"] == len(filled.drivers)   # copied drivers are exact
    cites_id = b1.model_copy(update={"summary": "Claim CLM1 repeats CLM2 for the same patient."})
    assert not any("finding" in m for m in completeness(cites_id, ev1))          # a claim ID counts as the finding
    ev2 = dict(ev, drivers=ev["drivers"][:2])                                    # fewer than 3 drivers: all required
    assert [m for m in completeness(b1, ev2) if m.startswith("driver")] == [f"driver {ev['drivers'][1]['feature']}"]
    print("completeness gate: ok (missing drivers and finding caught; code fills them faithfully)")
if (ROOT / "results/tables/agents/briefs.json").exists() and (cache / "shap_test_lgbm_no_size.pkl").exists():
    import copy
    from fraudrank.agents import dashboard_agent as G
    from fraudrank.agents import dashboard_data as DD
    from fraudrank.agents.dashboard import render
    data = DD.build(df, SPLITS)
    facts = G.overview_facts(data)
    ov = A.dashboard_overview_agent(llm, facts).model_dump()
    oc = G.check_overview(ov, facts)
    assert oc["passed"], oc
    queue = [p for p in data["providers"] if p["in_queue"]]
    evs = {p["id"]: G.provider_evidence(p, data) for p in queue}
    pages = {pid: A.dashboard_provider_agent(llm, ev).model_dump() for pid, ev in evs.items()}
    checks = {pid: G.check_page(pages[pid], evs[pid]) for pid in pages}
    assert all(c["passed"] for c in checks.values()), [c for c in checks.values() if not c["passed"]][:1]
    pid = next(iter(pages))
    bad = {
        "jargon": dict(pages[pid], summary="The SHAP values and model score are high."),
        "invented number": dict(pages[pid], summary="It billed $9.87M last year."),
        "proven fraud": dict(pages[pid], size_note="This provider committed fraud."),
        "unknown factor": dict(pages[pid], beyond=[{"feature": "made_up", "explanation": "x"}]),
        "wrong provider": dict(pages[pid], provider="PRV00000"),
    }
    for name, page in bad.items():
        assert not G.check_page(page, evs[pid])["passed"], f"check missed: {name}"
    ov_stray = dict(ov, actions=ov["actions"] + [{"action": "monitor", "why": "The rest are outside the queue."}])
    sc = G.check_overview(ov_stray, facts)
    assert sc["passed"] and sc["actions_not_in_queue"] == ["monitor"]
    assert [a["action"] for a in G.assemble(ov_stray, sc, {}, {})[0]["overview"]["actions"]] == \
        [a["action"] for a in ov["actions"]]                          # stray note dropped, the rest kept
    ov_bad = copy.deepcopy(ov); ov_bad["caveats"] = ov_bad["caveats"][1:]
    assert G.check_overview(ov_bad, facts)["missing_caveats"]
    checks_bad = dict(checks); checks_bad[pid] = G.check_page(bad["jargon"], evs[pid])
    ai, failed = G.assemble(ov, oc, pages, checks_bad)
    assert failed == [pid] and pid not in ai["providers"], failed      # failed page falls back to system text
    # text that is only too long for its slot passes the content checks but is dropped slot by slot
    long_page = dict(pages[pid], summary=" ".join(["word"] * 70),
                     what_to_check=pages[pid]["what_to_check"] + [" ".join(["step"] * 30)])
    lc = G.check_page(long_page, evs[pid])
    assert lc["passed"] and lc["too_long"] == ["summary", "what_to_check[3]"], lc["too_long"]
    long_ov = dict(ov, headline=" ".join(["word"] * 30))
    ai_l, _ = G.assemble(long_ov, G.check_overview(long_ov, facts), dict(pages, **{pid: long_page}),
                         dict(checks, **{pid: lc}))
    assert ai_l["overview"]["headline"] is None and ai_l["overview"]["summary"] == ov["summary"]
    assert ai_l["providers"][pid]["summary"] is None and ai_l["providers"][pid]["what_to_check"] == pages[pid]["what_to_check"]
    # grounding gate: evidence lines re-derived from case evidence; ungrounded AI fields removed
    from fraudrank.agents.grounding import case_evidence, check_item, ground
    tampered, tp = copy.deepcopy(ai), next(iter(ai["providers"]))
    tampered["providers"][tp] = dict(tampered["providers"][tp], summary="It was paid $9.87M.", cautions=["SHAP is high."])
    grounded, grows = ground(data, tampered)
    assert grounded["providers"][tp]["summary"] is None and grounded["providers"][tp]["cautions"] is None
    spelled = copy.deepcopy(ai)                                     # a number in words cannot be checked: removed
    spelled["providers"][tp] = dict(spelled["providers"][tp], what_to_check=["Pull the claims worth eighty-nine thousand dollars.",
                                                                           "Sample two claims."])
    assert ground(data, spelled)[0]["providers"][tp]["what_to_check"] == ["Sample two claims."]
    assert all(r["passed"] for r in grows if r["part"] == "evidence line")
    q0 = next(p for p in data["providers"] if p["in_queue"])
    assert q0["evidence_line"] and not check_item({"kind": "total_paid", "text": "$9.99M paid"}, case_evidence(q0, data))["passed"]
    print("grounding: evidence lines", sum(r["part"] == "evidence line" for r in grows), "checked; invented number and jargon removed")
    html = render(data, ai, {"generated": "2026-10-18", "checks_failed": failed})
    assert "</script><script" not in html.split('id="dashboard-data"')[1].split("</script>")[0]
    assert "agent_view" not in html          # the agent's suggested action is recorded, never shown
    print("dashboard checks: overview", oc["passed"], "| pages passed", sum(c["passed"] for c in checks.values()),
          "of", len(checks), "| failures caught:", len(bad) + 1, "| over-long slots fall back: yes")
print("all plumbing checks passed")
