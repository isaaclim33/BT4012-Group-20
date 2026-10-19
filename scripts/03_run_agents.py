"""Step 3: agent experiments.

  RETRIEVAL  retrieval quality of the playbook search (no LLM needed)
  CLEAN      Cleaning agent's plan scored against a reference checklist
  E10        Leakage agent on a benchmark of planted leaks
  E4         agent-assisted pipeline vs fixed pipeline on identical folds (3 runs + ablations)
  E8         Case-file briefs for the top test providers, with and without retrieval
  E11        Dashboard agent -> interactive provider review dashboard (needs E8 first)

Usage:
  python scripts/03_run_agents.py [STEPS...]            # live: needs ANTHROPIC_API_KEY
  python scripts/03_run_agents.py --replay [STEPS...]   # reproduce from results/cache/llm
Requires steps 01 and 02 (TEST) to have run first.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudrank import runner  # noqa: E402
from fraudrank.agents import agents as A  # noqa: E402
from fraudrank.agents import executor as X  # noqa: E402
from fraudrank.agents import leak_bench  # noqa: E402
from fraudrank.agents.case_file import build_evidence, check_brief  # noqa: E402
from fraudrank.agents.llm import LLMClient, LLMError  # noqa: E402
from fraudrank.agents.rag import HybridRetriever, evaluate_retrieval  # noqa: E402
from fraudrank.config import CACHE_DIR, RAW_DIR, TABLES_DIR, TRAIN_FILES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("steps", nargs="*")
ap.add_argument("--replay", action="store_true")
ap.add_argument("--model", default=None)
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--max-rounds", type=int, default=3)
ap.add_argument("--top-k", type=int, default=20)
args = ap.parse_args()

llm = LLMClient(**({"model": args.model} if args.model else {}), mode="replay" if args.replay else "live")
df, FAM, SPLITS = runner.load_all()
REPEAT0 = SPLITS["folds"][:5]
AGENT_DIR = TABLES_DIR / "agents"
AGENT_DIR.mkdir(exist_ok=True)


def save(name, obj):
    path = AGENT_DIR / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        path.write_text(json.dumps(obj, indent=2, default=str))


# ------------------------------------------------------------------ retrieval
def RETRIEVAL():
    rows = evaluate_retrieval()
    save("retrieval_eval.csv", pd.DataFrame(rows))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


# ------------------------------------------------------------------ cleaning
def raw_profile() -> str:
    lines = []
    for key, fname in TRAIN_FILES.items():
        d = pd.read_csv(RAW_DIR / fname, nrows=200000)
        lines.append(f"FILE {key} ({fname}), {len(d)} rows sampled")
        for col in d.columns:
            s = d[col]
            top = s.value_counts(dropna=True).head(4).to_dict()
            lines.append(f"  {col}: dtype={s.dtype}, missing={s.isna().mean():.1%}, distinct={s.nunique()}, "
                         f"top={top}")
    return "\n".join(lines)


REFERENCE = {
    "recode chronic conditions 1/2": lambda st: any("ChronicCond" in s.column and s.action == "recode_binary" for s in st),
    "recode gender 1/2": lambda st: any(s.column == "Gender" and s.action == "recode_binary" for s in st),
    "recode renal 0/Y": lambda st: any("Renal" in s.column and s.action == "recode_binary" for s in st),
    "drop constant inpatient deductible": lambda st: any("Deductible" in s.column and s.action == "drop_column"
                                                         and "out" not in s.detail.lower() for s in st),
    "parse dates": lambda st: any(s.action == "parse_date" for s in st),
    "physician/procedure missing kept as indicator": lambda st: any(
        ("Physician" in s.column or "Procedure" in s.column) and s.action in ("missing_indicator", "keep") for s in st),
    "DOD missing means alive": lambda st: any(s.column == "DOD" and s.action in ("missing_indicator", "keep", "parse_date")
                                              for s in st),
}
STRUCTURAL = ("Physician", "Procedure", "Diagnosis", "DOD", "Admission", "Discharge")


def CLEAN():
    plan = A.cleaning_agent(llm, raw_profile())
    hits = {k: bool(f(plan.steps)) for k, f in REFERENCE.items()}
    errors = [s.model_dump() for s in plan.steps if s.action == "impute" and any(t in s.column for t in STRUCTURAL)]
    result = {"reference_items": len(hits), "caught": sum(hits.values()), "items": hits,
              "structural_imputations_proposed": len(errors), "errors": errors, "steps": len(plan.steps)}
    save("clean_plan.json", plan.model_dump())
    save("clean_score.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "errors"}, indent=1))


# ------------------------------------------------------------------ E10
def E10():
    rows = []
    specs, truth = leak_bench.benchmark()
    for r in range(args.runs):
        review = A.leakage_agent(llm, specs, "stratified provider-level split; 80% development, 20% test "
                                 "opened once; 5x5 repeated stratified CV on development",
                                 run_tag=f"(Independent review {r + 1} of {args.runs}.)")
        sc = leak_bench.score(review, truth)
        rows.append({"run": r + 1, **sc, "split_ok": review.split_ok})
        save(f"e10_review_run{r + 1}.json", review.model_dump())
    save("e10_leak_bench.csv", pd.DataFrame(rows))
    print(pd.DataFrame(rows).to_string(index=False))


# ------------------------------------------------------------------ E4
def _cv(cfg, cols, base, folds):
    res, _ = runner.cv_evaluate(X.make_model(cfg), cols, base, SPLITS, folds=folds)
    return res


def agent_loop(run: int, allow_agent_features: bool = True, tag: str = "") -> dict:
    run_tag = f"(Independent run {run}{tag}.)"
    existing = {fam: cols for fam, cols in FAM.items()}
    agent_cols, base, feat_log = [], df, {}
    if allow_agent_features:
        try:
            prop = A.feature_agent(llm, existing, run_tag)
        except LLMError as err:            # recorded, and the loop continues without agent features
            print(f"  feature agent failed: {err}")
            prop = None
            feat_log = {"error": str(err)[:300]}
    if allow_agent_features and prop is not None:
        feats = X.compute_agent_features(prop, df)
        specs = [X.describe_feature(f) for f in prop.claim_aggregates + prop.ratios]
        review = A.leakage_agent(llm, specs, "stratified provider-level split with 5-fold CV; features are "
                                 "computed from claims only")
        rejected = {v.feature.split(":")[0].strip() for v in review.verdicts if v.verdict != "ok"}
        agent_cols = [c for c in feats.columns if c not in rejected]
        base = df.join(feats)
        feat_log = {"proposal": prop.model_dump(), "review": review.model_dump(),
                    "accepted": agent_cols, "rejected": sorted(rejected), "all_proposed": list(feats.columns)}
    history, runs = [], {}

    def execute(cfgs):
        for cfg in cfgs:
            if not allow_agent_features and cfg.feature_set.endswith("+agent"):
                cfg = cfg.model_copy(update={"feature_set": cfg.feature_set.replace("+agent", "")})
            rid = len(runs) + 1
            res = _cv(cfg, X.feature_set(cfg, FAM, agent_cols), base, REPEAT0)
            runs[rid] = cfg
            history.append({"run_id": rid, **cfg.model_dump(exclude={"rationale"}),
                            "pr_auc_mean": round(res.pr_auc.mean(), 4), "pr_auc_sd": round(res.pr_auc.std(), 4),
                            "prec_k50_mean": round(res.prec_k50.mean(), 4), "roc_auc_mean": round(res.roc_auc.mean(), 4)})

    ctx = {"feature_sets": {"all": len(sum(FAM.values(), [])), "no_size": len(sum(FAM.values(), [])) - len(FAM["size"]),
                            "agent_features": agent_cols if allow_agent_features else "not available"},
           "families": {k: len(v) for k, v in FAM.items()}, "history": history}
    execute(A.experiment_agent(llm, ctx, run_tag).proposals)
    first_round_best = max(history, key=lambda h: h["pr_auc_mean"])["run_id"]
    verdicts = []
    for rnd in range(args.max_rounds):
        v = A.critic_agent(llm, {**ctx, "history": history}, args.max_rounds - rnd - 1, run_tag)
        verdicts.append(v.model_dump())
        if v.stop or not v.next_experiments or rnd == args.max_rounds - 1:
            break
        execute(v.next_experiments)
    chosen = verdicts[-1]["chosen_run"]
    if chosen not in runs:                      # an invalid id is recorded, then the best CV run is used
        chosen = max(history, key=lambda h: h["pr_auc_mean"])["run_id"]
    return {"runs": runs, "history": history, "verdicts": verdicts, "chosen": chosen,
            "first_round_best": first_round_best, "features": feat_log, "agent_cols": agent_cols, "base": base}


def E4():
    fixed = pd.read_csv(TABLES_DIR / "cv_folds.csv").query("experiment == 'E1' and model == 'lightgbm'") \
        .sort_values("fold").reset_index(drop=True)
    rows, summary = [], []

    def full_eval(label, run, out, cfg_id, cols_override=None):
        cfg = out["runs"][cfg_id]
        cols = cols_override or X.feature_set(cfg, FAM, out["agent_cols"])
        res = _cv(cfg, cols, out["base"], None)
        d = res.pr_auc.to_numpy() - fixed.pr_auc.to_numpy()
        rows.append(res.assign(pipeline=label, run=run))
        summary.append({"pipeline": label, "run": run, "config": cfg.model_dump(exclude={"rationale"}),
                        "n_features": len(cols), "pr_auc_mean": res.pr_auc.mean(), "pr_auc_sd": res.pr_auc.std(),
                        "prec_k100_mean": res.prec_k100.mean(), "diff_vs_fixed_mean": d.mean(),
                        "diff_vs_fixed_sd": d.std(), "folds_better": int((d > 0).sum()),
                        "loop_rounds": len(out["verdicts"]), "configs_tried": len(out["runs"])})
        print(f"  {label:32s} run {run}: PR-AUC {res.pr_auc.mean():.3f} ± {res.pr_auc.std():.3f}  "
              f"diff vs fixed {d.mean():+.3f} ({int((d > 0).sum())}/25 folds better)")

    failures = []
    for r in range(1, args.runs + 1):
        try:
            out = agent_loop(r)
        except LLMError as err:
            print(f"  run {r} failed: {err}")
            failures.append({"run": r, "error": str(err)[:300]})
            continue
        save(f"e4_run{r}_log.json", {k: v for k, v in out.items() if k not in ("base", "runs")} |
             {"runs": {k: v.model_dump() for k, v in out["runs"].items()}})
        full_eval("agent pipeline", r, out, out["chosen"])
        if r == 1:
            full_eval("ablation: no critic (best of round 1)", r, out, out["first_round_best"])
            cfg = out["runs"][out["chosen"]]
            if cfg.feature_set.endswith("+agent"):
                cols = X.feature_set(cfg, FAM, out["features"]["all_proposed"])
                full_eval("ablation: no leakage review", r, out, out["chosen"], cols)
    try:
        out = agent_loop(1, allow_agent_features=False, tag=", no feature agent")
        full_eval("ablation: no feature agent", 1, out, out["chosen"])
    except LLMError as err:
        failures.append({"run": "ablation: no feature agent", "error": str(err)[:300]})
    save("e4_failures.json", failures)
    fixed_row = {"pipeline": "fixed pipeline (E1 LightGBM)", "run": 0, "pr_auc_mean": fixed.pr_auc.mean(),
                 "pr_auc_sd": fixed.pr_auc.std(), "prec_k100_mean": fixed.prec_k100.mean(), "diff_vs_fixed_mean": 0}
    save("e4_summary.csv", pd.DataFrame([fixed_row] + summary))
    save("e4_folds.csv", pd.concat(rows))


# ------------------------------------------------------------------ E8
def E8():
    scores = pd.read_pickle(CACHE_DIR / "test_scores.pkl")["lightgbm"].sort_values(ascending=False)
    shap = pd.read_pickle(CACHE_DIR / "shap_test_lightgbm.pkl")
    Xte = pd.read_pickle(CACHE_DIR / "X_test_lightgbm.pkl")
    retr = HybridRetriever()
    rows, briefs = [], []
    for cond, use_rag in [("with retrieval", True), ("without retrieval", False)]:
        for rank, (prov, sc) in enumerate(scores.head(args.top_k).items(), start=1):
            ev = build_evidence(prov, rank, sc, Xte, shap, df, SPLITS["dev"], retr if use_rag else None)
            t0 = (llm.input_tokens, llm.output_tokens)
            try:
                brief = A.case_file_agent(llm, ev, use_rag)
                chk = check_brief(brief, ev)
                chk.update(schema_valid=True)
                briefs.append({"condition": cond, "evidence": ev, "brief": brief.model_dump()})
            except LLMError as e:
                chk = {"provider": prov, "schema_valid": False, "error": str(e)[:200]}
            chk.update(condition=cond, rank=rank, label=int(df.loc[prov, "y"]),
                       input_tokens=llm.input_tokens - t0[0], output_tokens=llm.output_tokens - t0[1])
            rows.append(chk)
    res = pd.DataFrame(rows)
    save("e8_checks.csv", res)
    save("e8_briefs.json", briefs)
    agg = res.groupby("condition").agg(
        briefs=("provider", "count"), schema_valid=("schema_valid", "mean"),
        drivers=("drivers", "sum"), drivers_faithful=("drivers_faithful", "sum"),
        citations=("citations", "sum"), citations_exist=("citations_exist", "sum"),
        citations_supported=("citations_supported", "sum"), says_confirmed=("says_fraud_confirmed", "sum"),
        input_tokens=("input_tokens", "sum"), output_tokens=("output_tokens", "sum")).reset_index()
    save("e8_summary.csv", agg)
    print(agg.to_string(index=False))


# ------------------------------------------------------------------ E11
def E11():
    """Dashboard agent: overview + one page per queued provider -> checks -> interactive HTML dashboard."""
    from fraudrank.agents import dashboard_agent as G
    from fraudrank.agents import dashboard_data as DD
    from fraudrank.agents.dashboard import render
    from fraudrank.agents.text_checks import reading_ease
    from fraudrank.config import RESULTS_DIR
    out_dir = RESULTS_DIR / "dashboard"
    out_dir.mkdir(exist_ok=True)
    data = DD.build(df, SPLITS)
    facts = G.overview_facts(data)
    queue = [p for p in data["providers"] if p["in_queue"]]
    evidence = {p["id"]: G.provider_evidence(p, data) for p in queue}
    notes = " ".join(p.get("case_summary", "") for p in queue)
    rows = [{"part": "case-file summaries (input)", "run": 0, "reading_ease": round(reading_ease(notes), 1),
             "words": len(notes.split())}]
    records = []
    for r in range(1, args.runs + 1):
        tag = f"(Independent run {r} of {args.runs}.)"
        t0 = (llm.input_tokens, llm.output_tokens)
        try:
            ov = A.dashboard_overview_agent(llm, facts, tag).model_dump()
            ov_chk = G.check_overview(ov, facts)
        except LLMError as err:
            ov, ov_chk = None, {"passed": False, "error": str(err)[:300]}
        rows.append({"part": "overview", "run": r, **ov_chk})
        pages, checks = {}, {}
        for pid, ev in evidence.items():
            try:
                pg = A.dashboard_provider_agent(llm, ev, tag).model_dump()
                pages[pid], checks[pid] = pg, G.check_page(pg, ev)
            except LLMError as err:
                checks[pid] = {"passed": False, "error": str(err)[:300]}
            rows.append({"part": "provider page", "run": r, "provider": pid, **checks[pid]})
        ai, failed = G.assemble(ov, ov_chk, pages, checks)
        cost = {"input_tokens": llm.input_tokens - t0[0], "output_tokens": llm.output_tokens - t0[1]}
        records.append({"run": r, "overview": ov, "overview_check": ov_chk, "pages": pages, "checks": checks,
                        "failed": failed, **cost})
        print(f"  run {r}: overview passed={ov_chk.get('passed')}  pages passed "
              f"{sum(c.get('passed', False) for c in checks.values())}/{len(evidence)}  tokens {cost}")
        if r == 1:
            n_long = len(ov_chk.get("too_long", [])) + sum(len(c.get("too_long", [])) for c in checks.values())
            meta = {"generated": "2026-10-18", "cycle": "claims Nov 2008 to Dec 2009", "checks_failed": failed,
                    "ai_note": f"Plain-language text written by AI agents ({llm.model}) and checked automatically; "
                               f"{len(evidence) - sum(1 for f in failed if f != 'home page')} of {len(evidence)} "
                               f"provider pages and {'the' if 'home page' not in failed else 'no'} home page passed"
                               + (f"; {n_long} passage(s) too long for their space show standard wording." if n_long
                                  else ".")}
            (out_dir / "provider_review_dashboard.html").write_text(render(data, ai, meta), encoding="utf-8")
    save("e11_checks.csv", pd.DataFrame(rows))
    save("e11_outputs.json", records)


STEPS = {"RETRIEVAL": RETRIEVAL, "CLEAN": CLEAN, "E10": E10, "E4": E4, "E8": E8, "E11": E11}
if __name__ == "__main__":
    for step in args.steps or list(STEPS):
        print(f"== {step}")
        t = time.time()
        STEPS[step]()
        print(f"== {step} done in {time.time() - t:.0f}s")
    usage_file = AGENT_DIR / "llm_usage.json"
    usage = json.loads(usage_file.read_text()) if usage_file.exists() else {}
    usage[" ".join(args.steps or list(STEPS))] = llm.usage()
    save("llm_usage.json", usage)
    print(llm.usage())
