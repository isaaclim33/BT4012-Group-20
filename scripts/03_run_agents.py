"""Step 3: agent experiments.

  RETRIEVAL  retrieval quality of the playbook search (no LLM needed)
  CLEAN      Cleaning agent's plan (one call per raw file, every column covered, 3 runs) scored against
             the analyst's own cleaning decisions in data.py; the pipeline itself uses data.py
  LEAK_AGENT        Leakage agent on a benchmark of planted leaks
  AGENT_MODEL         agent-assisted pipeline vs fixed pipeline on CV repeats 2-5 and the test set
             (3 runs; every ablation repeated per run)
  QUEUE      two-gate audit selection: the review queue and each action, decided by code (no LLM)
  BRIEFS         Case-file briefs for every queued provider, with and without retrieval (needs QUEUE)
  DASHBOARD        Dashboard agent -> interactive provider review dashboard (needs QUEUE and BRIEFS)

Usage:
  python scripts/03_run_agents.py [STEPS...]            # live: needs GEMINI_API_KEY
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
from fraudrank.agents.case_file import completeness, fill_gaps  # noqa: E402
from fraudrank.agents.context import critic_context, experiment_context  # noqa: E402
from fraudrank.agents.feature_brief import feature_brief  # noqa: E402
from fraudrank.agents.schemas import FeatureProposal  # noqa: E402
from fraudrank.agents.llm import LLMClient, LLMError  # noqa: E402
from fraudrank.agents.rag import HybridRetriever, evaluate_retrieval  # noqa: E402
from fraudrank.agents.selection import choose_config, fold_test  # noqa: E402
from fraudrank.config import CACHE_DIR, RAW_DIR, TABLES_DIR, TRAIN_FILES  # noqa: E402
from fraudrank.metrics import bootstrap_ci, evaluate, paired_bootstrap_diff  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("steps", nargs="*")
ap.add_argument("--replay", action="store_true")
ap.add_argument("--model", default=None)
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--max-rounds", type=int, default=3)
ap.add_argument("--agent-rows", default="old,defs,one",
                help="AGENT_MODEL rows: old (three agents, no definitions, cached), defs, one")
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
def raw_profiles() -> dict[str, tuple[str, dict[str, str]]]:
    """Per raw file: a header line and one profile line per column."""
    out = {}
    for key, fname in TRAIN_FILES.items():
        d = pd.read_csv(RAW_DIR / fname, nrows=200000)
        lines = {}
        for col in d.columns:
            s = d[col]
            top = s.value_counts(dropna=True).head(4).to_dict()
            lines[col] = (f"  {col}: dtype={s.dtype}, missing={s.isna().mean():.1%}, distinct={s.nunique()}, "
                          f"top={top}")
        out[key] = (f"FILE {key} ({fname}), {len(d)} rows sampled", lines)
    return out


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


def clean_file(header: str, lines: dict[str, str], run_tag: str) -> tuple[list, list[str], int]:
    """One file's plan. Coverage is enforced by code: columns left out get one more call."""
    plan = A.cleaning_agent(llm, header + "\n" + "\n".join(lines.values()), list(lines), run_tag)
    steps = [s for s in plan.steps if s.column in lines]
    off_profile = len(plan.steps) - len(steps)
    missing = [c for c in lines if c not in {s.column for s in steps}]
    if missing:
        print(f"    {header.split()[1]}: {len(missing)} column(s) left out, asking again: {missing[:6]}")
        plan = A.cleaning_agent(llm, header + "\n" + "\n".join(lines[c] for c in missing), missing, run_tag)
        steps += [s for s in plan.steps if s.column in missing]
        missing = [c for c in lines if c not in {s.column for s in steps}]
    return steps, missing, off_profile


def CLEAN():
    profiles = raw_profiles()
    n_cols = sum(len(lines) for _, lines in profiles.values())
    rows = []
    for r in range(1, args.runs + 1):
        run_tag = f"(Independent run {r} of {args.runs}.)"
        steps, uncovered, off_profile = [], {}, 0
        for key, (header, lines) in profiles.items():
            got, missing, off = clean_file(header, lines, run_tag)
            steps += [(key, s) for s in got]
            uncovered[key], off_profile = missing, off_profile + off
        plan = [s for _, s in steps]
        hits = {k: bool(f(plan)) for k, f in REFERENCE.items()}
        errors = [s.model_dump() for s in plan if s.action == "impute" and any(t in s.column for t in STRUCTURAL)]
        rows.append({"run": r, "caught": sum(hits.values()), "reference_items": len(hits), **hits,
                     "structural_imputations_proposed": len(errors), "steps": len(plan),
                     "columns": n_cols, "columns_covered": n_cols - sum(map(len, uncovered.values())),
                     "steps_off_profile": off_profile})
        save(f"clean_plan_run{r}.json", {"steps": [{"file": k, **s.model_dump()} for k, s in steps],
                                         "uncovered": uncovered, "structural_imputations": errors})
        print(f"  run {r}: caught {sum(hits.values())}/{len(hits)}, "
              f"{n_cols - sum(map(len, uncovered.values()))}/{n_cols} columns covered, "
              f"missed {[k for k, v in hits.items() if not v]}")
    runs = pd.DataFrame(rows)
    save("clean_runs.csv", runs)
    save("clean_score.json", {"runs": len(runs), "caught_per_run": runs.caught.tolist(),
                              "caught_mean": runs.caught.mean(), "reference_items": len(REFERENCE),
                              "items_caught_share": {k: runs[k].mean() for k in REFERENCE},
                              "structural_imputations_proposed": int(runs.structural_imputations_proposed.sum()),
                              "columns": n_cols, "columns_covered_min": int(runs.columns_covered.min())})


# ------------------------------------------------------------------ LEAK_AGENT
def LEAK_AGENT():
    rows = []
    specs, truth = leak_bench.benchmark()
    for r in range(args.runs):
        review = A.leakage_agent(llm, specs, "stratified provider-level split; 80% development, 20% test "
                                 "opened once; 5x5 repeated stratified CV on development",
                                 run_tag=f"(Independent review {r + 1} of {args.runs}.)")
        sc = leak_bench.score(review, truth)
        rows.append({"run": r + 1, **sc, "split_ok": review.split_ok})
        save(f"leak_agent_review_run{r + 1}.json", review.model_dump())
    save("leak_agent_bench.csv", pd.DataFrame(rows))
    print(pd.DataFrame(rows).to_string(index=False))


# ------------------------------------------------------------------ AGENT_MODEL
# The loop chooses configurations on the first CV repeat (folds 1-5, previews on folds
# 1-3), and tuning also used that repeat. Both pipelines are therefore compared on
# repeats 2-5 only, which no choice has seen.
EVAL_FOLDS = SPLITS["folds"][5:]
PREVIEW_FOLDS = REPEAT0[:3]


def _cv(cfg, cols, base, folds):
    res, _ = runner.cv_evaluate(X.make_model(cfg), cols, base, SPLITS, folds=folds)
    return res


def agent_loop(run: int, allow_agent_features: bool = True, tag: str = "", definitions: bool = False,
               gate: str = "enforce", design: str = "three_agents") -> dict:
    """design 'three_agents': Feature agent, Experiment agent, and a Critic that decides when to stop.
    design 'one_agent': one modelling agent proposes features and configurations; code stops the search
    after a round that does not change the chosen configuration.
    definitions: the feature-proposing agent sees each existing feature's definition, not only its name.
    gate 'enforce' drops duplicate features before the leakage review; 'report' only records them, which
    is the design the cached runs without definitions used, so they replay unchanged."""
    run_tag = f"(Independent run {run}{tag}.)"
    existing = feature_brief(FAM) if definitions else {fam: cols for fam, cols in FAM.items()}
    agent_cols, base, feat_log, prop, first = [], df, {}, None, None
    if design == "one_agent":
        ctx0 = experiment_context(FAM, [], [])
        ctx0["feature_sets"]["agent"] = "the features you propose, after the duplicate and leakage checks"
        first = A.combined_agent(llm, existing, ctx0, run_tag)
        prop = FeatureProposal(claim_aggregates=first.claim_aggregates, ratios=first.ratios)
    elif allow_agent_features:
        try:
            prop = A.feature_agent(llm, existing, run_tag, definitions)
        except LLMError as err:            # recorded, and the loop continues without agent features
            print(f"  feature agent failed: {err}")
            feat_log = {"error": str(err)[:300]}
    if prop is not None:
        feats = X.compute_agent_features(prop, df)
        kept, duplicates = X.duplicate_gate(feats, df, [c for cols in FAM.values() for c in cols], SPLITS["dev"])
        if duplicates:
            print(f"  duplicate gate ({gate}): {len(duplicates)} of {len(feats.columns)} proposed features add no information")
        candidates = kept if gate == "enforce" else list(feats.columns)
        specs = [X.describe_feature(f) for f in prop.claim_aggregates + prop.ratios if f.name in candidates]
        review = A.leakage_agent(llm, specs, "stratified provider-level split with 5-fold CV; features are "
                                 "computed from claims only") if specs or gate == "report" else None
        rejected = {v.feature.split(":")[0].strip() for v in review.verdicts if v.verdict != "ok"} if review else set()
        agent_cols = [c for c in candidates if c not in rejected]
        base = df.join(feats[candidates])
        feat_log = {"proposal": prop.model_dump(), "gate": gate, "duplicate_gate": duplicates, "after_gate": kept,
                    "candidates": candidates, "review": review.model_dump() if review else None,
                    "accepted": agent_cols, "rejected": sorted(rejected), "all_proposed": list(feats.columns)}
    history, runs, fold_details = [], {}, {}

    def strip_agent(cfg):
        if not allow_agent_features and cfg.feature_set.endswith("+agent"):
            return cfg.model_copy(update={"feature_set": cfg.feature_set.replace("+agent", "")})
        return cfg

    def execute(cfgs):
        for cfg in map(strip_agent, cfgs):
            rid = len(runs) + 1
            cols = X.feature_set(cfg, FAM, agent_cols)
            res = _cv(cfg, cols, base, REPEAT0)
            runs[rid] = cfg
            fold_details[rid] = list(res.pr_auc)
            history.append({"run_id": rid, **cfg.model_dump(exclude={"rationale"}), "n_features": len(cols),
                            "pr_auc_mean": round(res.pr_auc.mean(), 4), "pr_auc_sd": round(res.pr_auc.std(), 4),
                            "prec_k50_mean": round(res.prec_k50.mean(), 4), "roc_auc_mean": round(res.roc_auc.mean(), 4)})

    preview_log = []

    def preview_gate(proposals):
        # Every number compared here is a mean over the same 3 preview folds: completed runs
        # were scored on REPEAT0, whose first 3 folds are the preview folds.
        previews = []
        for cfg in proposals:
            c = strip_agent(cfg)
            res = _cv(c, X.feature_set(c, FAM, agent_cols), base, PREVIEW_FOLDS)
            previews.append({"cfg": cfg, "mean": float(res.pr_auc.mean()), "sd": float(res.pr_auc.std())})
        done = [float(np.mean(f[:len(PREVIEW_FOLDS)])) for f in fold_details.values()]
        floor = max([p["mean"] for p in previews] + done)
        survived = [p for p in previews if p["mean"] + p["sd"] >= floor]
        if not survived:
            survived = [max(previews, key=lambda p: p["mean"])]
        for p in previews:
            print(f"    preview gate {'kept' if p in survived else 'dropped'} "
                  f"{p['cfg'].model}/{p['cfg'].feature_set}/{p['cfg'].imbalance}: "
                  f"PR-AUC {p['mean']:.3f} ± {p['sd']:.3f} (floor {floor:.3f})")
        preview_log.append({"floor": round(floor, 4),
                            "previews": [{"model": p["cfg"].model, "feature_set": p["cfg"].feature_set,
                                          "imbalance": p["cfg"].imbalance, "mean": round(p["mean"], 4),
                                          "sd": round(p["sd"], 4), "kept": p in survived} for p in previews]})
        return [p["cfg"] for p in survived]

    if design == "one_agent":
        execute(preview_gate(first.proposals))
    else:
        ctx = experiment_context(FAM, agent_cols if allow_agent_features else [], history)
        execute(preview_gate(A.experiment_agent(llm, ctx, run_tag).proposals))
    first_round = choose_config(history, fold_details)
    selection, verdicts, rounds = first_round, [], 1
    if design == "one_agent":
        # same budget as the Critic's: at most max_rounds rounds of experiments in all
        for rnd in range(1, args.max_rounds):
            ctx = critic_context(history, args.max_rounds - rnd - 1, selection) | {"agent_features": len(agent_cols)}
            fu = A.combined_followup(llm, ctx, run_tag, tool_ctx={"fold_details": fold_details})
            verdicts.append(fu.model_dump())
            execute(preview_gate(fu.proposals))
            rounds += 1
            new = choose_config(history, fold_details)
            unchanged = new["chosen"] == selection["chosen"]
            selection = new
            if unchanged:
                break
    else:
        for rnd in range(args.max_rounds):
            rounds_left = args.max_rounds - rnd - 1
            v = A.critic_agent(llm, critic_context(history, rounds_left, selection), rounds_left, run_tag,
                               tool_ctx={"fold_details": fold_details})
            verdicts.append(v.model_dump())
            if v.stop or not v.next_experiments or rnd == args.max_rounds - 1:
                break
            execute(preview_gate(v.next_experiments))
            rounds += 1
            selection = choose_config(history, fold_details)
    print(f"  chosen run {selection['chosen']} (best mean: run {selection['best_mean_run']}), {rounds} round(s)")
    return {"runs": runs, "history": history, "verdicts": verdicts, "chosen": selection["chosen"],
            "selection": selection, "first_round_best": first_round["chosen"], "features": feat_log,
            "agent_cols": agent_cols, "base": base, "preview_log": preview_log, "experiment_rounds": rounds,
            "design": design, "definitions": definitions}


def _ci(col):
    return 1.96 * col.std() / np.sqrt(len(col))


# Rows of the AGENT_MODEL table. The first two agent rows measure the feature definitions (by the
# duplicate rate among proposals, counted before the gate; not by PR-AUC, because the rows also differ
# in whether the gate filters); the last two are the fold test (both enforce the gate).
AGENT_ROWS = {
    "old": ("three agents, no definitions", {"definitions": False, "gate": "report", "design": "three_agents", "tag": ""},
            "agent_model_run{r}_log.json"),
    "defs": ("three agents, with definitions", {"definitions": True, "gate": "enforce", "design": "three_agents",
                                                "tag": ", definitions"}, "agent_model_defs_run{r}_log.json"),
    "one": ("one agent, with definitions", {"definitions": True, "gate": "enforce", "design": "one_agent",
                                            "tag": ", one agent"}, "agent_model_one_agent_run{r}_log.json"),
}
def _llm_mark():
    """Start counting calls, cache hits and model versions for one row and run."""
    saved = set(llm.model_versions)
    llm.model_versions.clear()
    return llm.calls, llm.cache_hits, llm.budget_hits, saved


def _llm_since(mark) -> dict:
    calls, hits, budget, saved = mark
    out = {"api_calls": llm.calls - calls, "cache_hits": llm.cache_hits - hits,
           "tool_budget_reached": llm.budget_hits - budget,
           "model_versions": " ".join(sorted(llm.model_versions))}
    llm.model_versions |= saved
    return out


def _funnel(label: str, run: int, f: dict) -> dict:
    """Feature proposals by outcome. Duplicates are counted before the gate acts, so the count means the
    same thing whether the gate filters or only reports."""
    gate = f.get("duplicate_gate", {})
    return {"pipeline": label, "run": run, "gate": f.get("gate"), "proposed": len(f.get("all_proposed", [])),
            "near_constant": sum(v == "near-constant" for v in gate.values()),
            "duplicates": sum(v.startswith("duplicates") for v in gate.values()),
            "reviewed": len(f.get("candidates", [])), "leak_flagged": len(f.get("rejected", [])),
            "accepted": len(f.get("accepted", []))}


def AGENT_MODEL():
    fixed = pd.read_csv(TABLES_DIR / "cv_folds.csv").query("experiment == 'MODELS' and model == 'lightgbm'") \
        .sort_values("fold").iloc[len(REPEAT0):].reset_index(drop=True)
    test_scores = pd.read_pickle(CACHE_DIR / "test_scores.pkl")
    y_test = test_scores["y"].to_numpy()
    rows, summary, test_rows = [], [], []

    def full_eval(label, run, out, cfg_id, cols_override=None):
        cfg = out["runs"][cfg_id]
        cols = cols_override or X.feature_set(cfg, FAM, out["agent_cols"])
        res = _cv(cfg, cols, out["base"], EVAL_FOLDS)
        res["fold"] += len(REPEAT0)                                 # same fold numbering as cv_folds.csv
        d = pd.Series(res.pr_auc.to_numpy() - fixed.pr_auc.to_numpy())
        rows.append(res.assign(pipeline=label, run=run))
        summary.append({"pipeline": label, "run": run, "config": cfg.model_dump(exclude={"rationale"}),
                        "n_features": len(cols), "eval_folds": len(res),
                        "pr_auc_mean": res.pr_auc.mean(), "pr_auc_sd": res.pr_auc.std(),
                        "pr_auc_ci_lo": res.pr_auc.mean() - _ci(res.pr_auc),
                        "pr_auc_ci_hi": res.pr_auc.mean() + _ci(res.pr_auc),
                        "prec_k100_mean": res.prec_k100.mean(), "prec_k100_sd": res.prec_k100.std(),
                        "rec_k100_mean": res.rec_k100.mean(), "rec_k100_sd": res.rec_k100.std(),
                        "prec_k50_mean": res.prec_k50.mean(), "rec_k50_mean": res.rec_k50.mean(),
                        "diff_vs_fixed_mean": d.mean(), "diff_vs_fixed_sd": d.std(),
                        "diff_vs_fixed_ci_lo": d.mean() - _ci(d), "diff_vs_fixed_ci_hi": d.mean() + _ci(d),
                        "folds_better": int((d > 0).sum()),
                        "loop_rounds": len(out["verdicts"]), "configs_tried": len(out["runs"]),
                        "experiment_rounds": out.get("experiment_rounds")})
        print(f"  {label:32s} run {run}: PR-AUC {res.pr_auc.mean():.3f} "
              f"[{res.pr_auc.mean() - _ci(res.pr_auc):.3f}, {res.pr_auc.mean() + _ci(res.pr_auc):.3f}]  "
              f"Prec@100 {res.prec_k100.mean():.3f}  Rec@100 {res.rec_k100.mean():.3f}  "
              f"diff {d.mean():+.3f} ({int((d > 0).sum())}/{len(d)} folds)")

    test_scores_by = {}

    def test_eval(label, run, out):
        """Score the chosen configuration on the held-out test set once."""
        cfg = out["runs"][out["chosen"]]
        _, s, _, _ = runner.test_evaluate(X.make_model(cfg), X.feature_set(cfg, FAM, out["agent_cols"]),
                                          out["base"], SPLITS)
        s = s.reindex(test_scores.index).to_numpy()
        test_scores_by[(label, run)] = s
        m = evaluate(y_test, s)
        ci = {f"{k}_{b}": v for k in ("pr_auc", "prec_k100", "rec_k100")
              for b, v in zip(("lo", "hi"), bootstrap_ci(y_test, s, k))}
        d, lo, hi = paired_bootstrap_diff(y_test, s, test_scores["lightgbm"].to_numpy(), "pr_auc")
        test_rows.append({"pipeline": label, "run": run, "config": cfg.model_dump(exclude={"rationale"}),
                          **{k: m[k] for k in ("pr_auc", "prec_k50", "rec_k50", "prec_k100", "rec_k100")}, **ci,
                          "pr_auc_diff_vs_fixed": d, "pr_auc_diff_lo": lo, "pr_auc_diff_hi": hi})
        print(f"  TEST {label} run {run}: PR-AUC {m['pr_auc']:.3f} [{ci['pr_auc_lo']:.3f}, {ci['pr_auc_hi']:.3f}]  "
              f"Rec@100 {m['rec_k100']:.3f}  diff vs fixed {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    failures, funnel, llm_rows = [], [], []
    keys = [k.strip() for k in args.agent_rows.split(",")]
    for key in keys:
        label, kw, log_name = AGENT_ROWS[key]
        for r in range(1, args.runs + 1):
            mark = _llm_mark()
            try:
                out = agent_loop(r, **kw)
            except LLMError as err:
                print(f"  {label} run {r} failed: {err}")
                failures.append({"pipeline": label, "run": r, "error": str(err)[:300]})
                llm_rows.append({"pipeline": label, "run": r, **_llm_since(mark)})
                continue
            llm_rows.append({"pipeline": label, "run": r, **_llm_since(mark)})
            funnel.append(_funnel(label, r, out["features"]))
            save(log_name.format(r=r), {k: v for k, v in out.items() if k not in ("base", "runs")} |
                 {"runs": {k: v.model_dump() for k, v in out["runs"].items()}})
            full_eval(label, r, out, out["chosen"])
            test_eval(label, r, out)
            if key == "old":     # ablations belong to the original design only
                full_eval("ablation: no critic (round 1 only)", r, out, out["first_round_best"])
                cfg = out["runs"][out["chosen"]]
                if cfg.feature_set.endswith("+agent"):
                    cols = X.feature_set(cfg, FAM, out["features"]["candidates"])
                    full_eval("ablation: no leakage review", r, out, out["chosen"], cols)
    if "old" in keys:
        for r in range(1, args.runs + 1):
            mark = _llm_mark()
            try:
                out = agent_loop(r, allow_agent_features=False, tag=", no feature agent")
                full_eval("ablation: no feature agent", r, out, out["chosen"])
            except LLMError as err:
                failures.append({"pipeline": "ablation: no feature agent", "run": r, "error": str(err)[:300]})
            llm_rows.append({"pipeline": "ablation: no feature agent", "run": r, **_llm_since(mark)})
    # a run that hit a tool budget still finished; it is listed so partial answers are visible beside failures
    failures += [{"pipeline": row["pipeline"], "run": row["run"], "completed": True,
                  "error": f"tool budget reached {row['tool_budget_reached']} time(s); answered from the "
                           "tool results gathered"} for row in llm_rows if row["tool_budget_reached"]]
    save("agent_model_failures.json", failures)
    llm_df = pd.DataFrame(llm_rows)
    save("agent_model_llm_calls.csv", llm_df)
    versions = llm_df.groupby("pipeline").model_versions.apply(lambda s: " ".join(sorted(set(" ".join(s).split()))))
    print("model versions by row:\n" + versions.to_string())
    if versions[versions != ""].nunique() > 1:          # a row with no finished call has no version
        print("  WARNING: rows were answered by different model versions; state this as a limitation")
    if funnel:
        fun = pd.DataFrame(funnel)
        save("agent_model_feature_funnel.csv", fun)
        rates = fun.groupby("pipeline", sort=False)[["proposed", "near_constant", "duplicates", "accepted"]].sum()
        rates["duplicate_rate"] = rates.duplicates / rates.proposed
        print(rates.round(3).to_string())
    fixed_row = {"pipeline": "fixed pipeline (LightGBM)", "run": 0, "eval_folds": len(fixed),
                 "pr_auc_mean": fixed.pr_auc.mean(), "pr_auc_sd": fixed.pr_auc.std(),
                 "pr_auc_ci_lo": fixed.pr_auc.mean() - _ci(fixed.pr_auc),
                 "pr_auc_ci_hi": fixed.pr_auc.mean() + _ci(fixed.pr_auc),
                 "prec_k100_mean": fixed.prec_k100.mean(), "prec_k100_sd": fixed.prec_k100.std(),
                 "rec_k100_mean": fixed.rec_k100.mean(), "rec_k100_sd": fixed.rec_k100.std(),
                 "prec_k50_mean": fixed.prec_k50.mean(), "rec_k50_mean": fixed.rec_k50.mean(),
                 "diff_vs_fixed_mean": 0}
    summary_df = pd.DataFrame([fixed_row] + summary)
    save("agent_model_summary.csv", summary_df)
    save("agent_model_folds.csv", pd.concat(rows))
    # headline: every condition averaged over its runs, never the best run
    by_pipeline = summary_df.groupby("pipeline", sort=False).agg(
        runs=("run", "count"), pr_auc_mean=("pr_auc_mean", "mean"), pr_auc_min=("pr_auc_mean", "min"),
        pr_auc_max=("pr_auc_mean", "max"), diff_vs_fixed_mean=("diff_vs_fixed_mean", "mean"),
        folds_better_mean=("folds_better", "mean"), rec_k100_mean=("rec_k100_mean", "mean")).reset_index()
    save("agent_model_by_pipeline.csv", by_pipeline)
    print(by_pipeline.round(4).to_string(index=False))

    if not test_rows:
        return
    fixed_test = pd.read_csv(TABLES_DIR / "test_results.csv").query("model == 'lightgbm'").iloc[0]
    test_df = pd.DataFrame(test_rows)
    mean_rows = [{"pipeline": f"{label}, mean of runs", "run": len(g),
                  **g.drop(columns=["pipeline", "run", "config"]).mean().to_dict()}
                 for label, g in test_df.groupby("pipeline", sort=False)]
    fixed_test_row = {"pipeline": "fixed pipeline (LightGBM)", "run": 0,
                      **{k: fixed_test[k] for k in ("pr_auc", "prec_k50", "rec_k50", "prec_k100", "rec_k100",
                                                    "pr_auc_lo", "pr_auc_hi", "prec_k100_lo", "prec_k100_hi")}}
    save("agent_model_test.csv", pd.DataFrame([fixed_test_row] + mean_rows + test_rows))
    print(pd.DataFrame([fixed_test_row] + mean_rows).round(3).to_string(index=False))

    a, b = AGENT_ROWS["one"][0], AGENT_ROWS["defs"][0]
    folds_df = pd.concat(rows)
    if {a, b} <= set(folds_df.pipeline):
        ft = fold_test(folds_df, a, b)
        runs_both = sorted({r for (lab, r) in test_scores_by if lab == a} & {r for (lab, r) in test_scores_by if lab == b})
        ft["test_set_secondary"] = [dict(zip(("run", "diff", "lo", "hi"), (r, *paired_bootstrap_diff(
            y_test, test_scores_by[(a, r)], test_scores_by[(b, r)], "pr_auc")))) for r in runs_both]
        save("agent_model_fold_test.json", ft)
        print(f"fold test: one agent minus three agents {ft['diff_mean']:+.4f} "
              f"[{ft['ci_lo']:+.4f}, {ft['ci_hi']:+.4f}] over {ft['folds']} folds -> {ft['decision']}")


# ------------------------------------------------------------------ QUEUE (code only, no LLM)
def QUEUE():
    """Two-gate audit selection on the test set (audit_selection.py). Rule thresholds were set on
    development providers; BRIEFS and DASHBOARD read the queue this writes."""
    from fraudrank import audit_selection as AS
    from fraudrank import models as M
    from fraudrank.config import SEED, VOTE_SEEDS
    checks = AS.billing_checks(df.index)
    save("rule_check_dev.csv", AS.rule_check_dev(checks.loc[SPLITS["dev"]], df.loc[SPLITS["dev"], "y"]))
    scores = pd.read_pickle(CACHE_DIR / "test_scores.pkl")
    params = json.loads((TABLES_DIR / "tuned_params.json").read_text())["lightgbm"]["params"]
    no_size = [c for f, cols in FAM.items() if f != "size" for c in cols]
    size_free = {}
    for seed in VOTE_SEEDS:     # same tuned settings as lgbm:no_size; only the seed changes
        _, s, _, _ = runner.test_evaluate(lambda: M.lightgbm(params, seed=seed), no_size, df, SPLITS)
        size_free[seed] = s
    assert np.allclose(size_free[SEED].reindex(scores.index), scores["lgbm:no_size"]), "seed 42 must match TEST"
    pd.DataFrame(size_free).to_pickle(CACHE_DIR / "size_free_seed_scores.pkl")
    queue = AS.two_gate_selection(scores["lightgbm"], checks, AS.seed_votes(size_free))
    save("audit_queue.csv", queue)
    gate1 = AS.rule_check(checks) != ""
    save("queue_evaluation.csv", AS.queue_evaluation(scores["lightgbm"], scores["y"], gate1, queue))
    print(queue.drop(columns=["risk_score", "of_models"]).to_string(index=False))
    print(queue.groupby(["gate", "action", "reason"]).size().to_string())


def _queue() -> pd.DataFrame:
    path = AGENT_DIR / "audit_queue.csv"
    if not path.exists():
        sys.exit("No review queue yet: run the QUEUE step first (python scripts/03_run_agents.py QUEUE).")
    return pd.read_csv(path).fillna({"reason": ""})


# ------------------------------------------------------------------ BRIEFS
def BRIEFS():
    """Case briefs for every provider in the review queue (two-gate audit selection, script 02 QUEUE).
    The queue and each action are decided by code; the brief's own recommended action is recorded
    next to it as the agent's suggestion, so agreement can be measured."""
    from fraudrank.audit_selection import billing_exhibits
    scores = pd.read_pickle(CACHE_DIR / "test_scores.pkl")["lightgbm"]
    queue = _queue()
    shap = pd.read_pickle(CACHE_DIR / "shap_test_lightgbm.pkl")
    Xte = pd.read_pickle(CACHE_DIR / "X_test_lightgbm.pkl")
    retr = HybridRetriever()
    tool_ctx = {"df": df, "dev": SPLITS["dev"], "retriever": retr, "shap": shap}
    exhibits = billing_exhibits(list(queue[queue.gate.str.startswith("1")].provider))
    rows, briefs = [], []
    for cond, use_rag in [("with retrieval", True), ("without retrieval", False)]:
        for q in queue.itertuples():
            prov = q.provider
            ev = build_evidence(prov, q.risk_rank, scores[prov], Xte, shap, df, SPLITS["dev"], retr if use_rag else None)
            ev["billing_checks"] = {"duplicate_claims": q.duplicate_claims, "overlapping_hospital_stays": q.overlapping_stays}
            # why it is queued, but not the seed votes or the action code chose: the agent's suggested action
            # stays its own reading, so a disagreement with code means something
            if q.gate.startswith("1"):
                ev["queued_because"] = f"rule check: {q.reason}"
                ev["billing_exhibits"] = {k: [x["summary"] for x in exhibits[prov][k]] for k in ("duplicates", "overlaps")}
            else:
                ev["queued_because"] = "risk ranking: among the highest-ranked providers the rule check did not refer"
            t0 = (llm.input_tokens, llm.output_tokens)
            b0 = llm.budget_hits
            trace = []
            try:
                brief = A.case_file_agent(llm, ev, use_rag,
                                          tool_ctx=tool_ctx if use_rag else None, trace_out=trace)
                missing_first = completeness(brief, ev)
                missing = missing_first
                if missing_first:                      # one retry naming what is missing, then code fills the gaps
                    brief = A.case_file_agent(llm, ev, use_rag, tool_ctx=tool_ctx if use_rag else None,
                                              missing=missing_first)
                    missing = completeness(brief, ev)
                chk = check_brief(brief, ev)           # checks the agent's own text, before any fallback
                if missing:
                    brief = fill_gaps(brief, ev)
                chk.update(schema_valid=True, agrees_with_code=brief.recommended_action == q.action,
                           complete_first_attempt=not missing_first, complete_after_retry=not missing,
                           missing_first_attempt="; ".join(missing_first), filled_by_code="; ".join(missing))
                if use_rag:      # trajectory: which tools, on which provider, with what result
                    with_id = [t for t in trace if "provider_id" in t["input"]]
                    chk.update(tool_calls=len(trace),
                               verified_case_provider=any(t["tool"] in ("lookup_provider", "get_peer_stats")
                                                          and t["input"].get("provider_id") == prov for t in trace),
                               searched_playbook=any(t["tool"] == "search_playbook" for t in trace),
                               id_calls=len(with_id),
                               id_calls_correct=sum(t["input"]["provider_id"] == prov for t in with_id),
                               tool_errors=sum("error" in t["output"] for t in trace))
                briefs.append({"condition": cond, "evidence": ev, "brief": brief.model_dump()})
            except LLMError as e:
                chk = {"provider": prov, "schema_valid": False, "error": str(e)[:200]}
            chk.update(condition=cond, position=q.position, gate=q.gate, code_action=q.action, rank=q.risk_rank,
                       label=int(df.loc[prov, "y"]), tool_budget_reached=llm.budget_hits - b0,
                       input_tokens=llm.input_tokens - t0[0], output_tokens=llm.output_tokens - t0[1])
            rows.append(chk)
    res = pd.DataFrame(rows)
    save("briefs_checks.csv", res)
    save("briefs.json", briefs)
    agg = res.groupby("condition").agg(
        briefs=("provider", "count"), schema_valid=("schema_valid", "mean"),
        drivers=("drivers", "sum"), drivers_faithful=("drivers_faithful", "sum"),
        citations=("citations", "sum"), citations_exist=("citations_exist", "sum"),
        citations_supported=("citations_supported", "sum"), says_confirmed=("says_fraud_confirmed", "sum"),
        agrees_with_code=("agrees_with_code", "mean"),
        complete_first_attempt=("complete_first_attempt", "mean"), complete_after_retry=("complete_after_retry", "mean"),
        input_tokens=("input_tokens", "sum"), output_tokens=("output_tokens", "sum")).reset_index()
    save("briefs_summary.csv", agg)
    ok = res[res.schema_valid]
    save("briefs_completeness.csv", ok.groupby(["condition", "gate"]).agg(
        briefs=("provider", "count"), complete_first_attempt=("complete_first_attempt", "sum"),
        complete_after_retry=("complete_after_retry", "sum"),
        missing_driver_first=("missing_first_attempt", lambda s: int(s.str.contains("driver").sum())),
        missing_finding_first=("missing_first_attempt", lambda s: int(s.str.contains("finding").sum()))).reset_index())
    tr = res[(res.condition == "with retrieval") & res.schema_valid]
    if "tool_calls" in tr:   # trajectory metrics: tool recall and argument accuracy
        traj = {"briefs": len(tr), "tool_calls": int(tr.tool_calls.sum()),
                "tool_calls_per_brief_mean": float(tr.tool_calls.mean()),
                "verified_case_provider": int(tr.verified_case_provider.sum()),
                "searched_playbook": int(tr.searched_playbook.sum()),
                "provider_id_calls": int(tr.id_calls.sum()), "provider_id_calls_correct": int(tr.id_calls_correct.sum()),
                "tool_errors": int(tr.tool_errors.sum()),
                "tool_budget_reached": int(tr.tool_budget_reached.sum())}
        save("briefs_trajectory.json", traj)
        print("trajectory:", traj)
    save("briefs_action_agreement.csv", res.groupby(["condition", "code_action", "action"]).size()
         .rename("briefs").reset_index().rename(columns={"action": "agent_suggested_action"}))
    print(agg.to_string(index=False))


# ------------------------------------------------------------------ DASHBOARD
def DASHBOARD():
    """Dashboard agent: overview + one page per queued provider -> checks -> interactive HTML dashboard."""
    from fraudrank.agents import dashboard_agent as G
    from fraudrank.agents import dashboard_data as DD
    from fraudrank.agents.dashboard import render
    from fraudrank.agents.grounding import ground
    from fraudrank.agents.text_checks import reading_ease
    from fraudrank.config import RESULTS_DIR
    out_dir = RESULTS_DIR / "dashboard"
    out_dir.mkdir(exist_ok=True)
    _queue()
    data = DD.build(df, SPLITS)
    facts = G.overview_facts(data)
    queue = [p for p in data["providers"] if p["in_queue"]]
    by_id = {p["id"]: p for p in queue}
    # desk reviews get a lighter page in standard code wording; AI text only for full audits and borderline
    evidence = {p["id"]: G.provider_evidence(p, data) for p in queue if p["lane"] != "desk"}
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
                pages[pid], checks[pid] = pg, G.check_page(pg, ev, G.exhibit_lines(by_id[pid]))
            except LLMError as err:
                checks[pid] = {"passed": False, "error": str(err)[:300]}
            rows.append({"part": "provider page", "run": r, "provider": pid, **checks[pid]})
        ai, failed = G.assemble(ov, ov_chk, pages, checks)
        ai, grounding_rows = ground(data, ai)       # final gate: everything shown traces to case evidence
        cost = {"input_tokens": llm.input_tokens - t0[0], "output_tokens": llm.output_tokens - t0[1]}
        records.append({"run": r, "overview": ov, "overview_check": ov_chk, "pages": pages, "checks": checks,
                        "failed": failed, **cost})
        print(f"  run {r}: overview passed={ov_chk.get('passed')}  pages passed "
              f"{sum(c.get('passed', False) for c in checks.values())}/{len(evidence)}  tokens {cost}")
        if r == 1:
            # check results go to dashboard_checks.csv and dashboard_grounding.csv, not onto the page
            meta = {"generated": "2026-11-17", "cycle": "claims Nov 2008 to Dec 2009"}
            save("dashboard_grounding.csv", pd.DataFrame(grounding_rows))
            (out_dir / "provider_review_dashboard.html").write_text(render(data, ai, meta), encoding="utf-8")
    save("dashboard_checks.csv", pd.DataFrame(rows))
    save("dashboard_outputs.json", records)


STEPS = {"RETRIEVAL": RETRIEVAL, "CLEAN": CLEAN, "LEAK_AGENT": LEAK_AGENT, "AGENT_MODEL": AGENT_MODEL, "QUEUE": QUEUE, "BRIEFS": BRIEFS, "DASHBOARD": DASHBOARD}
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
