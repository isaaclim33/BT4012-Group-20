"""Refresh checks, statistics and HTML using saved scores and agent text only.

Usage: python scripts/rebuild_offline.py
Writes results/dashboard/; the original experiment tables are read-only inputs.
Missing caches cause an error, never a training run or a provider API call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def forbid_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
        raise RuntimeError("Network access is disabled in rebuild_offline.py")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def validate_caches(df, splits, scores, seeds, X, shap, shap_ns):
    """Check cached predictions/explanations belong to the same provider table."""
    import numpy as np
    dev, test = set(splits["dev"]), set(splits["test"])
    if not df.index.is_unique or dev & test or dev | test != set(df.index):
        raise ValueError("Invalid provider split or duplicate provider IDs")
    for name, frame in [("scores", scores), ("seeds", seeds), ("features", X),
                        ("SHAP", shap), ("size-free SHAP", shap_ns)]:
        if not frame.index.is_unique or set(frame.index) != test:
            raise ValueError(f"{name} does not match the held-out provider IDs")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} contains non-finite values")
    if not scores.y.equals(df.loc[scores.index, "y"].astype(scores.y.dtype)):
        raise ValueError("Cached labels do not match the processed data")
    if list(X.columns) != list(shap.columns) or not set(shap_ns.columns) <= set(X.columns):
        raise ValueError("SHAP feature schema mismatch")
    if not np.allclose(X, df.loc[X.index, X.columns], rtol=0, atol=1e-10):
        raise ValueError("Cached model inputs differ from the processed features")
    from fraudrank.config import SEED, VOTE_SEEDS
    if set(seeds.columns) != set(VOTE_SEEDS):
        raise ValueError("Missing or unexpected seed prediction columns")
    if not np.allclose(seeds[SEED], scores.loc[seeds.index, "lgbm:no_size"]):
        raise ValueError("Seed predictions disagree with the saved size-free model")


def refresh_intervals(source, target, scores, cache, n):
    import numpy as np
    import pandas as pd
    from fraudrank.metrics import bootstrap_ci, paired_bootstrap_diff
    all_scores = scores.copy()
    deep = cache / "test_scores_deep.pkl"
    if deep.exists():
        extra = pd.read_pickle(deep)
        if not extra.index.is_unique or set(extra.index) != set(scores.index):
            raise ValueError("Deep scores have different provider IDs")
        if not np.array_equal(extra.y.reindex(scores.index), scores.y):
            raise ValueError("Deep scores have different labels")
        all_scores = all_scores.join(extra.drop(columns="y"))
    rows = pd.read_csv(source / "test_results.csv")
    unavailable = []
    for i, row in rows.iterrows():
        if row.model not in all_scores:
            unavailable.append(row.model)
            continue
        print(f"  intervals: {row.model}", flush=True)
        for metric in ("pr_auc", "prec_k100"):
            lo, hi = bootstrap_ci(all_scores.y, all_scores[row.model], metric, n=n)
            rows.loc[i, [metric + "_lo", metric + "_hi"]] = [lo, hi]
    rows.to_csv(target / "test_results.csv", index=False)
    path = source / "test_paired_diffs.csv"
    if path.exists():
        pairs = pd.read_csv(path)
        for i, row in pairs.iterrows():
            if row.a in all_scores and row.b in all_scores:
                d, lo, hi = paired_bootstrap_diff(all_scores.y, all_scores[row.a], all_scores[row.b], row.metric, n=n)
                pairs.loc[i, ["diff", "lo", "hi"]] = [d, lo, hi]
        pairs.to_csv(target / path.name, index=False)
    return unavailable


def refresh_fold_intervals(source, target, splits):
    import pandas as pd
    from scipy.stats import t
    from fraudrank.agents.selection import corrected_se, fold_test
    folds = pd.read_csv(source / "agents/agent_model_folds.csv")
    fixed = pd.read_csv(source / "cv_folds.csv").query("experiment == 'MODELS' and model == 'lightgbm'")
    fixed = fixed.set_index("fold").pr_auc
    from fraudrank.config import CV_SPLITS
    ratio = sum(len(f["val"]) / len(f["train"]) for f in splits["folds"][CV_SPLITS:]) / len(splits["folds"][CV_SPLITS:])
    ft = fold_test(folds, "one agent, with definitions", "three agents, with definitions", ratio)
    save_json(target / "agents/agent_model_fold_test.json", ft)
    summary = pd.read_csv(source / "agents/agent_model_summary.csv")
    for i, row in summary.iterrows():
        if row.pipeline == "fixed pipeline (LightGBM)":
            values = fixed.loc[sorted(folds.fold.unique())]
        else:
            values = folds[(folds.pipeline == row.pipeline) & (folds.run == row.run)].set_index("fold").pr_auc
        if len(values) < 2:
            raise ValueError(f"Missing saved folds for {row.pipeline}, run {row.run}")
        diff = values - fixed.reindex(values.index)
        for prefix, v in (("pr_auc", values), ("diff_vs_fixed", diff)):
            half = t.ppf(.975, len(v) - 1) * corrected_se(v, ratio)
            summary.loc[i, [prefix + "_ci_lo", prefix + "_ci_hi"]] = [v.mean() - half, v.mean() + half]
    summary.to_csv(target / "agents/agent_model_summary.csv", index=False)
    return ft


def refresh_briefs(source, target, df, splits, queue, X, shap, claims):
    from fraudrank.audit_selection import billing_exhibits
    from fraudrank.agents.case_file import build_evidence, check_brief, completeness
    from fraudrank.agents.rag import HybridRetriever
    from fraudrank.agents.schemas import CaseBrief
    from pydantic import ValidationError
    import pandas as pd
    raw = json.loads((source / "agents/briefs.json").read_text())
    retriever = HybridRetriever()
    exhibits = billing_exhibits(queue[queue.gate.str.startswith("1")].provider.tolist(), claims)
    by_id = queue.set_index("provider")
    kept, checks = [], []
    for record in raw:
        pid = record["evidence"]["provider"]
        if pid not in by_id.index:
            continue
        q = by_id.loc[pid]
        ev = build_evidence(pid, int(q.risk_rank), q.risk_score, X, shap, df, splits["dev"],
                            retriever if record["condition"] == "with retrieval" else None)
        ev["billing_checks"] = {"duplicate_claims": int(q.duplicate_claims),
                                "overlapping_hospital_stays": int(q.overlapping_stays)}
        ev["queued_because"] = (f"rule check: {q.reason}" if q.gate.startswith("1") else
                               "risk ranking: among the highest-ranked providers the rule check did not refer")
        if pid in exhibits:
            ev["billing_exhibits"] = {k: [x["summary"] for x in exhibits[pid][k]] for k in ("duplicates", "overlaps")}
        # Preserve text only when its evidence is unchanged; never treat old
        # analyst prose as authoritative evidence for a changed referral.
        same = ev == record["evidence"]
        try:
            brief = CaseBrief.model_validate(record["brief"])
            check = check_brief(brief, ev)
            accepted = same and check["passed"] and not completeness(brief, ev)
        except (ValidationError, KeyError, TypeError):
            accepted = False
        checks.append({"provider": pid, "condition": record["condition"],
                       "evidence_unchanged": same, "published": accepted})
        if accepted:
            kept.append({**record, "evidence": ev})
    save_json(target / "agents/briefs.json", kept)
    pd.DataFrame(checks).to_csv(target / "agents/briefs_revalidated.csv", index=False)
    return len(kept), len(raw)


def rebuild(output: Path, n: int):
    import pandas as pd
    from fraudrank import audit_selection as AS, data
    from fraudrank.config import CACHE_DIR, PROCESSED_DIR, TABLES_DIR, TRAIN_FILES, RAW_DIR, SEED
    from fraudrank.agents import dashboard_agent as G, dashboard_data as DD
    from fraudrank.agents.dashboard import render
    from fraudrank.agents.grounding import ground

    output = output.resolve()
    if output == ROOT or output in (TABLES_DIR, CACHE_DIR, PROCESSED_DIR) or any(
            output.is_relative_to(p) or p.is_relative_to(output) for p in (TABLES_DIR, CACHE_DIR, PROCESSED_DIR, RAW_DIR)):
        raise ValueError("Output must be separate from original inputs (default: results/dashboard)")
    required = [PROCESSED_DIR / p for p in ("provider_features.pkl", "splits.json")]
    required += [CACHE_DIR / p for p in ("test_scores.pkl", "size_free_seed_scores.pkl", "X_test_lightgbm.pkl",
                                        "shap_test_lightgbm.pkl", "shap_test_lgbm_no_size.pkl")]
    required += [TABLES_DIR / p for p in ("test_results.csv", "cv_folds.csv", "agents/audit_queue.csv",
                                         "agents/briefs.json", "agents/dashboard_outputs.json",
                                         "agents/agent_model_folds.csv", "agents/agent_model_summary.csv")]
    required += [RAW_DIR / TRAIN_FILES[k] for k in ("inpatient", "outpatient")]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Offline rebuild needs existing files; nothing will be trained: " + ", ".join(missing))
    optional = [CACHE_DIR / "test_scores_deep.pkl", TABLES_DIR / "test_paired_diffs.csv"]
    inputs = required + [p for p in optional if p.exists()]
    before = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    code_files = [*sorted((ROOT / "src/fraudrank").rglob("*.py")), Path(__file__).resolve(),
                  ROOT / "src/fraudrank/agents/dashboard_template.html"]
    code_before = {str(p.relative_to(ROOT)): digest(p) for p in code_files}
    df = pd.read_pickle(PROCESSED_DIR / "provider_features.pkl")
    splits = json.loads((PROCESSED_DIR / "splits.json").read_text())
    scores = pd.read_pickle(CACHE_DIR / "test_scores.pkl").sort_index()
    seeds = pd.read_pickle(CACHE_DIR / "size_free_seed_scores.pkl").sort_index()
    X = pd.read_pickle(CACHE_DIR / "X_test_lightgbm.pkl")
    shap = pd.read_pickle(CACHE_DIR / "shap_test_lightgbm.pkl")
    shap_ns = pd.read_pickle(CACHE_DIR / "shap_test_lgbm_no_size.pkl")
    validate_caches(df, splits, scores, seeds, X, shap, shap_ns)
    print("Validated cached predictions, labels, feature values and explanations.", flush=True)
    import tempfile
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="offline-build-", dir=output.parent) as staging:
        stage = Path(staging)
        tables = stage / "tables"
        agents = tables / "agents"
        agents.mkdir(parents=True)
        unavailable = refresh_intervals(TABLES_DIR, tables, scores, CACHE_DIR, n)
        ft = refresh_fold_intervals(TABLES_DIR, tables, splits)
        print("Rebuilding billing checks and queue from local claims and saved scores.", flush=True)
        claims = data.load_claims()
        from fraudrank.eda import run_eda
        run_eda(df, splits, tables, claims, verbose=False)
        checks = AS.billing_checks(df.index, claims)
        queue = AS.two_gate_selection(scores.lightgbm, checks, AS.seed_votes(dict(seeds.items())))
        queue.to_csv(agents / "audit_queue.csv", index=False)
        checks.to_csv(agents / "billing_checks.csv", index_label="provider")
        AS.rule_check_dev(checks.loc[splits["dev"]], df.loc[splits["dev"], "y"]).to_csv(agents / "rule_check_dev.csv", index=False)
        AS.queue_evaluation(scores.lightgbm, scores.y, AS.rule_check(checks) != "", queue).to_csv(
            agents / "queue_evaluation.csv", index=False)
        original = pd.read_csv(TABLES_DIR / "agents/audit_queue.csv").fillna({"reason": ""})
        old = original.set_index("provider")
        new = queue.set_index("provider")
        changed = set(old.index) ^ set(new.index)
        for p in set(old.index) & set(new.index):
            if any(old.loc[p, c] != new.loc[p, c] for c in ("gate", "action", "reason", "votes", "risk_rank", "duplicate_claims", "overlapping_stays")):
                changed.add(p)
        kept, total = refresh_briefs(TABLES_DIR, tables, df, splits, queue, X, shap, claims)
        dash = DD.build(df, splits, tables_dir=tables, claims=claims)
        saved = json.loads((TABLES_DIR / "agents/dashboard_outputs.json").read_text())
        record = next(r for r in saved if r["run"] == 1)
        facts = G.overview_facts(dash)
        ov = record.get("overview")
        oc = G.check_overview(ov, facts) if ov and not changed else {"passed": False}
        pages, page_checks = {}, {}
        for p in dash["providers"]:
            pid = p["id"]
            if not p["in_queue"] or pid in changed or pid not in record["pages"]:
                continue
            page = record["pages"][pid]
            pages[pid] = page
            page_checks[pid] = G.check_page(page, G.provider_evidence(p, dash), G.exhibit_lines(p))
        ai, failed = G.assemble(ov, oc, pages, page_checks)
        ai, rows = ground(dash, ai)
        pd.DataFrame(rows).to_csv(agents / "dashboard_grounding.csv", index=False)
        # Timestamps and identity come from the data, rather than a hard-coded run date.
        from datetime import datetime
        from zoneinfo import ZoneInfo
        meta = {"generated": datetime.now(ZoneInfo("Asia/Singapore")).date().isoformat(),
                "cycle": "claims Nov 2008 to Dec 2009", "offline": True}
        dashboard = stage / "provider_review_dashboard.html"
        dashboard.write_text(render(dash, ai, meta), encoding="utf-8")
        save_json(agents / "dashboard_revalidated.json", {"overview_check": oc, "page_checks": page_checks,
                                                         "failed_pages": failed, "changed_providers": sorted(changed)})
        after = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
        code_after = {str(p.relative_to(ROOT)): digest(p) for p in code_files}
        if before != after or code_before != code_after:
            raise RuntimeError("Inputs or code changed during the rebuild; output not published. Retry after the other run finishes.")
        manifest = {"api_calls": 0, "predictive_model_fits": 0, "network_disabled": True,
                    "bootstrap_samples": n, "seed": SEED, "input_sha256": before,
                    "source_sha256": code_before,
                    "queue_size": len(queue), "changed_queue_providers": sorted(changed),
                    "briefs_reused": kept, "briefs_checked": total, "agent_comparison": ft,
                    "unrefreshed_model_intervals_missing_predictions": unavailable,
                    "limitations": ["Agent test prediction vectors were not saved; their original bootstrap intervals were not recomputed.",
                                    "Original training fingerprints are unavailable; schema, IDs, labels and cached feature values were checked.",
                                    "SMOTE, patient-mix features and nested CV are unchanged.",
                                    "Original DOCX/PDF reports and original experiment tables are unchanged."],
                    "output_sha256": {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob("*")) if p.is_file()}}
        save_json(stage / "manifest.json", manifest)
        output.mkdir(parents=True, exist_ok=True)
        # Publish the manifest last, so it describes only a completed rebuild.
        for p in sorted(stage.rglob("*")):
            if p.is_file() and p.name != "manifest.json":
                dest = output / p.relative_to(stage)
                dest.parent.mkdir(parents=True, exist_ok=True)
                p.replace(dest)
        (stage / "manifest.json").replace(output / "manifest.json")
    print(f"Done: {output / 'provider_review_dashboard.html'}", flush=True)
    print(f"API calls: 0; model fits: 0; changed queue providers: {sorted(changed)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/dashboard")
    parser.add_argument("--bootstrap", type=int, default=2000, help="resamples (2000 for reported intervals)")
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error("--bootstrap must be at least 2")
    os.environ["FRAUDRANK_OFFLINE"] = "1"
    sys.addaudithook(forbid_network)
    rebuild(args.output, args.bootstrap)


if __name__ == "__main__":
    main()
