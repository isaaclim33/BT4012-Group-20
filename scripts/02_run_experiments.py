"""Step 2: run every non-LLM experiment and save tables to results/.

Usage:  python scripts/02_run_experiments.py [SIZE MODELS ...]   (default: all)

Order matters: TUNE writes the chosen hyperparameters that the model experiments use, and TEST
opens the held-out test set once, after all choices are frozen.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imblearn.over_sampling import SMOTE  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402

from fraudrank import models as M  # noqa: E402
from fraudrank import runner  # noqa: E402
from fraudrank.config import CACHE_DIR, SEED, TABLES_DIR  # noqa: E402
from fraudrank.metrics import (bootstrap_ci, evaluate, k_list, paired_bootstrap_diff,  # noqa: E402
                               threshold_metrics)
from fraudrank.network import homophily_test  # noqa: E402

df, FAM, SPLITS = runner.load_all()
ALL = [c for cols in FAM.values() for c in cols]
NO_SIZE = [c for f, cols in FAM.items() if f != "size" for c in cols]
REPEAT0 = SPLITS["folds"][:5]           # first repeat, used only for tuning
PARAMS_FILE = TABLES_DIR / "tuned_params.json"
CV_FILE = TABLES_DIR / "cv_folds.csv"


def save_cv(res: pd.DataFrame, name: str, experiment: str):
    res = res.assign(model=name, experiment=experiment)
    if CV_FILE.exists():
        old = pd.read_csv(CV_FILE)
        old = old[~((old.model == name) & (old.experiment == experiment))]
        res = pd.concat([old, res], ignore_index=True)
    res.to_csv(CV_FILE, index=False)


def run_cv(name, make, cols, experiment, **kw):
    t = time.time()
    res, oof = runner.cv_evaluate(make, cols, df, SPLITS, **kw)
    save_cv(res, name, experiment)
    oof.to_pickle(CACHE_DIR / f"oof_{experiment}_{name}.pkl")
    print(f"  {experiment:4s} {name:28s} PR-AUC {res.pr_auc.mean():.3f} ± {res.pr_auc.std():.3f}  "
          f"ROC {res.roc_auc.mean():.3f}  P@50 {res.prec_k50.mean():.3f}  ({time.time() - t:.0f}s)")
    return res, oof


def params():
    return json.loads(PARAMS_FILE.read_text())


# ---------------------------------------------------------------- tuning
def TUNE():
    """Small grids, scored on the first CV repeat only (5 folds)."""
    out = {}

    def score(make, cols=ALL):
        res, _ = runner.cv_evaluate(make, cols, df, SPLITS, folds=REPEAT0)
        return res.pr_auc.mean()

    lr = {C: score(lambda C=C: M.logistic(C=C)) for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}
    out["logistic"] = {"C": max(lr, key=lr.get), "grid": {str(k): v for k, v in lr.items()}}
    rf = {}
    for leaf, mf in itertools.product([1, 3, 10], ["sqrt", 0.3]):
        rf[f"{leaf}|{mf}"] = score(lambda leaf=leaf, mf=mf: M.random_forest(leaf, mf))
    best = max(rf, key=rf.get)
    out["random_forest"] = {"min_samples_leaf": int(best.split("|")[0]),
                            "max_features": best.split("|")[1] if best.split("|")[1] == "sqrt"
                            else float(best.split("|")[1]), "grid": rf}
    rng = np.random.default_rng(SEED)
    space = dict(num_leaves=[7, 15, 31], learning_rate=[0.02, 0.05], n_estimators=[200, 400, 800],
                 min_child_samples=[10, 20, 50], colsample_bytree=[0.5, 0.8])
    configs = [{k: v[rng.integers(len(v))] for k, v in space.items()} for _ in range(16)]
    configs = [{k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, np.floating) else v)
                for k, v in c.items()} for c in configs]
    lg = [(score(lambda c=c: M.lightgbm(c)), c) for c in configs]
    lg.sort(key=lambda t: -t[0])
    out["lightgbm"] = {"params": lg[0][1], "grid": [{"pr_auc": s, **c} for s, c in lg]}
    PARAMS_FILE.write_text(json.dumps(out, indent=2))
    print("tuned:", {k: (v.get("C") or v.get("params") or v.get("min_samples_leaf")) for k, v in out.items()})


def make_models():
    p = params()
    rf = p["random_forest"]
    return {
        "expert_rules": lambda: M.ExpertRules(),
        "logistic": lambda: M.logistic(C=p["logistic"]["C"]),
        "random_forest": lambda: M.random_forest(rf["min_samples_leaf"], rf["max_features"]),
        "lightgbm": lambda: M.lightgbm(p["lightgbm"]["params"]),
        "isolation_forest": lambda: M.isolation_forest(),
    }


# ---------------------------------------------------------------- experiments
def SIZE():
    """How much does size alone explain? No model is fitted."""
    for col in ["total_reimb", "n_claims", "n_ip_claims", "n_bene"]:
        run_cv(f"size_only:{col}", lambda col=col: M.SingleSignal(col), ALL, "SIZE")


def MODELS():
    """Model comparison on all features."""
    for name, make in make_models().items():
        run_cv(name, make, ALL, "MODELS")


def FAMILIES():
    """Feature-family ablation with LightGBM; plus size-free and size-only models."""
    make = make_models()["lightgbm"]
    run_cv("lgbm:all", make, ALL, "FAMILIES")
    run_cv("lgbm:no_size", make, NO_SIZE, "FAMILIES")
    run_cv("lgbm:size_only", make, FAM["size"], "FAMILIES")
    run_cv("logistic:no_size", make_models()["logistic"], NO_SIZE, "FAMILIES")
    for fam in FAM:
        if fam == "size":
            continue
        cols = [c for c in NO_SIZE if c not in FAM[fam]]
        run_cv(f"lgbm:no_size-minus-{fam}", make, cols, "FAMILIES")


def NETWORK():
    """Network: homophily test on the development set, then models with/without network features."""
    g = runner.graph(df)
    dev = SPLITS["dev"]
    h = homophily_test(g, dev, df.loc[dev, "y"].to_numpy())
    (TABLES_DIR / "homophily.json").write_text(json.dumps(h, indent=2))
    print("  homophily:", {k: round(v, 3) if isinstance(v, float) else v for k, v in h.items()})
    make = make_models()["lightgbm"]
    run_cv("lgbm:all+network", make, ALL, "NETWORK", use_network=True)
    run_cv("lgbm:no_size+network", make, NO_SIZE, "NETWORK", use_network=True)


def IMBALANCE():
    """Imbalance handling, inside the training fold only."""
    p = params()
    for name, make in {
        "lgbm:none": lambda: M.lightgbm(p["lightgbm"]["params"], balanced=False),
        "lgbm:class_weight": lambda: M.lightgbm(p["lightgbm"]["params"], balanced=True),
        "lgbm:smote": lambda: M.lightgbm(p["lightgbm"]["params"], balanced=False,
                                         sampler=SMOTE(random_state=SEED)),
        "logistic:none": lambda: M.logistic(p["logistic"]["C"], balanced=False),
        "logistic:class_weight": lambda: M.logistic(p["logistic"]["C"], balanced=True),
        "logistic:smote": lambda: M.logistic(p["logistic"]["C"], balanced=False,
                                             sampler=SMOTE(random_state=SEED)),
    }.items():
        res, oof = run_cv(name, make, ALL, "IMBALANCE")
    # calibration: Brier score of the out-of-fold probabilities (lower = better)
    rows = []
    for name in ["lgbm:none", "lgbm:class_weight", "lgbm:smote", "logistic:none",
                 "logistic:class_weight", "logistic:smote"]:
        oof = pd.read_pickle(CACHE_DIR / f"oof_IMBALANCE_{name}.pkl")
        y = df.loc[oof.index, "y"]
        rows.append({"model": name, "brier": brier_score_loss(y, oof), "mean_pred": oof.mean(),
                     "base_rate": y.mean()})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "imbalance_calibration.csv", index=False)


def NOISE():
    """Label noise: hide a share of flagged providers in training (they become 0)."""
    make = make_models()["lightgbm"]
    for rate in [0.0, 0.1, 0.2, 0.3, 0.5]:
        def flip(y, i, rate=rate):
            rng = np.random.default_rng(SEED + i)
            y = y.copy()
            pos = np.where(y == 1)[0]
            y[rng.choice(pos, int(round(rate * len(pos))), replace=False)] = 0
            return y
        run_cv(f"lgbm:all|noise={rate}", make, ALL, "NOISE", label_transform=flip)
        run_cv(f"lgbm:no_size|noise={rate}", make, NO_SIZE, "NOISE", label_transform=flip)


def TEST():
    """Open the test set once. Every model is fit on the full development set."""
    p = params()
    mm = make_models()
    candidates = {
        "size_only:total_reimb": (lambda: M.SingleSignal("total_reimb"), ALL, False),
        "expert_rules": (mm["expert_rules"], ALL, False),
        "isolation_forest": (mm["isolation_forest"], ALL, False),
        "logistic": (mm["logistic"], ALL, False),
        "random_forest": (mm["random_forest"], ALL, False),
        "lightgbm": (mm["lightgbm"], ALL, False),
        "lgbm:no_size": (mm["lightgbm"], NO_SIZE, False),
        "lgbm:all+network": (mm["lightgbm"], ALL, True),
    }
    rows, scores = [], {}
    y_test = None
    for name, (make, cols, net) in candidates.items():
        model, s, y_te, X_te = runner.test_evaluate(make, cols, df, SPLITS, use_network=net)
        y_test = y_te
        scores[name] = s
        m = evaluate(y_te, s.to_numpy())
        lo, hi = bootstrap_ci(y_te, s.to_numpy(), "pr_auc")
        lo100, hi100 = bootstrap_ci(y_te, s.to_numpy(), "prec_k100")
        rows.append({"model": name, **m, "pr_auc_lo": lo, "pr_auc_hi": hi,
                     "prec_k100_lo": lo100, "prec_k100_hi": hi100})
        print(f"  TEST {name:24s} PR-AUC {m['pr_auc']:.3f} [{lo:.3f}, {hi:.3f}]  ROC {m['roc_auc']:.3f} "
              f"P@50 {m['prec_k50']:.2f} P@100 {m['prec_k100']:.2f} R@100 {m['rec_k100']:.2f}")
        if name in ("lightgbm", "lgbm:no_size"):
            contrib = M.lgbm_contributions(model, X_te)
            contrib.to_pickle(CACHE_DIR / f"shap_test_{name.replace(':', '_')}.pkl")
            X_te.to_pickle(CACHE_DIR / f"X_test_{name.replace(':', '_')}.pkl")
    pd.DataFrame(rows).to_csv(TABLES_DIR / "test_results.csv", index=False)
    pd.DataFrame(scores).assign(y=y_test).to_pickle(CACHE_DIR / "test_scores.pkl")

    # paired differences on identical bootstrap resamples
    diffs = []
    for a, b in [("lightgbm", "size_only:total_reimb"), ("lgbm:no_size", "size_only:total_reimb"),
                 ("lgbm:all+network", "lightgbm"), ("lightgbm", "logistic"), ("lightgbm", "random_forest")]:
        for metric in ["pr_auc", "prec_k100"]:
            d, lo, hi = paired_bootstrap_diff(y_test, scores[a], scores[b], metric)
            diffs.append({"a": a, "b": b, "metric": metric, "diff": d, "lo": lo, "hi": hi})
    pd.DataFrame(diffs).to_csv(TABLES_DIR / "test_paired_diffs.csv", index=False)

    # QUEUE_COST: queue size chosen on development out-of-fold scores, applied to the test set
    oof = pd.read_pickle(CACHE_DIR / "oof_MODELS_lightgbm.pkl")
    y_dev = df.loc[oof.index, "y"].to_numpy()
    s_dev = oof.to_numpy()
    ops = []
    for ratio in [5, 10, 20, 50]:
        best = None
        for frac in np.linspace(0.01, 0.5, 50):
            k = int(round(frac * len(y_dev)))
            top = np.argsort(-s_dev)[:k]
            tp = y_dev[top].sum(); fp = k - tp; fn = y_dev.sum() - tp
            cost = fp + ratio * fn
            if best is None or cost < best[0]:
                best = (cost, frac)
        frac = best[1]
        k_te = int(round(frac * len(y_test)))
        s_te = scores["lightgbm"].to_numpy()
        thr = np.sort(s_te)[::-1][k_te - 1]
        tm = threshold_metrics(y_test, s_te, thr)
        ops.append({"fn_to_fp_cost": ratio, "queue_share": frac, "queue_size_test": k_te, **tm,
                    "cost_model": tm["fp"] + ratio * tm["fn"],
                    "cost_flag_none": ratio * int(y_test.sum()),
                    "cost_flag_all": int(len(y_test) - y_test.sum())})
    pd.DataFrame(ops).to_csv(TABLES_DIR / "queue_cost_operating_points.csv", index=False)
    print(pd.DataFrame(ops)[["fn_to_fp_cost", "queue_size_test", "tp", "fp", "fn", "precision", "recall"]])


def LEAKY_SPLIT():
    """Claim-level model: random claim split vs provider-grouped split."""
    from lightgbm import LGBMClassifier
    from fraudrank import data
    claims = data.load_claims()
    bene = data.load_beneficiaries()
    c = claims.merge(bene[["BeneID", "DOB", "n_chronic", "RenalDiseaseIndicator", "State", "County"]],
                     on="BeneID", how="left")
    c["age"] = (c["ClaimStartDt"] - c["DOB"]).dt.days / 365.25
    c["y"] = c["Provider"].map(df["y"])
    base = ["InscClaimAmtReimbursed", "is_ip", "los", "n_diag", "n_proc", "claim_days", "age",
            "n_chronic", "RenalDiseaseIndicator", "OPDeductible"]
    # naive identifier encodings that many public notebooks use
    for col in ["AttendingPhysician", "OperatingPhysician", "DiagnosisGroupCode", "ClmDiagnosisCode_1"]:
        c[col + "_code"] = pd.factorize(c[col])[0]
    ids = ["AttendingPhysician_code", "OperatingPhysician_code", "State", "County"]
    codes = ["DiagnosisGroupCode_code", "ClmDiagnosisCode_1_code"]
    rng = np.random.default_rng(SEED)
    random_test = rng.random(len(c)) < 0.2
    grouped_test = c["Provider"].isin(SPLITS["test"]).to_numpy()
    rows = []
    for fs_name, cols in [("claim features", base + codes), ("claim features + IDs", base + codes + ids)]:
        for split_name, te in [("random claim split", random_test), ("provider-grouped split", grouped_test)]:
            m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, random_state=SEED, n_jobs=2,
                               verbose=-1)
            m.fit(c.loc[~te, cols], c.loc[~te, "y"])
            s = m.predict_proba(c.loc[te, cols])[:, 1]
            yt = c.loc[te, "y"].to_numpy()
            claim_m = evaluate(yt, s, {})
            prov = pd.DataFrame({"p": c.loc[te, "Provider"].to_numpy(), "s": s, "y": yt}).groupby("p").mean()
            prov_m = evaluate(prov.y.round().astype(int), prov.s, {})
            seen = set(c.loc[~te, "AttendingPhysician"].dropna())
            rows.append({"features": fs_name, "split": split_name, "claim_base_rate": yt.mean(),
                         "test_physician_seen_in_train": c.loc[te, "AttendingPhysician"].isin(seen).mean(),
                         "claim_pr_auc": claim_m["pr_auc"],
                         "claim_roc_auc": claim_m["roc_auc"], "provider_pr_auc": prov_m["pr_auc"],
                         "provider_roc_auc": prov_m["roc_auc"], "test_claims": int(te.sum()),
                         "test_providers": len(prov)})
            print("  LEAKY_SPLIT", rows[-1])
    pd.DataFrame(rows).to_csv(TABLES_DIR / "leaky_split.csv", index=False)
    ph = c.groupby("AttendingPhysician")["Provider"].nunique()
    (TABLES_DIR / "leaky_split_physicians.json").write_text(json.dumps(
        {"attending_physicians": int(len(ph)), "share_billing_for_one_provider": float((ph == 1).mean())}, indent=2))


def SUMMARY():
    cv = pd.read_csv(CV_FILE)
    summ = (cv.drop(columns="fold").groupby(["experiment", "model"])
            .agg(["mean", "std"]))
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(TABLES_DIR / "cv_summary.csv", index=False)
    # paired per-fold differences for the key comparisons
    piv = cv.pivot_table(index="fold", columns="model", values="pr_auc")
    pairs = [("lightgbm", "size_only:total_reimb"), ("lgbm:no_size", "size_only:total_reimb"),
             ("lgbm:all+network", "lgbm:all"), ("lgbm:no_size+network", "lgbm:no_size"),
             ("lightgbm", "logistic"), ("lightgbm", "random_forest"), ("lightgbm", "expert_rules")]
    rows = []
    for a, b in pairs:
        if a in piv and b in piv:
            d = piv[a] - piv[b]
            rows.append({"a": a, "b": b, "mean_diff": d.mean(), "sd_diff": d.std(),
                         "folds_a_better": int((d > 0).sum()), "folds": int(d.notna().sum())})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "cv_paired_diffs.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())


STEPS = {"TUNE": TUNE, "SIZE": SIZE, "MODELS": MODELS, "FAMILIES": FAMILIES, "NETWORK": NETWORK, "IMBALANCE": IMBALANCE, "NOISE": NOISE, "TEST": TEST,
         "LEAKY_SPLIT": LEAKY_SPLIT, "SUMMARY": SUMMARY}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(STEPS)
    for step in todo:
        print(f"== {step}")
        t = time.time()
        STEPS[step]()
        print(f"== {step} done in {time.time() - t:.0f}s")
