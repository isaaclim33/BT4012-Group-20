"""Step 2: run every non-LLM experiment and save tables to results/.

Usage:  python scripts/02_run_experiments.py [SIZE MODELS ...]   (default: all)

Order matters: TUNE writes the chosen hyperparameters that the model experiments use, and TEST
opens the held-out test set once, after all choices are frozen.
"""
import ast
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


def EDA_TESTS():
    """Statistical EDA: robust descriptives, hypothesis tests, Benford's law, boxplot outliers."""
    from scipy import stats
    from fraudrank import data
    out = {}
    # skew: why the models log-transform and why the group tests are non-parametric
    out["skew_kurtosis"] = {c: {"skew_raw": float(stats.skew(df[c])), "kurtosis_raw": float(stats.kurtosis(df[c])),
                                "skew_log1p": float(stats.skew(np.log1p(df[c])))}
                            for c in ["total_reimb", "n_claims", "claims_per_bene"]}
    # chi-square: is the flag independent of the type of care billed?
    kind = np.select([(df.has_ip == 1) & (df.has_op == 1), df.has_ip == 1], ["both", "ip_only"], "op_only")
    tab = pd.crosstab(kind, df["y"])
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    out["chi2_care_type_vs_flag"] = {"chi2": float(chi2), "dof": int(dof), "p": float(p),
                                     "cramers_v": float(np.sqrt(chi2 / (len(df) * (min(tab.shape) - 1))))}
    # Mann-Whitney U (robust to skew) and Welch t-test, flagged vs not, with medians and MAD
    rows = []
    for c in ["total_reimb", "n_claims", "reimb_per_claim", "claims_per_bene", "los_max", "diag_per_claim",
              "n_chronic_mean", "age_mean"]:
        a, b = df.loc[df.y == 1, c], df.loc[df.y == 0, c]
        u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
        _, pt = stats.ttest_ind(a, b, equal_var=False)
        rows.append({"feature": c, "median_flagged": a.median(), "median_not": b.median(),
                     "mad_flagged": stats.median_abs_deviation(a), "mad_not": stats.median_abs_deviation(b),
                     "mw_p": pu, "welch_t_p": pt, "rank_biserial": 1 - 2 * u / (len(a) * len(b))})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "eda_group_tests.csv", index=False)
    # Benford's law: first-digit test, with Nigrini's mean absolute deviation (MAD) as the size-free measure
    claims = data.load_claims()
    ben = np.log10(1 + 1 / np.arange(1, 10))

    def benford(x):
        x = np.asarray(x, float)
        x = x[x >= 1]
        d = np.array([int(str(int(v))[0]) for v in x])
        obs = np.bincount(d, minlength=10)[1:]
        chi2, p = stats.chisquare(obs, ben * obs.sum())
        frac = obs / obs.sum()
        return {"n": int(obs.sum()), "chi2": float(chi2), "p": float(p),
                "mad": float(np.abs(frac - ben).mean()), "observed": [round(float(f), 4) for f in frac]}
    lab = claims["Provider"].map(df["y"])
    amt = claims["InscClaimAmtReimbursed"]
    out["benford"] = {
        "expected": [round(float(v), 4) for v in ben],
        "inpatient_claims": benford(amt[claims.is_ip == 1]),
        "outpatient_claims": benford(amt[claims.is_ip == 0]),
        "outpatient_claims_flagged": benford(amt[(claims.is_ip == 0) & (lab == 1)]),
        "outpatient_claims_not_flagged": benford(amt[(claims.is_ip == 0) & (lab == 0)]),
        "provider_totals": benford(df["total_reimb"]),
        "note": "Nigrini first-digit MAD: <0.006 close, 0.006-0.012 acceptable, 0.012-0.015 marginal, >0.015 nonconformity",
    }
    # boxplot rule (Q3 + 1.5 IQR) on log total reimbursement, as a univariate outlier screen
    lt = np.log1p(df["total_reimb"])
    out["outliers_total_reimb"] = {}
    for scale, v in [("raw", df["total_reimb"]), ("log1p", lt)]:
        q1, q3 = np.percentile(v, [25, 75])
        box = v > q3 + 1.5 * (q3 - q1)                 # upper tail only: unusually large billing
        z = (v - v.mean()) / v.std()
        out["outliers_total_reimb"][scale] = {
            "boxplot_n": int(box.sum()), "boxplot_share": float(box.mean()),
            "boxplot_flag_rate": float(df.loc[box, "y"].mean()) if box.any() else None,
            "boxplot_recall": float(df.loc[box, "y"].sum() / df["y"].sum()),
            "z3_n": int((z > 3).sum()), "z3_flag_rate": float(df.loc[z > 3, "y"].mean()) if (z > 3).any() else None}
    (TABLES_DIR / "eda_tests.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "benford"}, indent=1))
    print({k: (v["mad"], v["p"]) for k, v in out["benford"].items() if isinstance(v, dict)})


DEEP_PARAMS_FILE = TABLES_DIR / "tuned_params_deep.json"


def DEEP():
    """Deep models: a feedforward network (supervised) and an autoencoder trained on
    non-flagged providers (anomaly score = reconstruction error). Tuned on CV repeat 1,
    cross-validated on all 25 folds, then scored once on the test set."""
    def score(make):
        res, _ = runner.cv_evaluate(make, ALL, df, SPLITS, folds=REPEAT0)
        return res.pr_auc.mean()
    mlp_grid = {f"{h}|{a}": score(lambda h=h, a=a: M.mlp(h, a))
                for h in [(32,), (64, 32)] for a in [1e-4, 1e-3, 1e-2, 1e-1]}
    best_mlp = max(mlp_grid, key=mlp_grid.get)
    ae_grid = {str(h): score(lambda h=h: M.Autoencoder(h)) for h in [(32, 8, 32), (16, 4, 16)]}
    best_ae = max(ae_grid, key=ae_grid.get)
    h_mlp, a_mlp = ast.literal_eval(best_mlp.split("|")[0]), float(best_mlp.split("|")[1])
    h_ae = ast.literal_eval(best_ae)
    DEEP_PARAMS_FILE.write_text(json.dumps({"mlp": {"hidden": h_mlp, "alpha": a_mlp, "grid": mlp_grid},
                                            "autoencoder": {"hidden": h_ae, "grid": ae_grid}}, indent=2))
    print("  tuned:", best_mlp, best_ae)
    makes = {"mlp": lambda: M.mlp(h_mlp, a_mlp), "autoencoder": lambda: M.Autoencoder(h_ae)}
    for name, make in makes.items():
        run_cv(name, make, ALL, "MODELS")
    # test set, scored once; appended to the TEST tables (LightGBM scores are not refit)
    base = pd.read_pickle(CACHE_DIR / "test_scores.pkl")
    y_test = base["y"].to_numpy()
    rows, deep_scores = [], {}
    for name, make in makes.items():
        _, s, y_te, _ = runner.test_evaluate(make, ALL, df, SPLITS)
        assert (s.index == base.index).all()
        deep_scores[name] = s
        m = evaluate(y_te, s.to_numpy())
        lo, hi = bootstrap_ci(y_te, s.to_numpy(), "pr_auc")
        lo100, hi100 = bootstrap_ci(y_te, s.to_numpy(), "prec_k100")
        rows.append({"model": name, **m, "pr_auc_lo": lo, "pr_auc_hi": hi, "prec_k100_lo": lo100, "prec_k100_hi": hi100})
        print(f"  TEST {name:24s} PR-AUC {m['pr_auc']:.3f} [{lo:.3f}, {hi:.3f}]  P@50 {m['prec_k50']:.2f} "
              f"P@100 {m['prec_k100']:.2f}")
    tr = pd.read_csv(TABLES_DIR / "test_results.csv")
    tr = pd.concat([tr[~tr.model.isin(makes)], pd.DataFrame(rows)], ignore_index=True)
    tr.to_csv(TABLES_DIR / "test_results.csv", index=False)
    pd.DataFrame(deep_scores).assign(y=y_test).to_pickle(CACHE_DIR / "test_scores_deep.pkl")
    pdiff = pd.read_csv(TABLES_DIR / "test_paired_diffs.csv")
    new = []
    for a, b, sa, sb in [("lightgbm", "mlp", base["lightgbm"], deep_scores["mlp"]),
                         ("mlp", "size_only:total_reimb", deep_scores["mlp"], base["size_only:total_reimb"]),
                         ("autoencoder", "isolation_forest", deep_scores["autoencoder"], base["isolation_forest"])]:
        for metric in ["pr_auc", "prec_k100"]:
            d, lo, hi = paired_bootstrap_diff(y_test, sa, sb, metric)
            new.append({"a": a, "b": b, "metric": metric, "diff": d, "lo": lo, "hi": hi})
    keys = {(r["a"], r["b"]) for r in new}
    pdiff = pdiff[~pdiff.apply(lambda r: (r.a, r.b) in keys, axis=1)]
    pd.concat([pdiff, pd.DataFrame(new)], ignore_index=True).to_csv(TABLES_DIR / "test_paired_diffs.csv", index=False)


def _attack(X: pd.DataFrame, honest_median: pd.Series, split: bool, blend: bool) -> pd.DataFrame:
    """EVASION attacks on a provider's feature row (applied to flagged providers only).
    split: the provider bills through two IDs, so every size count and total halves; every ratio
           (per claim, per patient, per physician) is unchanged. This is the feature-level
           approximation; _claim_split simulates the same attack on claims.
    blend: the provider cuts 75% of its care intensity above the typical honest provider (median).
           Shrinking toward the median, not capping at it, leaves no identical values that a model
           could learn as an attack signature."""
    X = X.copy()
    if split:
        halve = FAM["size"]
        X[halve] = np.where(X[halve] > 0, np.maximum(1, X[halve] / 2), 0)
    if blend:
        for c in ["los_mean", "los_max", "los_long_share", "ip_claim_days_mean", "claims_per_bene"]:
            m = honest_median[c]
            X[c] = np.where(X[c] > m, m + 0.25 * (X[c] - m), X[c])
    return X


def _claim_split(claims: pd.DataFrame, bene: pd.DataFrame, flagged: list[str], seed: int):
    """The split attack on claims: each flagged provider's patients are divided at random between
    two IDs (P_A, P_B) and the provider table is rebuilt, so maxima and ratios change as they would
    in practice. Returns the rebuilt table and each row's original provider ID."""
    from fraudrank import features as F
    rng = np.random.default_rng(seed)
    hit = claims.Provider.isin(set(flagged)).to_numpy()
    pb = claims.loc[hit, ["Provider", "BeneID"]].drop_duplicates()
    pb["half"] = np.where(rng.integers(0, 2, len(pb)) == 0, "_A", "_B")
    cl = claims.copy()
    m = cl.loc[hit, ["Provider", "BeneID"]].merge(pb, on=["Provider", "BeneID"], how="left")
    cl.loc[hit, "Provider"] = (m.Provider + m.half).to_numpy()
    feat = F.build_provider_features(cl, bene, pd.DataFrame({"Provider": sorted(cl.Provider.unique()), "y": 0}))
    return feat, feat.index.to_series().str.replace(r"_[AB]$", "", regex=True), cl


def _physician_overlap(claims: pd.DataFrame, a: str, b: str) -> float:
    sa = set(claims.loc[claims.Provider == a, "AttendingPhysician"].dropna())
    sb = set(claims.loc[claims.Provider == b, "AttendingPhysician"].dropna())
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def EVASION():
    """Adversarial robustness on tabular data. Black-box evasion: flagged test
    providers change their observable billing to slip out of the queue; honest providers do not
    change. Proactive defence: adversarial training on attacked copies of flagged dev providers."""
    p = params()
    dev, test = SPLITS["dev"], SPLITS["test"]
    honest_median = df.loc[[i for i in dev if df.loc[i, "y"] == 0]].median()
    y_te = df.loc[test, "y"].to_numpy()
    flagged = df.loc[test, "y"] == 1
    attacks = {"none": (False, False), "split": (True, False), "blend": (False, True), "split+blend": (True, True)}
    X_dev = df.loc[dev, ALL]
    y_dev = df.loc[dev, "y"].to_numpy()
    # adversarial training: add split copies of every flagged dev provider, label 1. Only the split
    # attack is seen in training, so blend-in tests an attack the defence has not seen (training on
    # every attack makes attacked rows trivially separable, which says nothing about new attacks).
    pos = X_dev[y_dev == 1]
    aug = pd.concat([X_dev, _attack(pos, honest_median, True, False)])
    y_aug = np.concatenate([y_dev, np.ones(len(pos), int)])
    models = {
        "size_only:total_reimb": (M.SingleSignal("total_reimb").fit(X_dev, y_dev), ALL),
        "lightgbm": (M.lightgbm(p["lightgbm"]["params"]).fit(X_dev, y_dev), ALL),
        "lgbm:no_size": (M.lightgbm(p["lightgbm"]["params"]).fit(df.loc[dev, NO_SIZE], y_dev), NO_SIZE),
        "lightgbm:adversarial_training": (M.lightgbm(p["lightgbm"]["params"]).fit(aug, y_aug), ALL),
    }
    rows = []
    for aname, (s_, b_) in attacks.items():
        X_te = df.loc[test, ALL].copy()
        X_te.loc[flagged] = _attack(X_te.loc[flagged], honest_median, s_, b_)
        for mname, (model, cols) in models.items():
            s = model.score(X_te[cols])
            m = evaluate(y_te, s)
            rows.append({"attack": aname, "model": mname, "pr_auc": m["pr_auc"], "prec_k50": m["prec_k50"],
                         "rec_k100": m["rec_k100"], "prec_k100": m["prec_k100"]})
    # split on claims: a provider counts by its higher-scoring ID; three random divisions averaged
    from fraudrank import data
    claims, bene = data.load_claims(), data.load_beneficiaries()
    flagged_ids = [p_ for p_ in test if df.loc[p_, "y"] == 1]
    unsplit = [p_ for p_ in test if p_ not in flagged_ids]
    per_seed = {(a, mname): [] for a in ("split (claims)", "split (claims)+blend") for mname in models}
    halves_overlap = []
    for seed in (SEED, 1, 2):
        feat, parent, cl = _claim_split(claims, bene, flagged_ids, seed)
        assert np.allclose(feat.loc[unsplit, ALL], df.loc[unsplit, ALL]), "rebuild must leave other providers unchanged"
        keep = parent.isin(set(test))
        halves = feat.index[parent.isin(set(flagged_ids))]
        blended = feat.copy()
        blended.loc[halves, ALL] = _attack(blended.loc[halves, ALL], honest_median, False, True)
        for aname, table in [("split (claims)", feat), ("split (claims)+blend", blended)]:
            for mname, (model, cols) in models.items():
                s = pd.Series(model.score(table.loc[keep, cols]), index=table.index[keep]).groupby(parent[keep]).max()
                per_seed[(aname, mname)].append(evaluate(y_te, s.reindex(test).to_numpy()))
        if seed == SEED:
            halves_overlap = [_physician_overlap(cl, f"{p_}_A", f"{p_}_B") for p_ in flagged_ids]
    metrics_ = ["pr_auc", "prec_k50", "rec_k100", "prec_k100"]
    for (aname, mname), ms in per_seed.items():
        rows.append({"attack": aname, "model": mname, **{k: float(np.mean([m_[k] for m_ in ms])) for k in metrics_}})
    # what each attack costs, and whether shared attending physicians would expose a split
    ip = claims[claims.is_ip == 1]
    drg = ip.groupby("DiagnosisGroupCode").filter(lambda g: len(g) >= 50).groupby("DiagnosisGroupCode")
    stay_pay = drg.apply(lambda g: g.los.corr(g.InscClaimAmtReimbursed, method="spearman"), include_groups=False)
    pp = claims[["Provider", "AttendingPhysician"]].dropna().drop_duplicates()
    sets_ = pp.groupby("Provider").AttendingPhysician.apply(set)
    pairs = pp.merge(pp, on="AttendingPhysician")
    pairs = pairs[pairs.Provider_x < pairs.Provider_y][["Provider_x", "Provider_y"]].drop_duplicates()
    real = np.array([len(sets_[a] & sets_[b]) / len(sets_[a] | sets_[b]) for a, b in zip(pairs.Provider_x, pairs.Provider_y)])
    notes = {"split_payment_lost": 0.0,
             "within_drg_spearman_stay_vs_payment": {"drgs": int(len(stay_pay)), "median": float(stay_pay.median())},
             "attending_physician_overlap": {
                 "real_pairs_sharing_any": int(len(real)), "real_pairs_identical_sets": int((real == 1).sum()),
                 "real_pairs_median": float(np.median(real)),
                 "split_halves_median": float(np.median(halves_overlap)),
                 "threshold_catching_80pct_of_splits": float(np.quantile(halves_overlap, 0.2)),
                 "real_pairs_at_or_above_that_threshold": int((real >= np.quantile(halves_overlap, 0.2)).sum())}}
    (TABLES_DIR / "evasion_notes.json").write_text(json.dumps(notes, indent=2))
    print(json.dumps(notes, indent=1))
    res = pd.DataFrame(rows)
    res.to_csv(TABLES_DIR / "evasion.csv", index=False)
    print(res.pivot(index="model", columns="attack", values="pr_auc").round(3))
    print(res.pivot(index="model", columns="attack", values="prec_k50").round(2))


def FAIRNESS():
    """Fairness: false-positive rates by provider group at a fixed queue, and calibration
    (is the risk score a probability?)."""
    p = params()
    dev, test = SPLITS["dev"], SPLITS["test"]
    scores = pd.read_pickle(CACHE_DIR / "test_scores.pkl")
    no_pm = [c for c in ALL if c not in FAM["patient_mix"]]
    _, s_nopm, _, _ = runner.test_evaluate(lambda: M.lightgbm(p["lightgbm"]["params"]), no_pm, df, SPLITS)
    scores = scores.assign(**{"lgbm:no_patient_mix": s_nopm})
    te = df.loc[test]
    acuity_cuts = np.quantile(df.loc[dev, "n_chronic_mean"], [1 / 3, 2 / 3])
    groups = {
        "care": np.where(te.has_ip == 1, "bills inpatient", "outpatient only"),
        "patient_acuity": np.select([te.n_chronic_mean <= acuity_cuts[0], te.n_chronic_mean <= acuity_cuts[1]],
                                    ["low", "middle"], "high"),
    }
    rows = []
    for model in ["lightgbm", "lgbm:no_size", "lgbm:no_patient_mix"]:
        for k in [100, 184]:          # k=100 and the 5:1 cost-optimal queue (Table 7)
            top = scores[model].rank(ascending=False, method="first") <= k
            for gname, g in groups.items():
                for level in np.unique(g):
                    mask = g == level
                    yy, tt = te["y"].to_numpy()[mask], top.to_numpy()[mask]
                    neg = (yy == 0).sum()
                    rows.append({"model": model, "k": k, "group": gname, "level": level, "providers": int(mask.sum()),
                                 "flag_rate": yy.mean(), "queued": int(tt.sum()),
                                 "fp": int(((yy == 0) & tt).sum()), "fpr": ((yy == 0) & tt).sum() / max(1, neg),
                                 "recall": ((yy == 1) & tt).sum() / max(1, yy.sum())})
    fr = pd.DataFrame(rows)
    fr.to_csv(TABLES_DIR / "fairness.csv", index=False)
    print(fr.round(3).to_string())
    # calibration: expected calibration error over 10 equal-count bins of out-of-fold scores
    cal = []
    for name in ["lgbm:none", "lgbm:class_weight", "logistic:none", "logistic:class_weight"]:
        oof = pd.read_pickle(CACHE_DIR / f"oof_IMBALANCE_{name}.pkl")
        y = df.loc[oof.index, "y"].to_numpy()
        bins = pd.qcut(oof.rank(method="first"), 10, labels=False)
        g = pd.DataFrame({"p": oof.to_numpy(), "y": y, "b": bins.to_numpy()}).groupby("b").mean()
        cal.append({"model": name, "ece": float((g.p - g.y).abs().mean()),
                    "top_decile_pred": float(g.p.iloc[-1]), "top_decile_rate": float(g.y.iloc[-1])})
    pd.DataFrame(cal).to_csv(TABLES_DIR / "calibration.csv", index=False)
    print(pd.DataFrame(cal).round(3))


MAXIMA = ["los_max", "ip_reimb_max", "op_reimb_max"]


def SIZE_BANDS():
    """Is there signal beyond size? Dropping the size family is not enough, because the remaining
    features reconstruct size. So: (1) measure that reconstruction; (2) compare models within size
    quintiles; (3) separate the maxima, which grow with claim count, from volume-adjusted versions;
    (4) count queue actions that change when the size-free vote uses the adjusted maxima.
    Reported only: the pipeline's features are unchanged."""
    from lightgbm import LGBMRegressor
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, r2_score
    from fraudrank import audit_selection as AS, data, features as F
    from fraudrank.config import VOTE_SEEDS
    p = params()["lightgbm"]["params"]
    dev = SPLITS["dev"]
    adj = F.volume_adjusted_maxima(data.load_claims(), df.index, dev)
    d = df.join(adj)
    adjusted = [c for c in NO_SIZE if c not in MAXIMA] + list(adj.columns)

    size = np.log1p(d["total_reimb"])
    pred = pd.Series(index=dev, dtype=float)
    for fold in REPEAT0:
        reg = LGBMRegressor(n_estimators=300, random_state=SEED, verbose=-1)
        reg.fit(d.loc[fold["train"], NO_SIZE], size[fold["train"]])
        pred[fold["val"]] = reg.predict(d.loc[fold["val"], NO_SIZE])
    out = {"r2_size_from_size_free_features": float(r2_score(size[dev], pred[dev])),
           "volume_correlation": {}}
    for c, n in [("los_max", "n_ip_claims"), ("ip_reimb_max", "n_ip_claims"), ("op_reimb_max", "n_op_claims")]:
        has = d.loc[dev, n] > 0
        out["volume_correlation"][c] = {
            "raw": float(spearmanr(d.loc[dev][has][c], d.loc[dev][has][n])[0]),
            "adjusted": float(spearmanr(d.loc[dev][has][c + "_adj"], d.loc[dev][has][n])[0])}

    sets = {"lgbm:no_size": NO_SIZE, "lgbm:no_size-maxima": [c for c in NO_SIZE if c not in MAXIMA],
            "lgbm:no_size+adjusted_maxima": adjusted, "lgbm:all": ALL, "lgbm:all+adjusted_maxima": ALL + list(adj.columns)}
    oof = {}
    for name, cols in sets.items():
        res, oof[name] = runner.cv_evaluate(lambda: M.lightgbm(p), cols, d, SPLITS)
        save_cv(res, name, "SIZE_BANDS")
        print(f"  SIZE_BANDS {name:30s} PR-AUC {res.pr_auc.mean():.3f} ± {res.pr_auc.std():.3f}")

    y = d.loc[dev, "y"]
    band = pd.qcut(d.loc[dev, "total_reimb"].rank(method="first"), 5, labels=False)
    groups = [(f"quintile {b + 1}", band.index[band == b]) for b in range(5)] + [("quintiles 4-5", band.index[band >= 3])]
    rows = []
    for name, idx in groups:
        row = {"band": name, "providers": len(idx), "flagged": int(y[idx].sum()), "base_rate": float(y[idx].mean()),
               "size_only:total_reimb": average_precision_score(y[idx], d.loc[idx, "total_reimb"])}
        rows.append(row | {m: average_precision_score(y[idx], oof[m][idx]) for m in sets})
    bands = pd.DataFrame(rows)
    bands.to_csv(TABLES_DIR / "size_bands.csv", index=False)

    # the queue's gate 2 keeps the full model; only the size-free vote (the action) changes
    checks = AS.billing_checks(df.index)
    score = pd.read_pickle(CACHE_DIR / "test_scores.pkl")["lightgbm"]
    queues = {}
    for label, cols in [("current", NO_SIZE), ("adjusted", adjusted)]:
        votes = AS.seed_votes({s: runner.test_evaluate(lambda s=s: M.lightgbm(p, seed=s), cols, d, SPLITS)[1]
                               for s in VOTE_SEEDS})
        queues[label] = AS.two_gate_selection(score, checks, votes).set_index("provider")
    q = queues["current"][["gate", "action", "votes"]].join(
        queues["adjusted"][["action", "votes"]], rsuffix="_adjusted")
    q = q.assign(previously_flagged=d.loc[q.index, "y"].astype(int), total_reimb=d.loc[q.index, "total_reimb"])
    q.reset_index().to_csv(TABLES_DIR / "queue_sensitivity.csv", index=False)
    changed = q[q.action != q.action_adjusted]
    out["queue_actions_changed"] = {"queued": len(q), "changed": len(changed),
                                    "full_audit_to_desk": int((changed.action == "priority_audit").sum()),
                                    "desk_to_full_audit": int((changed.action == "desk_review").sum())}
    (TABLES_DIR / "size_bands.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    print(bands.round(3).to_string(index=False))


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
             ("lightgbm", "logistic"), ("lightgbm", "random_forest"), ("lightgbm", "expert_rules"),
             ("lightgbm", "mlp"), ("mlp", "size_only:total_reimb"), ("autoencoder", "isolation_forest"),
             ("lgbm:no_size-maxima", "lgbm:no_size"), ("lgbm:no_size+adjusted_maxima", "lgbm:no_size"),
             ("lgbm:no_size+adjusted_maxima", "size_only:total_reimb"), ("lgbm:all+adjusted_maxima", "lgbm:all")]
    rows = []
    for a, b in pairs:
        if a in piv and b in piv:
            d = piv[a] - piv[b]
            rows.append({"a": a, "b": b, "mean_diff": d.mean(), "sd_diff": d.std(),
                         "folds_a_better": int((d > 0).sum()), "folds": int(d.notna().sum())})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "cv_paired_diffs.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())


STEPS = {"EDA_TESTS": EDA_TESTS, "TUNE": TUNE, "SIZE": SIZE, "MODELS": MODELS, "FAMILIES": FAMILIES,
         "NETWORK": NETWORK, "IMBALANCE": IMBALANCE, "NOISE": NOISE, "TEST": TEST, "DEEP": DEEP,
         "EVASION": EVASION, "FAIRNESS": FAIRNESS, "SIZE_BANDS": SIZE_BANDS, "LEAKY_SPLIT": LEAKY_SPLIT,
         "SUMMARY": SUMMARY}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(STEPS)
    for step in todo:
        print(f"== {step}")
        t = time.time()
        STEPS[step]()
        print(f"== {step} done in {time.time() - t:.0f}s")
