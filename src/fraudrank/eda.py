"""Development-only exploratory statistics; no fitting or held-out labels."""
import json

import numpy as np
import pandas as pd


def run_eda(df, splits, tables_dir, claims=None, *, verbose=True):
    """Statistical EDA: robust descriptives, hypothesis tests, Benford's law, boxplot outliers."""
    from scipy import stats
    from . import data
    df = df.loc[splits["dev"]]
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
    pd.DataFrame(rows).to_csv(tables_dir / "eda_group_tests.csv", index=False)
    # Benford's law: first-digit test, with Nigrini's mean absolute deviation (MAD) as the size-free measure
    claims = data.load_claims() if claims is None else claims
    claims = claims[claims.Provider.isin(df.index)]
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
    (tables_dir / "eda_tests.json").write_text(json.dumps(out, indent=2))
    if verbose:
        print(json.dumps({k: v for k, v in out.items() if k != "benford"}, indent=1))
        print({k: (v["mad"], v["p"]) for k, v in out["benford"].items() if isinstance(v, dict)})

