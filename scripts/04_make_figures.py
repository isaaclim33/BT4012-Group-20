"""Step 4: figures for the report, drawn from results/tables and results/cache.

Static figures for a printed PDF: thin marks, recessive grid, one axis per panel,
categorical hues in fixed order (blue, orange, aqua), labels in text ink.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraudrank import runner  # noqa: E402
from fraudrank.config import CACHE_DIR, FIGURES_DIR, TABLES_DIR  # noqa: E402
from fraudrank.metrics import precision_recall_at_k  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29d"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 220,
    "savefig.bbox": "tight", "text.color": INK,
})

df, FAM, SPLITS = runner.load_all()
cv = pd.read_csv(TABLES_DIR / "cv_summary.csv")


def cvrow(exp, model):
    r = cv[(cv.experiment == exp) & (cv.model == model)].iloc[0]
    return r["pr_auc_mean"], r["pr_auc_std"]


# ---------------------------------------------------------------- Fig 1 data
def fig_data():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1, 1.25]})
    t = np.select([(df.has_ip == 1) & (df.has_op == 1), df.has_ip == 1], ["Inpatient +\noutpatient", "Inpatient\nonly"],
                  "Outpatient\nonly")
    g = df.assign(t=t).groupby("t")["y"].agg(["mean", "size", "sum"]).loc[
        ["Inpatient +\noutpatient", "Inpatient\nonly", "Outpatient\nonly"]]
    a.bar(g.index, g["mean"] * 100, color=BLUE, width=0.55)
    for i, (m, n, s) in enumerate(g.itertuples(index=False)):
        a.text(i, m * 100 + 0.6, f"{m * 100:.1f}%\n({int(s)} of {n:,})", ha="center", va="bottom", fontsize=7.5,
               color=INK)
    a.set_ylim(0, 32)
    a.set_ylabel("Providers flagged (%)")
    a.set_title("(a) Flag rate by type of care billed")
    a.grid(axis="x", visible=False)
    bins = np.logspace(0, np.log10(df.n_claims.max() + 1), 40)
    b.hist(df.loc[df.y == 0, "n_claims"], bins=bins, color=MUTED, alpha=0.9, label="Not flagged (4,904)")
    b.hist(df.loc[df.y == 1, "n_claims"], bins=bins, color=ORANGE, alpha=0.9, label="Flagged (506)")
    b.set_xscale("log")
    b.set_xlabel("Claims per provider (log scale)")
    b.set_ylabel("Providers")
    b.set_title("(b) Flagged providers file far more claims")
    b.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_data.png")


# ---------------------------------------------------------------- Fig 2 models (CV)
def fig_models():
    rows = [("LightGBM, all features", "E1", "lightgbm"), ("LightGBM, no size features", "E2", "lgbm:no_size"),
            ("Random forest", "E1", "random_forest"), ("Logistic regression", "E1", "logistic"),
            ("LightGBM, size features only", "E2", "lgbm:size_only"),
            ("Total reimbursement alone", "E0", "size_only:total_reimb"),
            ("Inpatient claim count alone", "E0", "size_only:n_ip_claims"),
            ("Expert rules", "E1", "expert_rules"), ("Claim count alone", "E0", "size_only:n_claims"),
            ("Isolation forest (unsupervised)", "E1", "isolation_forest")]
    vals = [(lab, *cvrow(e, m), e) for lab, e, m in rows]
    vals.sort(key=lambda v: v[1])
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    y = np.arange(len(vals))
    for i, (lab, m, s, e) in enumerate(vals):
        c = MUTED if e == "E0" else BLUE
        ax.errorbar(m, i, xerr=s, fmt="o", color=c, ms=5, elinewidth=1.2, capsize=0)
        ax.text(m + s + 0.012, i, f"{m:.3f}", va="center", fontsize=7.5, color=INK)
    ref = cvrow("E0", "size_only:total_reimb")[0]
    ax.axvline(ref, color=ORANGE, lw=1, ls="--")
    ax.text(ref - 0.006, len(vals) - 0.35, "size-alone bar", color=INK2, fontsize=7.5, ha="right", va="center")
    ax.axvline(df.y.mean(), color=MUTED, lw=0.8, ls=":")
    ax.text(df.y.mean() + 0.006, -0.6, "random (0.094)", color=INK2, fontsize=7, va="center")
    ax.set_yticks(y, [v[0] for v in vals])
    ax.set_xlim(0, 0.9)
    ax.set_ylim(-1, len(vals))
    ax.set_xlabel("PR-AUC, mean ± SD over 25 validation folds")
    ax.grid(axis="y", visible=False)
    ax.set_title("Supervised models clear the size-alone bar; grey points are single size signals, no model")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_models_cv.png")


# ---------------------------------------------------------------- Fig 3 ablation
def fig_ablation():
    base = cvrow("E2", "lgbm:no_size")[0]
    names = {"intensity": "Intensity (stay length, claims per patient)", "coding": "Coding (diagnoses, procedures)",
             "money": "Money per claim / patient", "service_mix": "Service mix (inpatient share)",
             "duplication": "Duplicate claims", "physicians": "Physician concentration",
             "patient_mix": "Patient mix (age, chronic conditions)"}
    d = sorted([(names[f], cvrow("E2", f"lgbm:no_size-minus-{f}")[0] - base) for f in names], key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    y = np.arange(len(d))
    ax.barh(y, [v for _, v in d], color=[BLUE if v < -0.01 else MUTED for _, v in d], height=0.55)
    for i, (_, v) in enumerate(d):
        lab = "0.000" if abs(v) < 0.0005 else f"{v:+.3f}"
        ax.text(v - 0.002 if v < 0 else v + 0.002, i, lab, va="center", ha="right" if v < 0 else "left",
                fontsize=7.5, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [n for n, _ in d])
    ax.set_xlim(-0.11, 0.025)
    ax.set_xlabel(f"Change in PR-AUC when the family is removed (size-free LightGBM, base {base:.3f})")
    ax.grid(axis="y", visible=False)
    ax.set_title("Intensity carries the size-free signal; most other families are interchangeable")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_ablation.png")


# ---------------------------------------------------------------- Fig 4 test precision@k
def fig_test():
    s = pd.read_pickle(CACHE_DIR / "test_scores.pkl")
    y = s["y"].to_numpy()
    ks = np.arange(10, 301, 5)
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    for col, lab, c in [("lightgbm", "LightGBM, all features", BLUE),
                        ("size_only:total_reimb", "Total reimbursement alone", ORANGE),
                        ("expert_rules", "Expert rules", AQUA)]:
        p = [precision_recall_at_k(y, s[col].to_numpy(), k)[0] for k in ks]
        ax.plot(ks, p, color=c, lw=2, label=lab)
    ax.plot(ks, np.minimum(1, y.sum() / ks), color=MUTED, lw=1, ls="--", label=f"Ceiling ({int(y.sum())} flagged in test)")
    for k in (50, 100):
        ax.axvline(k, color=GRID, lw=1)
    ax.set_xlabel("Audit queue size k (test set, 1,082 providers)")
    ax.set_ylabel("Precision at k")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(10, 300)
    ax.legend(loc="lower left", fontsize=7.5, ncol=2)
    ax.set_title("Held-out test: the model's lead over size alone is largest in the first 100 audits")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_test_precision_at_k.png")


# ---------------------------------------------------------------- Fig 5 SHAP
def fig_shap():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, key, title in [(axes[0], "lightgbm", "(a) All features"), (axes[1], "lgbm_no_size", "(b) No size features")]:
        sh = pd.read_pickle(CACHE_DIR / f"shap_test_{key}.pkl")
        imp = sh.abs().mean().sort_values(ascending=True).tail(10)
        size_cols = set(FAM["size"])
        ax.barh(range(len(imp)), imp.values, color=[ORANGE if c in size_cols else BLUE for c in imp.index], height=0.6)
        ax.set_yticks(range(len(imp)), imp.index, fontsize=7.5)
        ax.set_xlabel("Mean |SHAP| on test providers (log-odds)")
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    axes[0].text(0.98, 0.04, "orange = size feature", transform=axes[0].transAxes, ha="right", fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_shap.png")


# ---------------------------------------------------------------- Fig 6 label noise
def fig_noise():
    rates = [0.0, 0.1, 0.2, 0.3, 0.5]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for fs, lab, c in [("all", "All features", BLUE), ("no_size", "No size features", ORANGE)]:
        m = [cvrow("E7", f"lgbm:{fs}|noise={r}")[0] for r in rates]
        ax.plot([r * 100 for r in rates], m, color=c, lw=2, marker="o", ms=4, label=lab)
    ax.axhline(cvrow("E0", "size_only:total_reimb")[0], color=MUTED, lw=1, ls="--")
    ax.text(1, cvrow("E0", "size_only:total_reimb")[0] + 0.002, "size alone", fontsize=7, color=INK2, va="bottom",
            ha="left")
    ax.set_xlabel("Flagged training providers relabelled 0 (%)")
    ax.set_ylabel("PR-AUC (clean validation)")
    ax.legend(fontsize=7.5)
    ax.set_title("Label noise")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_noise.png")


if __name__ == "__main__":
    for f in [fig_data, fig_models, fig_ablation, fig_test, fig_shap, fig_noise]:
        f()
        print("drew", f.__name__)
