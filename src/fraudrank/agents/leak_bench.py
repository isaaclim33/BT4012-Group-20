"""E10: a benchmark of feature specifications with known leaks.

Ten clean specifications come from the real feature table; five leaky ones are
planted with plausible-sounding descriptions. The Leakage agent sees only the
names and descriptions, in a fixed shuffled order, and its verdicts are scored
against this ground truth.
"""
import random

CLEAN = [
    ("reimb_per_claim", "total reimbursement divided by number of claims, per provider"),
    ("los_mean", "mean inpatient length of stay over the provider's inpatient claims"),
    ("dup_same_share", "share of the provider's claims that match another of its claims on patient, date, diagnosis and amount"),
    ("top_attending_share", "share of the provider's claims carrying its most frequent attending physician"),
    ("n_chronic_mean", "mean number of chronic conditions of the provider's beneficiaries"),
    ("diag_per_claim", "mean number of diagnosis codes per claim"),
    ("net_degree", "number of other providers sharing at least one beneficiary, from claims of all providers"),
    ("net_nbr_fraud_share", "share of neighbouring providers flagged, where flags come only from training-fold providers and each training provider's own fold is excluded"),
    ("ip_claim_share", "inpatient claims divided by all claims, per provider"),
    ("claims_per_bene", "claims divided by distinct beneficiaries, per provider"),
]
LEAKS = [
    ("provider_flag_rate_smoothed", "PotentialFraud label of the provider smoothed with the global flag rate (target encoding fitted on all providers)"),
    ("state_flag_rate", "share of providers flagged in the provider's main state, computed over all 5,410 labelled providers"),
    ("ppr_all_labels", "personalised PageRank seeded on every flagged provider in the dataset, including validation and test providers"),
    ("audit_opened_flag", "1 if an investigation case was opened for the provider after the review period"),
    ("reimb_zscore_global", "provider total reimbursement standardised using the mean and SD of all providers including the test set, then multiplied by the provider's label rank"),
]


def benchmark(seed: int = 7) -> tuple[list[str], dict[str, bool]]:
    items = [(n, d, False) for n, d in CLEAN] + [(n, d, True) for n, d in LEAKS]
    random.Random(seed).shuffle(items)
    specs = [f"{n}: {d}" for n, d, _ in items]
    truth = {n: leak for n, _, leak in items}
    return specs, truth


def score(review, truth: dict[str, bool]) -> dict:
    v = {x.feature.split(":")[0].strip(): x.verdict for x in review.verdicts}
    tp = sum(truth[n] and v.get(n) == "leak" for n in truth)
    fp = sum((not truth[n]) and v.get(n) == "leak" for n in truth)
    unc = sum(v.get(n) == "uncertain" for n in truth)
    missing = sum(n not in v for n in truth)
    return {"leaks": sum(truth.values()), "leaks_caught": tp, "clean": len(truth) - sum(truth.values()),
            "false_alarms": fp, "uncertain": unc, "missing_verdicts": missing}
