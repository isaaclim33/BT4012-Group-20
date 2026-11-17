"""What a feature-proposing agent is told about the features that already exist.

Each existing feature is defined in the same operations a proposal can use (an aggregate of a
claim column over all, inpatient or outpatient claims, or a ratio of two provider bases), so the
agent can see that, say, n_claims / n_bene is already claims_per_bene.
"""
from __future__ import annotations

from ..data import CHRONIC_COLS

OPERATIONS = {
    "claim_aggregate": "mean, median, max, std or share_positive (share of claims with a value above 0) of one "
                       "claim column, over all, ip (inpatient) or op (outpatient) claims",
    "ratio": "one provider base divided by another",
    "duplicates": "a proposal that restates an existing feature (the same aggregate or ratio, the inverse of a "
                  "ratio, or a ratio that ranks providers the same way) adds no information and is rejected",
}

DEFINITIONS = {
    "n_claims": "provider base n_claims: number of claims",
    "n_ip_claims": "provider base n_ip_claims: number of inpatient claims",
    "n_op_claims": "provider base n_op_claims: number of outpatient claims",
    "n_bene": "provider base n_bene: number of distinct patients",
    "total_reimb": "provider base total_reimb: sum of InscClaimAmtReimbursed over all claims",
    "ip_total_reimb": "provider base ip_total_reimb: sum of InscClaimAmtReimbursed over ip claims",
    "op_total_reimb": "provider base op_total_reimb: sum of InscClaimAmtReimbursed over op claims",
    "n_attending": "provider base n_attending: number of distinct attending physicians",
    "n_operating": "provider base n_operating: number of distinct operating physicians",
    "n_other_phys": "number of distinct other physicians",
    "n_states": "provider base n_states: number of distinct patient states",
    "n_dgc": "provider base n_dgc: number of distinct diagnosis-related groups on ip claims",
    "n_admit_diag": "provider base n_admit_diag: number of distinct admission diagnosis codes",
    "has_ip": "1 if the provider has any ip claim",
    "has_op": "1 if the provider has any op claim",
    "ip_claim_share": "n_ip_claims / n_claims",
    "ip_reimb_share": "ip_total_reimb / total_reimb",
    "reimb_per_claim": "total_reimb / n_claims (= mean of InscClaimAmtReimbursed over all claims)",
    "reimb_per_bene": "total_reimb / n_bene",
    "ip_reimb_mean": "mean of InscClaimAmtReimbursed over ip claims",
    "ip_reimb_max": "max of InscClaimAmtReimbursed over ip claims",
    "op_reimb_mean": "mean of InscClaimAmtReimbursed over op claims",
    "op_reimb_median": "median of InscClaimAmtReimbursed over op claims",
    "op_reimb_max": "max of InscClaimAmtReimbursed over op claims",
    "op_deductible_mean": "mean of OPDeductible over op claims",
    "zero_reimb_share": "share of all claims with InscClaimAmtReimbursed = 0 (1 - share_positive)",
    "los_mean": "mean of los over ip claims",
    "los_max": "max of los over ip claims",
    "los_long_share": "share of ip claims with los of 10 days or more",
    "ip_claim_days_mean": "mean of claim_days over ip claims",
    "op_claim_days_mean": "mean of claim_days over op claims",
    "claims_per_bene": "n_claims / n_bene",
    "diag_per_claim": "mean of n_diag over all claims",
    "ip_proc_per_claim": "mean of n_proc over ip claims",
    "ip_surgery_share": "share_positive of n_proc over ip claims",
    "dgc_per_ip_claim": "n_dgc / n_ip_claims",
    "admit_diag_per_claim": "n_admit_diag / n_claims",
    "distinct_diag1_per_claim": "number of distinct primary diagnosis codes / n_claims",
    "claims_per_attending": "n_claims / n_attending",
    "top_attending_share": "claims of the busiest attending physician / n_claims",
    "operating_present_share": "share of all claims naming an operating physician",
    "other_present_share": "share of all claims naming an other physician",
    "att_is_op_share": "mean (= share_positive) of att_is_op over all claims",
    "dup_any_share": "mean (= share_positive) of dup_any_provider over all claims",
    "dup_same_share": "mean (= share_positive) of dup_same_provider over all claims",
    "age_mean": "mean of age over all claims",
    "dead_share": "mean (= share_positive) of bene_dead over all claims",
    "n_chronic_mean": "mean of n_chronic over all claims",
    "renal_share": "mean (= share_positive) of RenalDiseaseIndicator over all claims",
    "bene_ip_annual_mean": "mean over all claims of the patient's annual inpatient reimbursement",
    "bene_op_annual_mean": "mean over all claims of the patient's annual outpatient reimbursement",
    "states_per_bene": "n_states / n_bene",
    **{f"share_{c.split('_', 1)[1].lower()}": f"share of all claims whose patient has {c.split('_', 1)[1]}"
       for c in CHRONIC_COLS},
}


def feature_brief(families: dict[str, list[str]]) -> dict:
    """Existing features by family, each with its definition, plus the operations a proposal may use."""
    missing = [c for cols in families.values() for c in cols if c not in DEFINITIONS]
    assert not missing, f"no definition for {missing}"
    return {"existing_features": {fam: {c: DEFINITIONS[c] for c in cols} for fam, cols in families.items()},
            "operations": OPERATIONS}
