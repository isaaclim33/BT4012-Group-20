"""Aggregate claims to one row per provider, grouped into named feature families.

The label is assigned per provider, so the model works per provider too. Features
are split into families so that experiments can drop a family (FAMILIES) or keep only
size-free features. The SIZE family holds raw counts and totals that grow with a
provider's volume; every other family is a ratio, mean or share, so it describes
*how* a provider bills rather than *how much*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CHRONIC_COLS

FAMILIES: dict[str, list[str]] = {}


def _register(family: str, cols: list[str]) -> None:
    FAMILIES.setdefault(family, []).extend(cols)


def _safe_div(a, b):
    return np.where(np.asarray(b) > 0, np.asarray(a) / np.where(np.asarray(b) > 0, b, 1), 0.0)


def enrich_claims(claims: pd.DataFrame, bene: pd.DataFrame) -> pd.DataFrame:
    """Claim rows joined with beneficiary attributes and claim-level red-flag columns."""
    c = claims.merge(
        bene[["BeneID", "DOB", "DOD", "State", "County", "RenalDiseaseIndicator", "n_chronic",
              "IPAnnualReimbursementAmt", "OPAnnualReimbursementAmt"] + CHRONIC_COLS],
        on="BeneID", how="left")
    c["age"] = (c["ClaimStartDt"] - c["DOB"]).dt.days / 365.25
    c["bene_dead"] = c["DOD"].notna().astype(int)
    c["att_is_op"] = (c["AttendingPhysician"].notna()
                      & (c["AttendingPhysician"] == c["OperatingPhysician"])).astype(int)
    # duplicate-like claims: same beneficiary, start date, first diagnosis and amount
    key = ["BeneID", "ClaimStartDt", "ClmDiagnosisCode_1", "InscClaimAmtReimbursed"]
    c["dup_any_provider"] = c.duplicated(key, keep=False).astype(int)
    c["dup_same_provider"] = c.duplicated(key + ["Provider"], keep=False).astype(int)
    return c


def build_provider_features(claims: pd.DataFrame, bene: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    FAMILIES.clear()
    c = enrich_claims(claims, bene)
    g = c.groupby("Provider")
    ip = c[c.is_ip == 1].groupby("Provider")
    op = c[c.is_ip == 0].groupby("Provider")
    f = pd.DataFrame(index=labels["Provider"])

    # ---- SIZE: raw volumes and totals (grow with provider size)
    f["n_claims"] = g.size()
    f["n_ip_claims"] = ip.size()
    f["n_op_claims"] = op.size()
    f["n_bene"] = g["BeneID"].nunique()
    f["total_reimb"] = g["InscClaimAmtReimbursed"].sum()
    f["ip_total_reimb"] = ip["InscClaimAmtReimbursed"].sum()
    f["op_total_reimb"] = op["InscClaimAmtReimbursed"].sum()
    f["n_attending"] = g["AttendingPhysician"].nunique()
    f["n_operating"] = g["OperatingPhysician"].nunique()
    f["n_other_phys"] = g["OtherPhysician"].nunique()
    f["n_states"] = g["State"].nunique()
    f["n_dgc"] = ip["DiagnosisGroupCode"].nunique()
    f["n_admit_diag"] = g["ClmAdmitDiagnosisCode"].nunique()
    f = f.fillna(0)
    _register("size", list(f.columns))

    # ---- SERVICE MIX: what kind of care is billed
    f["has_ip"] = (f["n_ip_claims"] > 0).astype(int)
    f["has_op"] = (f["n_op_claims"] > 0).astype(int)
    f["ip_claim_share"] = _safe_div(f["n_ip_claims"], f["n_claims"])
    f["ip_reimb_share"] = _safe_div(f["ip_total_reimb"], f["total_reimb"])
    _register("service_mix", ["has_ip", "has_op", "ip_claim_share", "ip_reimb_share"])

    # ---- MONEY per unit (size-free)
    f["reimb_per_claim"] = _safe_div(f["total_reimb"], f["n_claims"])
    f["reimb_per_bene"] = _safe_div(f["total_reimb"], f["n_bene"])
    f["ip_reimb_mean"] = ip["InscClaimAmtReimbursed"].mean()
    f["ip_reimb_max"] = ip["InscClaimAmtReimbursed"].max()
    f["op_reimb_mean"] = op["InscClaimAmtReimbursed"].mean()
    f["op_reimb_median"] = op["InscClaimAmtReimbursed"].median()
    f["op_reimb_max"] = op["InscClaimAmtReimbursed"].max()
    f["op_deductible_mean"] = op["OPDeductible"].mean()
    f["zero_reimb_share"] = g["InscClaimAmtReimbursed"].apply(lambda s: (s == 0).mean())
    _register("money", ["reimb_per_claim", "reimb_per_bene", "ip_reimb_mean", "ip_reimb_max",
                        "op_reimb_mean", "op_reimb_median", "op_reimb_max",
                        "op_deductible_mean", "zero_reimb_share"])

    # ---- INTENSITY: length of stay and claim duration
    f["los_mean"] = ip["los"].mean()
    f["los_max"] = ip["los"].max()
    f["los_long_share"] = ip["los"].apply(lambda s: (s >= 10).mean())
    f["ip_claim_days_mean"] = ip["claim_days"].mean()
    f["op_claim_days_mean"] = op["claim_days"].mean()
    f["claims_per_bene"] = _safe_div(f["n_claims"], f["n_bene"])
    _register("intensity", ["los_mean", "los_max", "los_long_share", "ip_claim_days_mean",
                            "op_claim_days_mean", "claims_per_bene"])

    # ---- CODING: breadth and depth of diagnosis / procedure coding
    f["diag_per_claim"] = g["n_diag"].mean()
    f["ip_proc_per_claim"] = ip["n_proc"].mean()
    f["ip_surgery_share"] = ip["n_proc"].apply(lambda s: (s > 0).mean())
    f["dgc_per_ip_claim"] = _safe_div(f["n_dgc"], f["n_ip_claims"])
    f["admit_diag_per_claim"] = _safe_div(f["n_admit_diag"], f["n_claims"])
    f["distinct_diag1_per_claim"] = _safe_div(g["ClmDiagnosisCode_1"].nunique(), f["n_claims"])
    _register("coding", ["diag_per_claim", "ip_proc_per_claim", "ip_surgery_share",
                         "dgc_per_ip_claim", "admit_diag_per_claim", "distinct_diag1_per_claim"])

    # ---- PHYSICIANS: concentration of billing across doctors
    f["claims_per_attending"] = _safe_div(f["n_claims"], f["n_attending"])
    top_att = c.groupby(["Provider", "AttendingPhysician"]).size().groupby("Provider").max()
    f["top_attending_share"] = _safe_div(top_att.reindex(f.index).fillna(0), f["n_claims"])
    f["operating_present_share"] = g["OperatingPhysician"].apply(lambda s: s.notna().mean())
    f["other_present_share"] = g["OtherPhysician"].apply(lambda s: s.notna().mean())
    f["att_is_op_share"] = g["att_is_op"].mean()
    _register("physicians", ["claims_per_attending", "top_attending_share", "operating_present_share",
                             "other_present_share", "att_is_op_share"])

    # ---- DUPLICATION: claims repeated on the same day for the same patient
    f["dup_any_share"] = g["dup_any_provider"].mean()
    f["dup_same_share"] = g["dup_same_provider"].mean()
    _register("duplication", ["dup_any_share", "dup_same_share"])

    # ---- PATIENT MIX: who the provider treats
    f["age_mean"] = g["age"].mean()
    f["dead_share"] = g["bene_dead"].mean()
    f["n_chronic_mean"] = g["n_chronic"].mean()
    f["renal_share"] = g["RenalDiseaseIndicator"].mean()
    f["bene_ip_annual_mean"] = g["IPAnnualReimbursementAmt"].mean()
    f["bene_op_annual_mean"] = g["OPAnnualReimbursementAmt"].mean()
    f["states_per_bene"] = _safe_div(f["n_states"], f["n_bene"])
    for col in CHRONIC_COLS:
        f[f"share_{col.split('_', 1)[1].lower()}"] = g[col].mean()
    _register("patient_mix", ["age_mean", "dead_share", "n_chronic_mean", "renal_share",
                              "bene_ip_annual_mean", "bene_op_annual_mean", "states_per_bene"]
              + [f"share_{col.split('_', 1)[1].lower()}" for col in CHRONIC_COLS])

    # Inpatient-only statistics are undefined for outpatient-only providers. Fill with 0
    # and rely on has_ip to tell the model they are structural, not measured zeros.
    f = f.fillna(0).astype(float)
    f = f.join(labels.set_index("Provider")["y"])
    return f


def feature_columns(exclude: tuple[str, ...] = ()) -> list[str]:
    return [c for fam, cols in FAMILIES.items() if fam not in exclude for c in cols]
