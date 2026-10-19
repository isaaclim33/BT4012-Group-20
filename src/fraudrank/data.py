"""Load and clean the Kaggle Healthcare Provider Fraud Detection files.

Cleaning decisions (each justified in the report, section 3.1):
* Chronic-condition flags and gender are coded 1/2 in the raw files; recode to 1/0.
* RenalDiseaseIndicator is coded '0'/'Y'; recode to 0/1.
* The inpatient deductible is $1,068 on every non-blank claim, so it carries no
  signal and is dropped.
* Missing physician, procedure and diagnosis fields are structural (an outpatient
  visit has no admission or surgery), so they are counted, never imputed.
"""
from __future__ import annotations

import pandas as pd

from .config import RAW_DIR, TRAIN_FILES

CHRONIC_COLS = [
    "ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
    "ChronicCond_Diabetes", "ChronicCond_IschemicHeart", "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke",
]
DIAG_COLS = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
PROC_COLS = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
DATE_COLS = ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / TRAIN_FILES[name])


def load_labels() -> pd.DataFrame:
    lab = _read("labels")
    lab["y"] = (lab["PotentialFraud"] == "Yes").astype(int)
    return lab[["Provider", "y"]]


def load_beneficiaries() -> pd.DataFrame:
    b = _read("beneficiary")
    for c in CHRONIC_COLS:
        b[c] = (b[c] == 1).astype(int)          # 1 = has condition, 2 = does not
    b["Gender"] = (b["Gender"] == 1).astype(int)
    b["RenalDiseaseIndicator"] = (b["RenalDiseaseIndicator"] == "Y").astype(int)
    b["DOB"] = pd.to_datetime(b["DOB"])
    b["DOD"] = pd.to_datetime(b["DOD"])
    b["n_chronic"] = b[CHRONIC_COLS].sum(axis=1)
    return b


def load_claims() -> pd.DataFrame:
    """Inpatient and outpatient claims stacked into one frame with an is_ip flag."""
    ip = _read("inpatient")
    op = _read("outpatient")
    ip["is_ip"] = 1
    op["is_ip"] = 0
    claims = pd.concat([ip, op], ignore_index=True, sort=False)
    for c in DATE_COLS:
        claims[c] = pd.to_datetime(claims[c])
    claims = claims.drop(columns=["DeductibleAmtPaid"]).assign(
        OPDeductible=pd.concat([pd.Series(0.0, index=ip.index),
                                op["DeductibleAmtPaid"].fillna(0)], ignore_index=True)
    )
    claims["n_diag"] = claims[DIAG_COLS].notna().sum(axis=1)
    claims["n_proc"] = claims[PROC_COLS].notna().sum(axis=1)
    claims["claim_days"] = (claims["ClaimEndDt"] - claims["ClaimStartDt"]).dt.days + 1
    claims["los"] = (claims["DischargeDt"] - claims["AdmissionDt"]).dt.days
    return claims


def quality_report(claims: pd.DataFrame, bene: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Integrity checks reported in the data section; all should be zero."""
    merged = claims.merge(bene[["BeneID", "DOD"]], on="BeneID", how="left")
    return {
        "n_claims": len(claims),
        "n_inpatient": int(claims["is_ip"].sum()),
        "n_outpatient": int((1 - claims["is_ip"]).sum()),
        "n_beneficiaries": bene["BeneID"].nunique(),
        "n_providers": labels["Provider"].nunique(),
        "n_flagged": int(labels["y"].sum()),
        "duplicate_claim_ids": int(claims["ClaimID"].duplicated().sum()),
        "end_before_start": int((claims["ClaimEndDt"] < claims["ClaimStartDt"]).sum()),
        "discharge_before_admission": int((claims["los"] < 0).sum()),
        "claims_after_death": int((merged["ClaimStartDt"] > merged["DOD"]).sum()),
        "claims_without_beneficiary": int(merged["DOD"].isna().sum()
                                          - merged["BeneID"].isin(bene.loc[bene["DOD"].isna(), "BeneID"]).sum()),
        "providers_without_label": int((~claims["Provider"].isin(labels["Provider"])).sum()),
        "first_claim": str(claims["ClaimStartDt"].min().date()),
        "last_claim": str(claims["ClaimEndDt"].max().date()),
        "ip_reimb_multiple_of_1000": float((claims.loc[claims.is_ip == 1, "InscClaimAmtReimbursed"] % 1000 == 0).mean()),
        "op_reimb_multiple_of_10": float((claims.loc[claims.is_ip == 0, "InscClaimAmtReimbursed"] % 10 == 0).mean()),
    }
