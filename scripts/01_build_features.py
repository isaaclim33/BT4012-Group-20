"""Step 1: clean the raw files, build the provider feature table and freeze the splits.

Outputs (data/processed/): provider_features.pkl, families.json, splits.json,
and results/tables/data_quality.json.
"""
import json
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudrank import data, features, splits  # noqa: E402
from fraudrank.config import PROCESSED_DIR, TABLES_DIR  # noqa: E402

labels = data.load_labels()
bene = data.load_beneficiaries()
claims = data.load_claims()

quality = data.quality_report(claims, bene, labels)
(TABLES_DIR / "data_quality.json").write_text(json.dumps(quality, indent=2))
print(json.dumps(quality, indent=2))

prov = features.build_provider_features(claims, bene, labels)
prov.to_pickle(PROCESSED_DIR / "provider_features.pkl")
(PROCESSED_DIR / "families.json").write_text(json.dumps(features.FAMILIES, indent=2))
print("feature table", prov.shape, "families", {k: len(v) for k, v in features.FAMILIES.items()})

# Slim claim table for the agent feature executor (no labels involved)
from fraudrank.agents.schemas import CLAIM_COLUMNS  # noqa: E402
enriched = features.enrich_claims(claims, bene)
enriched[["Provider", "is_ip", *CLAIM_COLUMNS.__args__]].to_pickle(PROCESSED_DIR / "claims_enriched.pkl")

# Provider-beneficiary edges, used by the network features (no labels involved)
claims[["Provider", "BeneID"]].drop_duplicates().to_pickle(PROCESSED_DIR / "provider_bene_edges.pkl")
claims[["Provider", "AttendingPhysician"]].dropna().drop_duplicates().to_pickle(
    PROCESSED_DIR / "provider_physician_edges.pkl")

# Descriptive facts quoted in the report's data section
pb = claims[["Provider", "BeneID"]].drop_duplicates()
ptype = np.select([(prov.has_ip == 1) & (prov.has_op == 1), prov.has_ip == 1], ["both", "ip_only"], "op_only")
desc = {
    "median_claims_flagged": float(prov.loc[prov.y == 1, "n_claims"].median()),
    "median_claims_not_flagged": float(prov.loc[prov.y == 0, "n_claims"].median()),
    "flagged_share_of_claims": float(prov.loc[prov.y == 1, "n_claims"].sum() / prov.n_claims.sum()),
    "flagged_share_of_reimbursement": float(prov.loc[prov.y == 1, "total_reimb"].sum() / prov.total_reimb.sum()),
    "precision_top50_by_claims": float(prov.sort_values("n_claims", ascending=False).head(50).y.mean()),
    "beneficiaries_seeing_2plus_providers": float((pb.groupby("BeneID").size() > 1).mean()),
    "flag_rate_by_type": prov.assign(t=ptype).groupby("t").y.agg(["mean", "sum", "size"]).to_dict("index"),
    "ip_operating_physician_missing": float(claims.loc[claims.is_ip == 1, "OperatingPhysician"].isna().mean()),
    "op_procedure_code_missing": float(claims.loc[claims.is_ip == 0, "ClmProcedureCode_1"].isna().mean()),
}
(TABLES_DIR / "data_descriptives.json").write_text(json.dumps(desc, indent=2))
print(json.dumps(desc, indent=1))

s = splits.make_splits(prov)
print("dev", len(s["dev"]), "test", len(s["test"]), "test positives",
      int(prov.loc[s["test"], "y"].sum()), "folds", len(s["folds"]))
