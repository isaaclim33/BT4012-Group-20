"""Small regression tests, independent of Kaggle downloads and model caches."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from fraudrank import audit_selection as AS
from fraudrank.agents import llm
from fraudrank.agents.dashboard import render
from fraudrank.agents.grounding import _row, evidence_items, check_item
from fraudrank.agents.selection import fold_test
from fraudrank.metrics import bootstrap_ci, evaluate, paired_bootstrap_diff

spec = importlib.util.spec_from_file_location("rebuild_offline", ROOT / "scripts/rebuild_offline.py")
offline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline)


def claims_fixture():
    c = pd.DataFrame({"Provider": ["P"] * 3, "BeneID": ["B"] * 3, "ClaimID": ["A", "B", "C"],
                      "ClaimStartDt": ["2009-01-01", "2009-01-02", "2009-01-04"],
                      "ClaimEndDt": ["2009-01-10", "2009-01-03", "2009-01-05"],
                      "InscClaimAmtReimbursed": [100, 200, 300], "ClmDiagnosisCode_1": ["a", "b", "c"]})
    for col in ("ClaimStartDt", "ClaimEndDt"):
        c[col] = pd.to_datetime(c[col])
    return c.assign(AdmissionDt=c.ClaimStartDt, DischargeDt=c.ClaimEndDt)


class BillingTests(unittest.TestCase):
    def test_nested_stays_have_matching_exhibits_independent_of_row_order(self):
        c = claims_fixture()
        expected = AS.billing_exhibits(["P"], c)
        for seed in range(5):
            shuffled = c.sample(frac=1, random_state=seed).reset_index(drop=True)
            checks = AS.billing_checks(pd.Index(["P"]), shuffled)
            self.assertEqual(checks.loc["P", "overlapping_stays"], 2)
            self.assertEqual(AS.billing_exhibits(["P"], shuffled), expected)
        self.assertEqual([p["earlier_stay"]["ClaimID"] for p in expected["P"]["overlaps"]], ["A", "A"])

    def test_touching_stays_and_different_patients_do_not_overlap(self):
        c = claims_fixture().iloc[:2].copy()
        c.loc[1, "AdmissionDt"] = c.loc[0, "DischargeDt"]
        self.assertEqual(list(AS.overlap_pairs(c)), [])
        c = claims_fixture().iloc[:2].copy()
        c.loc[1, "BeneID"] = "OTHER"
        self.assertEqual(list(AS.overlap_pairs(c)), [])

    def test_duplicate_null_diagnosis_and_more_than_two_claims(self):
        c = pd.concat([claims_fixture().iloc[:1]] * 3, ignore_index=True)
        c.ClaimID = ["A", "B", "C"]
        c.ClmDiagnosisCode_1 = None
        self.assertEqual(AS.billing_checks(pd.Index(["P"]), c).loc["P", "duplicate_claims"], 2)
        exhibits = AS.billing_exhibits(["P"], c)["P"]["duplicates"]
        self.assertEqual(len(exhibits), 1)
        self.assertIn("billed 3 times", exhibits[0]["summary"])


class RankingTests(unittest.TestCase):
    def test_ties_across_cutoff_and_seed_votes_are_order_independent(self):
        ids = [f"P{i:03}" for i in range(60)]
        scores = pd.Series(.5, index=ids)
        checks = pd.DataFrame(0, index=ids, columns=["duplicate_claims", "overlapping_stays"])
        votes = AS.seed_votes({42: scores, 0: scores.iloc[::-1]})
        self.assertEqual(votes.loc[ids[:50]].tolist(), [2] * 50)
        self.assertEqual(votes.loc[ids[50:]].tolist(), [0] * 10)
        q = AS.two_gate_selection(scores, checks, votes)
        pd.testing.assert_frame_equal(q, AS.two_gate_selection(scores.iloc[::-1], checks, votes))
        self.assertEqual(q.provider.tolist(), ids[:50])
        self.assertEqual(q.risk_rank.tolist(), list(range(1, 51)))

    def test_bootstrap_ties_preserve_provider_order_not_label_order(self):
        y = np.array([0] * 90 + [1] * 10)
        scores = np.ones(100)
        self.assertEqual(evaluate(y, scores)["prec_k50"], 0)
        self.assertEqual(bootstrap_ci(y, scores, "prec_k50", n=100), (0, 0))
        self.assertEqual(paired_bootstrap_diff(y, scores, scores, "prec_k50", n=100), (0, 0, 0))
        self.assertEqual(set(evaluate(y, scores, {})), {"pr_auc", "roc_auc"})


class GroundingTests(unittest.TestCase):
    def test_invented_counts_wrong_units_and_wrong_fields_fail(self):
        for text, source in [
            ("There are 10 duplicate claims.", {"duplicate_claims": 0}),
            ("There are two duplicate claims.", {"duplicate_claims": 0}),
            ("Paid $3.21B.", {"paid": "$3.21M"}),
            ("The provider had 200 claims.", {"claims": 20, "patients": 200}),
        ]:
            self.assertFalse(_row("P", "summary", text, source)["passed"], text)

    def test_review_instructions_and_field_bound_facts(self):
        self.assertTrue(_row("P", "what_to_check[0]", "Sample two claims.", {})["passed"])
        self.assertTrue(_row("P", "what_to_check[0]", "Review the medical records.", {})["passed"])
        self.assertFalse(_row("P", "what_to_check[0]", "Review 10 duplicate claims.", {})["passed"])
        source = {"beyond_size": [{"feature": "los_mean", "name": "Average stay", "value": "3 days",
                                   "comparison": "higher than 80% of peers", "effect": "pushes rank up"}]}
        self.assertTrue(_row("P", "beyond[los_mean]", "Average stay: 3 days, higher than 80% of peers.", source)["passed"])
        self.assertFalse(_row("P", "beyond[los_mean]", "Average stay: 80 days, higher than 3% of peers.", source)["passed"])

    def test_money_separator_does_not_truncate_thousands(self):
        ev = {"billing_checks": {"duplicate_claims": 0, "overlapping_hospital_stays": 0},
              "key_measures": {"Total payments": "$9,999, higher than 50% of peers"}, "beyond_size": []}
        item = evidence_items(ev)[0]
        self.assertEqual(item["text"], "$9,999 paid")
        self.assertTrue(check_item(item, ev)["passed"])


class StatisticsTests(unittest.TestCase):
    def test_correlated_folds_do_not_count_as_independent(self):
        from scipy.stats import t
        d = np.array([-.03, .01, -.02, .004, -.01] * 4)
        f = pd.DataFrame([{"pipeline": p, "run": 1, "fold": i, "pr_auc": .7 + v}
                          for p, ds in [("one", d), ("three", np.zeros(20))] for i, v in enumerate(ds)])
        result = fold_test(f, "one", "three")
        half = t.ppf(.975, 19) * np.sqrt((1/20 + 1/4) * np.var(d, ddof=1))
        self.assertAlmostEqual(result["ci_hi"], d.mean() + half)
        self.assertEqual(result["decision"], "fold into one agent")
        with self.assertRaises(ValueError):
            fold_test(f.iloc[:-1], "one", "three")


class CacheTests(unittest.TestCase):
    def test_cache_validation_rejects_stale_labels_and_feature_values(self):
        from fraudrank.config import VOTE_SEEDS
        df = pd.DataFrame({"f": [1., 2., 3.], "y": [0, 1, 0]}, index=["D", "A", "B"])
        splits = {"dev": ["D"], "test": ["A", "B"]}
        scores = pd.DataFrame({"y": [1, 0], "lgbm:no_size": [.8, .2]}, index=["A", "B"])
        seeds = pd.DataFrame({s: [.8, .2] for s in VOTE_SEEDS}, index=scores.index)
        X = df.loc[scores.index, ["f"]]
        shap = X * .1
        offline.validate_caches(df, splits, scores, seeds, X, shap, shap)
        with self.assertRaisesRegex(ValueError, "labels"):
            offline.validate_caches(df, splits, scores.assign(y=[0, 1]), seeds, X, shap, shap)
        with self.assertRaisesRegex(ValueError, "features"):
            offline.validate_caches(df, splits, scores, seeds, X + 1, shap, shap)

    def test_brief_identity_and_driver_checks_are_enforced(self):
        from fraudrank.agents.case_file import check_brief
        from fraudrank.agents.schemas import CaseBrief
        driver = {"feature": "f", "value": 2., "peer_percentile": 70., "shap": .1}
        ev = {"provider": "P", "risk_rank": 1, "drivers": [driver], "playbook": []}
        brief = CaseBrief(provider="P", risk_rank=1, summary="Review required.", drivers=[driver],
                          typologies=[], recommended_action="desk_review", action_reason="Review records.",
                          uncertainty="A flag is not proof.")
        self.assertTrue(check_brief(brief, ev)["passed"])
        self.assertFalse(check_brief(brief.model_copy(update={"provider": "Q"}), ev)["passed"])
        self.assertFalse(check_brief(brief.model_copy(update={"risk_rank": 2}), ev)["passed"])
        bad = brief.model_copy(deep=True)
        bad.drivers[0].value = 99
        self.assertFalse(check_brief(bad, ev)["passed"])

    def test_legacy_cache_replay_only_for_original_generation_defaults(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(llm, "LLM_CACHE", Path(tmp)), patch.object(llm, "TEMPERATURE", None):
            client = llm.LLMClient(mode="replay")
            legacy = {"model": client.model, "system": "s", "messages": [], "schema": {}, "effort": client.effort}
            (Path(tmp) / f"{client._key(legacy)}.json").write_text(json.dumps({"output": {"ok": True}}))
            self.assertEqual(client._raw("s", [], {}), {"ok": True})
            client.max_tokens = 100
            with self.assertRaises(llm.LLMError):
                client._raw("s", [], {})
            client.max_tokens = 16000
            with patch.object(llm, "TEMPERATURE", "0.2"), self.assertRaises(llm.LLMError):
                client._raw("s", [], {})

    def test_offline_client_cannot_access_api_even_with_key_present(self):
        with patch.dict("os.environ", {"FRAUDRANK_OFFLINE": "1", "GEMINI_API_KEY": "dummy"}):
            with self.assertRaises(llm.LLMError):
                llm.LLMClient()._api()

    def test_cycle_identity_stable_across_dates_and_changes_with_evidence(self):
        import re
        def identity(data, date):
            html = render(data, None, {"generated": date, "cycle": "one"})
            payload = re.search(r'<script id="dashboard-data" type="application/json">(.*?)</script>', html, re.S)[1]
            return json.loads(payload)["meta"]["storage_cycle"]
        self.assertEqual(identity({"providers": ["P"]}, "a"), identity({"providers": ["P"]}, "b"))
        self.assertNotEqual(identity({"providers": ["P"]}, "a"), identity({"providers": ["Q"]}, "a"))


class EdaTests(unittest.TestCase):
    def test_held_out_data_cannot_change_development_statistics(self):
        from contextlib import redirect_stdout
        import io
        from fraudrank.eda import run_eda
        cols = ["total_reimb", "n_claims", "reimb_per_claim", "claims_per_bene", "los_max",
                "diag_per_claim", "n_chronic_mean", "age_mean"]
        df = pd.DataFrame({c: np.arange(1, 11, dtype=float) for c in cols}, index=[f"P{i}" for i in range(10)])
        df["y"] = [0, 1] * 5
        df["has_ip"] = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        df["has_op"] = 1
        claims = pd.DataFrame({"Provider": np.repeat(df.index, 2), "is_ip": [0, 1] * 10,
                               "InscClaimAmtReimbursed": list(range(100, 120))})
        splits = {"dev": list(df.index[:8]), "test": list(df.index[8:])}
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            out = Path(tmp)
            run_eda(df, splits, out, claims)
            original = {p.name: p.read_bytes() for p in out.iterdir()}
            df.loc[splits["test"], cols] = 1e12
            df.loc[splits["test"], "y"] = 1
            claims.loc[claims.Provider.isin(splits["test"]), "InscClaimAmtReimbursed"] = 9999999
            run_eda(df, splits, out, claims)
            self.assertEqual(original, {p.name: p.read_bytes() for p in out.iterdir()})


if __name__ == "__main__":
    unittest.main()
