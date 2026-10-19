"""Structured outputs for every agent. Agents choose; code executes.

Every field an agent can set is a closed set or a bounded number, so an agent can
only propose actions the executor knows how to run safely (lecture 05).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

CLAIM_COLUMNS = Literal["InscClaimAmtReimbursed", "los", "claim_days", "n_diag", "n_proc",
                        "OPDeductible", "age", "n_chronic", "RenalDiseaseIndicator", "bene_dead",
                        "dup_any_provider", "dup_same_provider", "att_is_op"]
PROVIDER_BASES = Literal["n_claims", "n_ip_claims", "n_op_claims", "n_bene", "total_reimb",
                         "ip_total_reimb", "op_total_reimb", "n_attending", "n_operating",
                         "n_states", "n_dgc", "n_admit_diag"]


# ---------------------------------------------------------------- 1 cleaning
class CleaningStep(BaseModel):
    column: str
    action: Literal["recode_binary", "drop_column", "missing_indicator", "parse_date", "keep", "impute"]
    detail: str = Field(description="e.g. '1->1, 2->0' or why it is dropped")
    reason: str


class CleaningPlan(BaseModel):
    steps: list[CleaningStep]


# ---------------------------------------------------------------- 2 features
class ClaimAggFeature(BaseModel):
    name: str = Field(pattern=r"^agent_[a-z0-9_]{3,40}$")
    column: CLAIM_COLUMNS
    agg: Literal["mean", "median", "max", "std", "share_positive"]
    scope: Literal["all", "ip", "op"]
    rationale: str


class RatioFeature(BaseModel):
    name: str = Field(pattern=r"^agent_[a-z0-9_]{3,40}$")
    numerator: PROVIDER_BASES
    denominator: PROVIDER_BASES
    rationale: str


class FeatureProposal(BaseModel):
    claim_aggregates: list[ClaimAggFeature] = Field(max_length=12)
    ratios: list[RatioFeature] = Field(max_length=8)


# ---------------------------------------------------------------- 3 leakage review
class FeatureVerdict(BaseModel):
    feature: str
    verdict: Literal["ok", "leak", "uncertain"]
    reason: str


class LeakageReview(BaseModel):
    verdicts: list[FeatureVerdict]
    split_ok: bool
    split_comment: str


# ---------------------------------------------------------------- 4-5 experiment loop
class ModelParams(BaseModel):
    C: Optional[float] = Field(None, ge=1e-4, le=100)
    min_samples_leaf: Optional[int] = Field(None, ge=1, le=50)
    max_features: Optional[float] = Field(None, ge=0.05, le=1.0)
    num_leaves: Optional[int] = Field(None, ge=4, le=64)
    learning_rate: Optional[float] = Field(None, ge=0.005, le=0.2)
    n_estimators: Optional[int] = Field(None, ge=50, le=1500)
    min_child_samples: Optional[int] = Field(None, ge=5, le=100)
    colsample_bytree: Optional[float] = Field(None, ge=0.3, le=1.0)


class ExperimentProposal(BaseModel):
    model: Literal["logistic", "random_forest", "lightgbm"]
    feature_set: Literal["all", "no_size", "all+agent", "no_size+agent"]
    imbalance: Literal["none", "class_weight", "smote"]
    params: ModelParams
    rationale: str


class ExperimentBatch(BaseModel):
    proposals: list[ExperimentProposal] = Field(min_length=3, max_length=4)


class CriticVerdict(BaseModel):
    chosen_run: int = Field(description="run id of the configuration to carry forward")
    stop: bool = Field(description="true if no further experiment is justified")
    concerns: list[str]
    next_experiments: list[ExperimentProposal] = Field(default_factory=list, max_length=3)


# ---------------------------------------------------------------- 6 case file
class Driver(BaseModel):
    feature: str
    value: float
    peer_percentile: float = Field(ge=0, le=100)
    shap: float


class TypologyCitation(BaseModel):
    chunk_id: str
    why: str


class CaseBrief(BaseModel):
    provider: str
    risk_rank: int
    summary: str = Field(description="3-5 sentences for an investigator; facts only from the evidence")
    drivers: list[Driver] = Field(min_length=1, max_length=5)
    typologies: list[TypologyCitation] = Field(max_length=3)
    recommended_action: Literal["priority_audit", "desk_review", "monitor"]
    uncertainty: str


# ---------------------------------------------------------------- 7 dashboard agent
ACTION = Literal["priority_audit", "desk_review", "monitor"]
CAVEAT_ID = Literal["suspicion_not_proof", "size_effect", "label_origin", "benchmark_data", "human_decision"]
REQUIRED_CAVEATS = ("suspicion_not_proof", "size_effect", "label_origin", "benchmark_data", "human_decision")


# Every agent-written field fills a fixed slot in the dashboard template. These limits (in words) keep the
# text inside its card; text over the limit is dropped by the checks and that slot shows system wording.
# (slot shown to the reader, word limit)
OVERVIEW_SLOTS = {
    "headline": ("the line under the greeting in the dark header at the top of the home page", 25),
    "summary": ("the first card of the green insights panel, titled 'This cycle'", 45),
    "actions.why": ("one card per recommendation in the green insights panel", 35),
    "how_to_read": ("the last card of the green insights panel, 'How to read the ranking'", 50),
    "watch_out_for": ("one bullet under 'Watch out for' in the 'Before you act' card", 25),
    "caveats.text": ("one bullet under 'Limits of this ranking' in the 'Before you act' card", 25),
    "glossary.meaning": ("one entry under 'Words used here'", 20),
}
PAGE_SLOTS = {
    "summary": ("the 'Summary' card, read first on the provider page", 60),
    "size_note": ("a card in the green insights panel on whether it is here for how it bills or for its size", 35),
    "beyond.explanation": ("the bottom of one tile under 'What stands out beyond size'", 35),
    "reasons.explanation": ("the right-hand column of one row under 'Why it ranks here overall'", 40),
    "what_to_check": ("one tickable row in the 'What to check' checklist", 25),
    "patterns.explanation": ("the text shown when one 'Patterns it resembles' row is opened", 30),
    "cautions": ("one bullet in the 'Keep in mind' card of the green insights panel", 25),
}


def slot_rules(slots: dict) -> str:
    """Prompt text telling the agent where each field appears and how long it may be."""
    return "Each field fills a fixed place on the page; stay within its word limit: " + "; ".join(
        f"{k} ({where}): at most {n} words" for k, (where, n) in slots.items()) + "."


class Caveat(BaseModel):
    caveat_id: CAVEAT_ID
    text: str = Field(description="one plain-English sentence, at most 25 words")


class ActionNote(BaseModel):
    action: ACTION
    why: str = Field(description="one or two plain sentences, at most 35 words, on why these providers got this "
                                 "recommendation")


class GlossaryItem(BaseModel):
    term: str
    meaning: str = Field(description="at most 20 words")


class DashboardOverview(BaseModel):
    headline: str = Field(description="one sentence, at most 25 words: the most important thing to know this cycle")
    summary: str = Field(description="2-3 short sentences, at most 45 words")
    actions: list[ActionNote] = Field(min_length=1, max_length=3)
    how_to_read: str = Field(description="2-3 sentences, at most 50 words, on what a rank means and does not mean")
    watch_out_for: list[str] = Field(min_length=2, max_length=5, description="each at most 25 words")
    caveats: list[Caveat] = Field(min_length=5, max_length=6)
    glossary: list[GlossaryItem] = Field(max_length=6)


class FactorNote(BaseModel):
    feature: str = Field(description="the feature key exactly as given in the evidence")
    explanation: str = Field(description="one or two plain sentences, at most 35 words: what this means for a reviewer")


class PatternNote(BaseModel):
    pattern_id: str
    explanation: str = Field(description="one sentence, at most 30 words, on how this provider's numbers match the "
                                         "pattern")


class ProviderPage(BaseModel):
    provider: str
    summary: str = Field(description="2-3 plain sentences, at most 60 words: why this provider is in the queue")
    size_note: str = Field(description="one sentence, at most 35 words: is it here because of how it bills, or "
                                       "mainly its size?")
    beyond: list[FactorNote] = Field(max_length=3)
    reasons: list[FactorNote] = Field(max_length=5)
    what_to_check: list[str] = Field(min_length=3, max_length=5,
                                     description="concrete review steps, each at most 25 words: which records to "
                                                 "request or compare")
    patterns: list[PatternNote] = Field(max_length=3)
    cautions: list[str] = Field(min_length=1, max_length=3, description="each at most 25 words")
