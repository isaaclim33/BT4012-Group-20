"""Central configuration: paths, seeds and split settings.

Every experiment reads from here so that one change (e.g. the seed) propagates
everywhere and results stay reproducible.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = RESULTS_DIR / "cache"

TRAIN_FILES = {
    "labels": "Train-1542865627584.csv",
    "beneficiary": "Train_Beneficiarydata-1542865627584.csv",
    "inpatient": "Train_Inpatientdata-1542865627584.csv",
    "outpatient": "Train_Outpatientdata-1542865627584.csv",
}

SEED = 42
TEST_SIZE = 0.20          # held-out provider test set, opened once
CV_SPLITS = 5             # folds per repeat on the development set
CV_REPEATS = 5            # repeats -> 25 fits per configuration
INNER_SPLITS = 5          # inner folds for out-of-fold network features
BOOTSTRAP = 2000          # bootstrap resamples for test-set CIs
K_VALUES = (50, 100)      # absolute audit capacities on the test set
K_FRACTIONS = (0.05, 0.10)  # audit capacity as a share of providers

# Two-gate audit selection (see audit_selection.py). Rule thresholds were set on
# development providers only; the test set applies them once.
RULE_DUPLICATE_MIN = 1    # gate 1: claims repeating an earlier claim (patient, date, amount, diagnosis)
RULE_OVERLAP_MIN = 2      # gate 1: inpatient stays starting before the same patient's earlier stay ended
RISK_QUEUE_SIZE = 50      # gate 2: highest risk scores among providers gate 1 did not refer (the report's k=50)
BATCH_SIZE = 12           # dashboard: the queue is released to reviewers in batches of this size
VOTE_SEEDS = (42, 0, 1, 2, 3, 4, 5, 6, 7, 8)   # size-free models retrained with these seeds
VOTE_AUDIT = 8            # full audit if at least this many put the provider in their size-free top RISK_QUEUE_SIZE
VOTE_BORDERLINE = 3       # desk review, borderline, from here up to VOTE_AUDIT - 1

for d in (PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)
