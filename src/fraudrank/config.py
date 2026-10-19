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

for d in (PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)
