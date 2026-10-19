"""Step 0: download the Kaggle dataset into data/raw/ (skip if you placed the files by hand).

Uses kagglehub, which downloads the dataset once into its own cache and returns the folder.
The 8 CSVs are then copied into data/raw/ under their original names, and their sizes are
checked against the version the reported results were produced from.

If Kaggle asks you to sign in, create an API token (kaggle.com > Settings > API > Create New
Token) and either save kaggle.json to ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudrank.config import RAW_DIR  # noqa: E402

DATASET = "rohitrox/healthcare-provider-fraud-detection-analysis"
# file -> size in bytes of the copy used for the reported results
FILES = {
    "Train-1542865627584.csv": 87094,
    "Train_Beneficiarydata-1542865627584.csv": 11444821,
    "Train_Inpatientdata-1542865627584.csv": 8572555,
    "Train_Outpatientdata-1542865627584.csv": 77391090,
    "Test-1542969243754.csv": 14894,
    "Test_Beneficiarydata-1542969243754.csv": 5325982,
    "Test_Inpatientdata-1542969243754.csv": 2020712,
    "Test_Outpatientdata-1542969243754.csv": 18806934,
}


def have_all() -> bool:
    return all((RAW_DIR / n).exists() for n in FILES)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if have_all() and "--force" not in sys.argv:
        print(f"All 8 files already in {RAW_DIR}; nothing to do (use --force to download again).")
    else:
        try:
            import kagglehub
        except ImportError:
            sys.exit("kagglehub is not installed: pip install -r requirements.txt")
        src = Path(kagglehub.dataset_download(DATASET, force_download="--force" in sys.argv))
        print("downloaded to", src)
        for name in FILES:
            found = next(src.rglob(name), None)
            if found is None:
                sys.exit(f"{name} is not in the download; the dataset may have changed. "
                         "Download it by hand instead (README step 2).")
            shutil.copy2(found, RAW_DIR / name)
    changed = [n for n, size in FILES.items() if (RAW_DIR / n).stat().st_size != size]
    if changed:
        print("WARNING: these files differ in size from the version used for the reported results, "
              f"so numbers may not match exactly: {changed}")
    else:
        print(f"OK: 8 files in {RAW_DIR}, identical in size to the version used for the reported results.")


if __name__ == "__main__":
    main()
