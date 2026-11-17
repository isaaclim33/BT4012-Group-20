"""One command to run the agent experiments on your own machine and pack the results.

    python scripts/run_agents_local.py

What it does:
 1. checks Python packages, the 8 Kaggle CSVs in data/raw/ and the API key in .env;
 2. rebuilds what the agents depend on if it is missing (feature table, tuned models,
    the held-out test scores and SHAP values): about 10 minutes;
 3. runs scripts/03_run_agents.py (RETRIEVAL, CLEAN, LEAK_AGENT, AGENT_MODEL, QUEUE, BRIEFS, DASHBOARD): about 520
    API calls, more than the free tier's 500 requests a day, so a full run spans two days and can take several
    hours when Gemini is busy. Most calls come from the evaluation design (three runs of each experiment, briefs
    with and without retrieval); one production cycle with the chosen design is about 125 calls. Finished calls
    are cached, so re-running after a quota stop or a failed call carries on where it stopped;
 4. writes agent_outputs_group20.zip in the repo root: the agent result tables, the
    cached LLM responses (prompts and answers, no key), the run log and package
    versions. Every file is scanned for the key before the zip is written.

Options: --full also re-runs every non-agent experiment (adds about 10 minutes);
--model to override the LLM; --skip-prereq if steps 01-02 already ran on this machine.
"""
import argparse
import importlib
import json
import os
import platform
import re
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PY = sys.executable
if sys.version_info < (3, 12):
    sys.exit("Python 3.12 or newer is needed (the pinned numpy and scipy require it).")

ap = argparse.ArgumentParser()
ap.add_argument("--full", action="store_true")
ap.add_argument("--skip-prereq", action="store_true")
ap.add_argument("--model", default=None)
ap.add_argument("--steps", default=None, help="comma-separated agent steps, e.g. DASHBOARD (default: all)")
args = ap.parse_args()

LOG = ROOT / "results" / "agent_run_log.txt"
LOG.parent.mkdir(parents=True, exist_ok=True)


def say(msg):
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


def run(cmd: list[str]) -> bool:
    say(f"\n$ {' '.join(cmd[1:])}")
    with LOG.open("a") as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
        proc.wait()
    if proc.returncode != 0:
        say(f"!! step failed with exit code {proc.returncode}")
    return proc.returncode == 0


LOG.write_text(f"agent run started {datetime.now().isoformat(timespec='seconds')}\n")

# ---------------------------------------------------------------- 1 checks
def _has(p):
    try:
        return importlib.util.find_spec(p) is None
    except ModuleNotFoundError:      # parent package missing, e.g. google for google.genai
        return True


missing_pkgs = [p for p in ["numpy", "pandas", "sklearn", "lightgbm", "imblearn", "pydantic", "google.genai",
                            "matplotlib", "scipy"] if _has(p)]
if missing_pkgs:
    sys.exit(f"Missing packages {missing_pkgs}. Run: pip install -r requirements.txt")

from fraudrank.config import RAW_DIR, TRAIN_FILES  # noqa: E402
from fraudrank.agents.llm import API_KEY_VAR, DEFAULT_MODEL, load_dotenv  # noqa: E402

absent = [f for f in TRAIN_FILES.values() if not (RAW_DIR / f).exists()]
if absent:
    sys.exit(f"Missing data files in data/raw/: {absent}. See README step 2.")

load_dotenv()
MODEL = args.model or DEFAULT_MODEL
KEY_VAR = API_KEY_VAR
KEY = os.environ.get(KEY_VAR, "").strip()
if not KEY:
    sys.exit(f"No API key found. Copy .env.example to .env and paste your key after {KEY_VAR}=")
say("checks passed: packages, data files, API key found (not printed)")

# ---------------------------------------------------------------- 2 prerequisites
needed = [ROOT / "data/processed/provider_features.pkl", ROOT / "results/tables/cv_folds.csv",
          ROOT / "results/cache/test_scores.pkl", ROOT / "results/cache/shap_test_lightgbm.pkl",
          ROOT / "results/cache/shap_test_lgbm_no_size.pkl"]
if args.full or (not args.skip_prereq and not all(p.exists() for p in needed)):
    ok = run([PY, "scripts/01_build_features.py"])
    steps = [] if args.full else ["TUNE", "MODELS", "TEST", "SUMMARY"]
    ok = ok and run([PY, "scripts/02_run_experiments.py", *steps])
    if not ok:
        sys.exit("Prerequisite steps failed; see results/agent_run_log.txt")

# ---------------------------------------------------------------- 3 agents
agent_cmd = [PY, "scripts/03_run_agents.py"] + (["--model", args.model] if args.model else [])
ap_steps = args.steps.split(",") if args.steps else ["RETRIEVAL", "CLEAN", "LEAK_AGENT", "AGENT_MODEL", "QUEUE", "BRIEFS", "DASHBOARD"]
results = {step: run(agent_cmd + [step]) for step in ap_steps}
say(f"\nstep results: {results}")

# ---------------------------------------------------------------- 4 bundle
versions = {}
for p in ["numpy", "pandas", "scipy", "sklearn", "lightgbm", "imblearn", "pydantic", "google.genai"]:
    try:
        versions[p] = importlib.import_module(p).__version__
    except Exception as e:  # noqa: BLE001
        versions[p] = f"unavailable ({e})"
env = {"python": sys.version, "platform": platform.platform(), "packages": versions,
       "model": MODEL,
       "steps": results, "finished": datetime.now().isoformat(timespec="seconds")}
(ROOT / "results" / "agent_environment.json").write_text(json.dumps(env, indent=2))

files = (sorted((ROOT / "results/tables/agents").glob("*"))
         + sorted((ROOT / "results/cache/llm").glob("*.json"))
         + sorted((ROOT / "results/dashboard").glob("*.html"))
         + [ROOT / "results/tables" / n for n in ["cv_summary.csv", "cv_folds.csv", "test_results.csv",
                                                  "tuned_params.json"]]
         + [LOG, ROOT / "results/agent_environment.json"])
files = [f for f in files if f.is_file()]

key_patterns = [re.escape(KEY), r"sk-ant-[A-Za-z0-9_\-]{10,}", r"AIza[0-9A-Za-z_\-]{30,}"]
for f in files:
    text = f.read_text(errors="ignore")
    if any(re.search(p, text) for p in key_patterns):
        sys.exit(f"Refusing to bundle: {f.relative_to(ROOT)} contains something that looks like an API key.")

out = ROOT / "agent_outputs_group20.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f.relative_to(ROOT))
say(f"\nWrote {out.name}: {len(files)} files, scanned for the key, none found. Send this zip back.")
if not all(results.values()):
    say("Some steps failed; the zip still holds the log so the failure can be diagnosed.")
