# Who Should Medicare Audit Next? Provider Fraud Ranking Beyond Provider Size

BT4012 Fraud Analytics, Group 20 — final project code.

We rank 5,410 Medicare providers for audit using their claims, and ask what a model
finds that provider size alone does not. Total reimbursement alone, with no model,
reaches cross-validated PR-AUC 0.674; our best model reaches 0.732, and a model given
no size features at all still reaches 0.714. An agent layer (six LLM agents with
structured, validated outputs) proposes features and experiments and writes cited
case briefs for the top of the audit queue.

The report is `report/report20.pdf`.

## Repository layout

```
src/fraudrank/            library code
  config.py               paths, seeds, split settings (all experiments read from here)
  data.py                 loading, recoding and integrity checks
  features.py             claim -> provider aggregation, 8 named feature families
  splits.py               frozen 80/20 provider split + 5x5 repeated stratified CV
  models.py               size-only baselines, expert rules, LR, RF, LightGBM, Isolation Forest
  network.py              provider graph, homophily test, leak-free PageRank features
  metrics.py              PR-AUC, precision/recall@k with ceilings, bootstrap CIs
  runner.py               CV and one-shot test evaluation
  agents/                 LLM agents, schemas, executor, hybrid RAG, checks, review dashboard
    corpus/               the typology playbook the case-file agent retrieves from
scripts/
  00_get_data.py          download the Kaggle CSVs into data/raw (kagglehub)
  01_build_features.py    clean data, build provider table, freeze splits
  02_run_experiments.py   tuning, E0-E3, E5-E7, E9, held-out test (no LLM)
  03_run_agents.py        retrieval eval, cleaning-plan score, E4, E8, E10, E11 (LLM)
  run_agents_local.py     one command: checks, prerequisites, agent run, output zip
  04_make_figures.py      report figures
tests/test_agent_plumbing.py   offline check of the agent code paths (fake LLM, no results written)
results/
  tables/                 every number in the report, as CSV/JSON
  figures/                report figures
  cache/llm/              cached LLM responses, so agent results replay without an API key
report/                   LaTeX source and report20.pdf
```

## Reproducing the results

**1. Environment** (Python 3.12 or newer; tested with Python 3.13 on Linux, 2 CPU cores).
The pinned numpy and scipy need Python 3.12+, and the Anthropic SDK needs 3.10+. If your
default Python is older, `uv` downloads 3.13 into a project venv without touching it:

```bash
python3 -m pip install --user uv              # Windows: py -m pip install --user uv
python3 -m uv venv --python 3.13 --seed .venv  # --seed puts pip inside the venv
source .venv/bin/activate                     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

With Python 3.12+ already installed, `python -m venv .venv` and `pip install -r requirements.txt` work too.

**2. Data.** The dataset is not redistributed here. Download it with

```bash
python scripts/00_get_data.py
```

which fetches [Healthcare Provider Fraud Detection Analysis](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)
through `kagglehub`, copies the 8 CSVs into `data/raw/` and checks they match the version used for the
reported results. If Kaggle asks you to sign in, create an API token on kaggle.com (Settings > API) and
save `kaggle.json` to `~/.kaggle/`. Or download the files by hand from the link and place them, unrenamed,
in `data/raw/`:

```
data/raw/Train-1542865627584.csv
data/raw/Train_Beneficiarydata-1542865627584.csv
data/raw/Train_Inpatientdata-1542865627584.csv
data/raw/Train_Outpatientdata-1542865627584.csv
data/raw/Test*-1542969243754.csv        (4 files, unlabelled; not used for evaluation)
```

**3. Run the pipeline** (about 20 minutes on 2 cores; most of it is tuning, the random forest and the test bootstrap):

```bash
python scripts/01_build_features.py        # ~10 s
python scripts/02_run_experiments.py       # TUNE, E0-E3, E5, E7, TEST, E9, SUMMARY
python scripts/04_make_figures.py
```

`02_run_experiments.py` accepts step names (e.g. `python scripts/02_run_experiments.py E0 E1 SUMMARY`).
`TUNE` must run before the model experiments, and `TEST` opens the held-out test set;
run it only after every modelling choice is fixed.

**4. Agent experiments (needs your own API key; it never leaves your machine).**

```bash
cp .env.example .env        # Windows: copy .env.example .env  -- then paste your key after ANTHROPIC_API_KEY=
python scripts/run_agents_local.py
```

This checks the setup, rebuilds what the agents need if it is missing (about 10 minutes),
runs the agent experiments (about 15-30 minutes, roughly 70 API calls, a few dollars at most
on Claude Sonnet 5.5 with low effort) and writes `agent_outputs_group20.zip`: the agent result
tables, the cached LLM responses, the run log and package versions. Every file is scanned for
the key before the zip is written. `.env` is git-ignored.

To reproduce the agent results later without a key or cost, replay from the cache:
`python scripts/03_run_agents.py --replay`. `RETRIEVAL` needs no LLM at all.

LLM calls use JSON-schema structured output (`output_config.format`), pydantic validation with one
retry, and a disk cache. Current Claude models reject `temperature`, so runs are repeated three
times and the spread is reported. Change the model with `--model` or `FRAUDRANK_LLM_MODEL`, and
the effort level with `FRAUDRANK_LLM_EFFORT` (default `low`).

**Provider review dashboard (business layer).** Step `E11` builds an interactive, self-contained
HTML dashboard for Medicare program-integrity analysts (`results/dashboard/provider_review_dashboard.html`;
open it in any browser; it works offline, and loads its two web fonts when online):

* **Overview:** this cycle's headline, start/continue-review and download tiles, key counts, the
  track record (top 100 vs. sorting by payments), an insights panel, recommendations to double-check,
  the review queue, the queue mix, recent activity, and a "size or behaviour?" map of all 1,082 ranked
  providers (with a queue close-up).
* **Providers:** all 1,082 providers as cards or a sortable table, with filters (queue / top 100 / all,
  recommendation, type of care, focus), search and CSV export.
* **Provider page:** profile and summary, an insights panel, what stands out beyond size, why it ranks
  where it does overall (each factor against the range for similar providers), what to check (a
  checklist), matching fraud patterns, data exploration across 16 measures, a status field, notes,
  previous/next in the queue (also the arrow keys) and a copyable case summary.

Status, checklist ticks and notes are saved in the browser only. Light and dark themes and phone
screens are supported.

Every number and chart is computed by code (`dashboard_data.py`). The Dashboard agent writes the
plain-language text: one overview and one page per queued provider. Each piece of text is checked:
every number must match the data shown, no technical terms, no claim that fraud is proven, only
the factors and patterns given, and all required cautions on the home page. Text that fails a
check is dropped and that page shows standard system wording instead. Each field also has a word
limit set by the space it fills on the page (`OVERVIEW_SLOTS`, `PAGE_SLOTS` in `schemas.py`); the
agent is told where each field appears, and a passage that runs over is replaced by standard
wording in that one place only. Run only this step with
`python scripts/run_agents_local.py --skip-prereq --steps E11` (about 15 minutes, roughly 63 API
calls, around US$1).

**5. Test the agent code offline:** `python tests/test_agent_plumbing.py`

## Experimental protocol

* **Unit of analysis:** the provider. Claims are aggregated to one row per provider because the label is per provider.
* **Splits:** a stratified 80/20 provider split (4,328 development, 1,082 test with 101 flagged), seed 42, saved to `data/processed/splits.json`. All model selection uses 5×5 repeated stratified CV on the development set (25 folds). The test set is scored once.
* **Leakage controls:** every fitted transform lives inside the model; SMOTE runs inside the training fold; network features take labels from the training fold only, and training rows get out-of-fold values so no provider sees its own label.
* **Metrics:** PR-AUC (primary), precision@k and recall@k at k = 50, 100, 5% and 10% of providers with the best achievable value printed beside each, ROC-AUC, and confusion-matrix metrics at chosen queue sizes. Test intervals are 2,000-sample stratified bootstrap 95% CIs.
* **Tuning:** small grids scored on the first CV repeat only (`results/tables/tuned_params.json`).

| ID | Question | Output |
|---|---|---|
| E0 | How much of the label does size alone explain? | `cv_summary.csv` (experiment E0) |
| E1 | Which model ranks best? | `cv_summary.csv` (E1) |
| E2 | Which behaviours matter beyond size? | `cv_summary.csv` (E2) |
| E3 | Does the provider network help? | `homophily.json`, `cv_summary.csv` (E3) |
| E4 | Do the build-time agents help? | `agents/e4_summary.csv` |
| E5 | How to handle class imbalance? | `cv_summary.csv` (E5), `e5_calibration.csv` |
| E6 | Which audit queue size? | `e6_operating_points.csv` |
| E7 | How fragile is the result to label noise? | `cv_summary.csv` (E7) |
| E8 | Are case briefs faithful and grounded? | `agents/e8_summary.csv`, `agents/retrieval_eval.csv` |
| E9 | How much does a leaky claim-level split inflate results? | `e9_leakage.csv` |
| E10 | Does the Leakage agent catch planted leaks? | `agents/e10_leak_bench.csv` |
| E11 | Can an agent turn the ranking into a dashboard analysts can act on? | `agents/e11_checks.csv`, `results/dashboard/provider_review_dashboard.html` (first run, before the evidence fix: `agents/e11_checks_first_run.csv`) |

## Important caveat about the data

The claims appear to derive from CMS's synthetic DE-SynPUF Medicare files (same layout,
2008–2009 window, coarsened payments), and the Kaggle page does not say how
`PotentialFraud` was assigned. We treat the label as "previously flagged as potentially
fraudulent" and our conclusions as statements about this benchmark, not about real providers.

## Team

Group 20: <!-- member names --> — see `report/report20.pdf` for the contribution statement.
