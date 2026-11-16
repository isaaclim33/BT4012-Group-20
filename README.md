# Who Should Medicare Audit Next?

BT4012 Fraud Analytics, Group 20. We rank 5,410 Medicare providers for audit from their claims,
build a review queue that people can work through, and use LLM agents to explain each case. The
report is [`report/report20.docx`](report/report20.docx) (exported as [`report/report20.pdf`](report/report20.pdf)) and
the presentation is [`report/presentation20.pptx`](report/presentation20.pptx), with a speaker script in its notes.

**Design rule throughout: LLMs propose and explain; code decides and checks.** No LLM decides who
is reviewed, which action they get, or which model is used, and nothing an LLM writes is shown
without a check.

## How it works

The workflow runs in five phases, from claims to an investigator's decision. Step names are the
commands that run each part (`scripts/02_run_experiments.py STEP`, `scripts/03_run_agents.py STEP`).

```mermaid
flowchart TB
    classDef code fill:#eceff3,stroke:#6b7280,color:#111
    classDef llm fill:#ede9fe,stroke:#7c3aed,color:#2e1065
    classDef gate fill:#fef3c7,stroke:#b45309,color:#451a03
    classDef out fill:#dcfce7,stroke:#15803d,color:#052e16
    classDef person fill:#dbeafe,stroke:#1d4ed8,color:#172554

    subgraph P1["1 · Build · scripts 00-01"]
        direction LR
        A1["Claims<br/>5,410 providers"]:::code --> A2["Provider profiles<br/>63 features"]:::code
    end
    subgraph P2["2 · Rank · script 02"]
        direction LR
        B1["Risk score<br/>LightGBM"]:::code ~~~ B2["Size-free score<br/>10 seeds"]:::code
    end
    subgraph P3["3 · Select · QUEUE"]
        direction LR
        C1{{"Rule check<br/>clearly wrong billing"}}:::gate --> C3[("Review queue<br/>65 cases · 3 lanes")]:::out
        C2{{"Risk ranking<br/>top 50 · seed vote"}}:::gate --> C3
    end
    subgraph P4["4 · Explain · BRIEFS, DASHBOARD"]
        direction LR
        D1["Case briefs"]:::llm --> D2["Dashboard text"]:::llm --> D3{{"Checks +<br/>grounding gate"}}:::gate
    end
    subgraph P5["5 · Review · the dashboard"]
        direction LR
        R1["Work the lanes<br/>batches of 12"]:::person --> R2["Case page<br/>evidence · checklist"]:::person
        R2 --> R3["Decision<br/>reason if overriding"]:::person --> R4["Case notes CSV"]:::out
        R4 -.-> R5["Outcomes become labels<br/>future work"]:::code
    end
    P1 --> P2
    P1 --> P3
    P2 --> P3
    P3 --> P4
    P4 --> P5
```

Colours: grey = code that decides, purple = LLM agent, amber = check, blue = investigator, green = output.
Alongside, **CLEAN**, **LEAK_AGENT** and **AGENT_MODEL** test the agents themselves and do not feed
the queue; the model experiments (EDA_TESTS, SIZE, MODELS, FAMILIES, NETWORK, IMBALANCE, NOISE, LEAKY_SPLIT,
QUEUE_COST, TEST, DEEP, EVASION, FAIRNESS) evaluate the ranking.

| Stage | What happens | Who decides |
|---|---|---|
| **Clean + features** (`01`) | Recode flags, parse dates, drop constants, count (never impute) missing fields; aggregate claims into 63 provider features in 8 families; freeze a stratified 80/20 provider split and 5×5 CV folds | Code (`data.py`, `features.py`) |
| **Statistics** (`02 EDA_TESTS`) | Chi-square, Mann-Whitney and Welch tests, skew and kurtosis, Benford's law, boxplot and z-score outlier rules | Code |
| **Models** (`02`) | Size-only baselines, expert rules, logistic regression, random forest, LightGBM, feedforward network, autoencoder, Isolation Forest, network features; tuning on CV repeat 1; the test set is scored once | Code |
| **Stress tests** (`02 EVASION, FAIRNESS`) | Black-box evasion (provider splitting, blend-in) with adversarial training; false-positive rates by provider group; calibration | Code |
| **Review queue** (`03 QUEUE`) | Two-gate audit selection (below) | Code |
| **Case briefs** (`03 BRIEFS`) | One brief per queued provider: drivers, playbook citations, uncertainty, using lookup tools | LLM writes; code checks every number and citation |
| **Dashboard** (`03 DASHBOARD`) | Plain-language overview and provider pages for full-audit and borderline cases | LLM writes; code checks, then a final grounding gate |

### Two-gate audit selection

* **Gate 1 – rule check.** Billing that is clearly wrong goes straight to review: any claim that
  repeats an earlier one (same patient, date, amount, main diagnosis), or two or more inpatient stays
  that start before the same patient's previous stay ended. A single overlap is shown as a warning only.
* **Gate 2 – risk ranking.** The 50 highest-scoring remaining providers (LightGBM) join the queue.
  The size-free model is retrained with 10 seeds: **full audit** if ≥8 of 10 put the provider in
  their size-free top 50, **borderline** desk review at 3–7, otherwise **desk review**.
* **Order.** The rule decides *whether* a provider is reviewed; the model decides *when*: the queue is
  sorted by risk score, then provider ID, so it is identical on every rebuild.

All thresholds are in `src/fraudrank/config.py` and were set on development providers only.

### Review dashboard

`results/dashboard/provider_review_dashboard.html` – one self-contained file, works offline.

* **Today's work:** progress, the next case, and the review queue with lane tabs (full audit,
  borderline, desk review), grouping (reason, pattern, type of care, batch) and **batches of 12**.
  Each row shows one line of evidence, facts only, e.g. `1 duplicate claim · 2 overlapping hospital stays · $3.21M paid`.
* **Case page**, in working order: why it's here (reason and evidence), what to check (a checklist),
  then details on request (comparisons, matching fraud patterns, all measures). Desk reviews get a
  lighter page in standard wording.
* **Decision panel:** open full audit, desk review or no further action; a reason is required when
  differing from the recommendation; *Save & next case* moves on. Keys: 1-3 choose, S save, J/K move.
* **Download case notes:** each case's decision, reason, checklist progress and note as CSV. Decisions
  and notes stay in the browser until downloaded.
* **All providers:** look up any of the 1,082 providers (search, table or cards).
* **Ranking insights:** this cycle at a glance, why cases are here, what investigators will find in
  the records, and the size-or-behaviour map.
* **How it works:** a help panel from the **?** icon or key, with ⓘ links next to key terms.
* **Grounding gate:** every evidence item is re-derived from the case evidence, and every number in
  the AI text must appear in it, written as digits (a number spelled out in words cannot be checked,
  so it fails); anything that fails is not shown, item by item (`agents/dashboard_grounding.csv`).

## Agent harness: design and architecture

The agent layer (`src/fraudrank/agents/`, run by `scripts/03_run_agents.py`) is built around one rule:
**agents propose and explain; code decides and checks.**

### Design principles

| Principle | How it is enforced | Why |
|---|---|---|
| Code makes every decision | Who is reviewed, each action, and the model chosen in AGENT_MODEL are set by code (`audit_selection.py`, `selection.py`). Agents only propose, critique and explain | LLM outputs vary between runs, and a decision that affects a provider must be repeatable and explainable. BRIEFS shows why: with retrieval, the agent's own suggested action matches the rules in 77% of briefs, but it suggests a full audit for 11 of 12 desk-review cases |
| Closed outputs | Every agent returns JSON that matches a schema (`schemas.py`): closed sets of options and bounded numbers, validated with pydantic | An agent can only ask for actions the code knows how to run safely. No free text is executed |
| Agents never see raw data or labels | Agents receive summaries and specifications; the executor (`executor.py`) computes features and runs models | No label leakage, no patient-level data in prompts, and results that do not depend on the model reading data correctly |
| Read-only tools, least privilege | Tools look things up and return facts; no tool writes or changes anything (`tools.py`) | A tool call cannot alter the data, the queue or another agent's output |
| Every output is checked before use | Each step has its own code check; failing output is dropped and replaced with standard wording or a fallback, never shown unchecked | What reaches an investigator must trace to the evidence |
| Untrusted text is evidence, not instructions | Data goes inside `<data>` tags, and every prompt says text in them is to be analysed, never followed | A defence against prompt injection from claim text or retrieved documents |
| Reproducible by design | Every response is cached and replays without a key; each experiment runs 3 times; the model version is logged | Sampling cannot be fixed on current models, so the cache is the record of what each agent said |

### Architecture

Colours: **grey** = code that decides, **purple** = LLM agent (proposes or explains), **amber** = code check,
**green** = output.

**1. System overview.** Everything that decides who is reviewed, and how, is code. LLM agents only
write explanations, and each one is checked before it is shown.

```mermaid
flowchart TB
    classDef code fill:#eceff3,stroke:#6b7280,color:#111
    classDef llm fill:#ede9fe,stroke:#7c3aed,color:#2e1065
    classDef gate fill:#fef3c7,stroke:#b45309,color:#451a03
    classDef out fill:#dcfce7,stroke:#15803d,color:#052e16

    subgraph S1["Scripts 00-02 · data and models"]
        A1["Kaggle claims<br/>5,410 providers"]:::code --> A2["Clean + 63 provider features<br/>data.py · features.py"]:::code
        A2 --> A3["LightGBM risk score<br/>+ size-free model, 10 seeds"]:::code
    end

    subgraph S2["QUEUE · two-gate audit selection"]
        B1{{"Gate 1: rule check<br/>duplicate claims · overlapping stays"}}:::gate
        B2{{"Gate 2: risk ranking<br/>top 50 · seed vote sets the action"}}:::gate
        B3[("audit_queue.csv<br/>65 cases · lane · order")]:::out
        B1 --> B3
        B2 --> B3
    end

    subgraph S3["BRIEFS · case briefs"]
        C1["Evidence from code<br/>drivers · peer percentiles · billing checks"]:::code --> C2["Case-file agent<br/>+ read-only tools"]:::llm
        C2 --> C3{{"check_brief<br/>numbers · citations · no fraud claim"}}:::gate
    end

    subgraph S4["DASHBOARD · dashboard text"]
        D1["Dashboard agent<br/>overview + case pages"]:::llm --> D2{{"Page checks<br/>numbers · terms · provider"}}:::gate
        D2 --> D3{{"Grounding gate<br/>item by item, digits only"}}:::gate
    end

    A2 --> B1
    A3 --> B2
    B3 --> C1
    C3 --> D1
    B3 --> D1
    D3 --> E1["Review dashboard<br/>lanes · decision panel · case notes"]:::out
```

The agent experiments run alongside and do not feed the queue:
**CLEAN** (cleaning plan, scored against `data.py`), **LEAK_AGENT** (leak benchmark) and **AGENT_MODEL** (diagram 3).

**2. One agent step.** Every LLM call goes through the same harness (`llm.py`), and every result
goes through a code check before it is used.

```mermaid
flowchart TB
    classDef code fill:#eceff3,stroke:#6b7280,color:#111
    classDef llm fill:#ede9fe,stroke:#7c3aed,color:#2e1065
    classDef gate fill:#fef3c7,stroke:#b45309,color:#451a03
    classDef out fill:#dcfce7,stroke:#15803d,color:#052e16

    E["Evidence from code<br/>inside data tags"]:::code --> P["Prompt + output schema<br/>closed sets · bounded numbers"]:::code
    subgraph H["LLM client · llm.py"]
        K{"Same request<br/>in the cache?"}:::code
        L["Gemini<br/>timeout · back-off · quota stop"]:::llm
        T["Read-only tools"]:::code
        R["Response<br/>stored in the cache"]:::code
        V{{"Schema check<br/>pydantic"}}:::gate
        K -->|no| L
        L <-->|ReAct| T
        L --> R
        K -->|yes| R
        R --> V
        V -.->|"invalid: retry once<br/>with the error"| L
    end
    P --> K
    V -->|valid| G{{"Code check<br/>for this step"}}:::gate
    G -->|pass| U["Used or shown"]:::out
    G -->|fail| F["Dropped · fallback<br/>or standard wording"]:::out
```

**3. AGENT_MODEL loop.** Agents propose; code computes, filters and chooses.

```mermaid
flowchart LR
    classDef code fill:#eceff3,stroke:#6b7280,color:#111
    classDef llm fill:#ede9fe,stroke:#7c3aed,color:#2e1065
    classDef gate fill:#fef3c7,stroke:#b45309,color:#451a03
    classDef out fill:#dcfce7,stroke:#15803d,color:#052e16

    F["Feature agent<br/>proposes features"]:::llm --> X["Executor<br/>computes them from claims"]:::code
    X --> K["Leakage agent<br/>screens them"]:::llm
    K --> E["Experiment agent<br/>3-4 configurations"]:::llm
    E --> G{{"Preview gate<br/>3 folds"}}:::gate
    G --> C["Cross-validation<br/>CV repeat 1"]:::code
    C --> S{{"Paired-fold rule<br/>chooses the model"}}:::gate
    S --> R["Critic<br/>stop or next round"]:::llm
    R -->|"next round<br/>max 3"| E
    R -->|stop| V["Evaluate once<br/>unseen CV repeats 2-5 + test"]:::out
```

**Shared LLM client (`llm.py`)**, used by every agent:

| Feature | What it does | Why |
|---|---|---|
| Structured output | Output is constrained to a JSON schema (Gemini `response_json_schema`), then validated with pydantic. On a validation error the call is retried once with the error appended; a second failure raises an error | Malformed output is never silently dropped or half-used |
| Response cache | Every response is stored, keyed by a SHA-256 hash of the full request (model, system prompt, messages, schema or tools, effort). `--replay` runs every step from the cache with no API key | Exact reproduction of agent results, and free re-runs of steps whose inputs did not change |
| Resilience | 3-minute timeout per request; up to 7 attempts with back-off (10 to 120 s) on rate limits, server errors and dropped connections. A used-up **daily** quota stops the step at once, before it writes anything | Free-tier overload is common; a step that runs out of quota must not overwrite good results with partial ones |
| Effort | Low by default; high for judgement tasks (cleaning plan, critic, case briefs) | Spend thinking where the task needs judgement, not on formatting |
| ReAct loop | The model calls tools until it is done (at most 3 turns for the critic, 6 for case briefs), then one final structured call turns the investigation trace into schema-checked output. Every turn is cached | Tools let an agent verify facts before it writes, and the final output still goes through the same schema and checks |
| Provenance | The exact model version that answered is stored with each response and in `llm_usage.json`, with call and token counts | Model aliases move over time; the log shows what produced each result |

### Steps: what each does and why

Steps run in this order (`run_agents_local.py`), and each can be re-run on its own:
`RETRIEVAL → CLEAN → LEAK_AGENT → AGENT_MODEL → QUEUE → BRIEFS → DASHBOARD`.

**RETRIEVAL** (no LLM). Hybrid search over the ten-section fraud-typology playbook: BM25 for exact
terms plus a dense latent-semantic index, fused by reciprocal rank. It is scored on 20 labelled queries
written with different words from the playbook.
*Why:* case briefs cite the playbook, so retrieval quality is measured on its own, before any generation.

**CLEAN: can an LLM reproduce the analyst's cleaning decisions?**
* The cleaning agent proposes a plan for each raw file (4 calls), with a general checklist (yes/no
  flags not coded 0/1, constant columns, dates stored as text, values missing because they do not apply).
* Code requires exactly one step for every column; it asks once more for any column left out, and
  drops steps for columns that do not exist.
* The plan is scored against the 7 decisions in our own `data.py`. **It is not executed**: the
  pipeline always uses `data.py`.
* *Why:* a single call asked for "columns that need action" covered only the easy ones (2 of 7).
  Enforcing coverage in code, not in the prompt, raised it to 6.3 of 7.

**LEAK_AGENT: does the leakage agent catch leaks?** The agent reviews 15 feature specifications, 5 of them
with planted leaks (label, test-set or after-the-fact information), and code scores catches and
false alarms.
*Why:* AGENT_MODEL relies on this agent to screen features, so it is tested against known answers first.

**AGENT_MODEL: can agents build a model as good as the hand-tuned pipeline?** One loop, repeated 3 times:
1. **Feature agent** proposes new features from a closed grammar (a claim column × an aggregate ×
   inpatient/outpatient/all, or a ratio of provider totals). The executor computes them from the
   claims; the agent never sees the data.
2. **Leakage agent** screens them; rejected features are removed.
3. **Experiment agent** proposes 3–4 configurations (model, feature set, imbalance handling, bounded
   settings), from a trimmed context: feature-set sizes, model options and past results only.
4. **Preview gate** (code) scores each proposal on 3 folds and drops any whose mean plus one standard
   deviation is below the best seen so far on those same 3 folds; at least one always survives.
5. **Cross-validation** on the first CV repeat, then **code chooses** with a paired-fold rule
   (`selection.py`): the best-scoring configuration wins only if it beats a simpler one by more than
   one paired standard error *and* on most folds; otherwise the simpler one is kept.
6. **Critic** (ReAct, with a tool that returns per-fold scores) sees the code's choice and decides
   whether to stop, which experiments to run next (up to 3 rounds), and what looks wrong.
7. **Evaluation** on CV repeats 2–5, which the loop never saw, and once on the test set; ablations
   (no critic, no feature agent, no leakage review) are repeated for every run.

*Why this design:*
* In the first version, the critic chose the model using a "beat it by one standard deviation" rule
  and picked weaker models; comparing models fold by fold is the right yardstick, and code applies it
  exactly.
* An earlier experiment agent was told to drop weak proposals and did not; the preview gate does it in code.
* Scoring on the folds the loop chose on would flatter the agents, hence repeats 2–5.

**QUEUE** (no LLM). Two-gate audit selection: the rule check, the risk ranking and the 10-seed model
vote (see above). Written to `audit_queue.csv` before any later agent runs; BRIEFS and DASHBOARD stop if it is missing.
*Why:* who gets audited, and how, must not depend on an LLM.

**BRIEFS: case briefs.** One brief per queued provider, with and without playbook retrieval (130 briefs).
* Evidence comes from code: the five strongest model drivers with values and peer percentiles, and
  the billing-check counts.
* The case-file agent can call `lookup_provider`, `get_peer_stats` and `search_playbook` before it writes.
* `check_brief` (code) checks every brief: driver values, percentiles and model contributions match the
  evidence; every cited playbook section was actually retrieved and lists one of the brief's drivers;
  the rank is right; and there is no claim that fraud is proven (negation-aware).
* The agent's suggested action is recorded next to the code's decision, but decides nothing.
* The trajectory is logged (`briefs_trajectory.json`): which tools were called, on which provider, with what result.
  In practice the agent made no tool calls, because every number it needed was already in its prompt.
* *Why:* tools let the agent verify before writing; code checks make faithfulness measurable; the
  no-retrieval condition shows what retrieval adds.

**DASHBOARD: dashboard text.** One overview and one page per full-audit or borderline case; desk reviews
get standard wording. Run 3 times; the dashboard is built from run 1.
* Evidence is formatted exactly as the page shows it (the page's rounding is mirrored in Python),
  and each text field has a word limit set by its space on the page.
* Code checks (`dashboard_agent.py`): every number must appear in the evidence, no technical terms, no
  claim that fraud is proven, the right provider, only known factors and patterns, and all required
  cautions on the overview.
* A page that fails is replaced by standard wording; text that only runs over its word limit is
  dropped in that one place.
* A final **grounding gate** (`grounding.py`) then checks everything shown about each case: each
  evidence item is re-derived from the case evidence, and each displayed AI sentence is checked item
  by item, with numbers written as digits.
* *Why:* investigators act on this text, so every statement must trace to the case. Checking item by
  item means one bad sentence does not take a whole page down.

**Checks and tests.** `tests/test_agent_plumbing.py` runs every agent code path with a fake LLM
(nothing written to `results/`), including the selection rules, the queue rebuild, and the grounding
gate's handling of invented numbers, jargon and numbers written in words.

## Results

Data: 5,410 providers, 9.35% previously flagged. The label means *previously suspected*, not proven
fraud. Test set: 1,082 providers, 101 flagged.

**Ranking** (PR-AUC; CV = 25 folds on development data, test = held-out 20%, 95% bootstrap CI)

| Model | CV | Test |
|---|---|---|
| Total payments alone (no model) | 0.674 | 0.743 [0.670, 0.815] |
| Expert rules | 0.390 | 0.404 |
| Logistic regression | 0.699 | 0.775 |
| Random forest | 0.708 | 0.782 |
| **LightGBM** | **0.732** | **0.805** [0.740, 0.867] |
| LightGBM, no size features | 0.714 | 0.784 [0.716, 0.853] |
| Feedforward neural network | 0.677 | 0.785 [0.722, 0.848] |
| Autoencoder (trained on non-flagged) | 0.140 | 0.088 [0.080, 0.101] |

* A model beats provider size alone, and still does without any size features.
* Hospital-stay intensity is the strongest behaviour signal: dropping it from the size-free model costs 0.087 PR-AUC.
* Network features add nothing (+0.001); the naive homophily test is confounded by degree.
* With 30% of flags hidden, PR-AUC falls gracefully to 0.686; a leaky claim-level split inflates results.
* LightGBM on test: 86% of the top 50 and 74% of the top 100 were previously flagged.
* Deep models do not beat boosting: the feedforward network is level with size alone in CV (LightGBM is better on
  25 of 25 folds), and the autoencoder's most anomalous providers have a median of one claim (unusual, not suspicious).
* Statistics: flag and type of care are dependent (chi-square 607, Cramér's V 0.33); Benford's law gives no signal
  (flagged and other providers deviate equally, MAD 0.0176 vs 0.0175).
* Evasion (test PR-AUC): splitting each flagged provider into two IDs cuts size alone 0.743 → 0.522 and LightGBM
  0.805 → 0.661, but not the size-free model (0.784); cutting stay lengths cuts the size-free model to 0.408.
  Adversarial training on split copies fixes splitting (0.825) but not the unseen attack (0.448).
* Fairness: in the cost-optimal queue, honest providers that bill inpatient care are queued by mistake at 22.8%
  against 2.6% for outpatient-only ones; dropping patient-mix features narrows the patient-acuity gap at no cost.

**Review queue** (test set)

| | Providers | Previously flagged |
|---|---|---|
| Gate 1 rule check | 15 (7 duplicate claims, 10 overlapping stays) | 12 |
| Gate 2 risk ranking | 50 → 38 full audit, 4 borderline, 8 desk review | 41 |
| **Queue** | **65** | **53** |

On development data, the gate-1 rules refer 59 providers, 81% of them flagged (base rate 9%).

**Agents** (Gemini 3.5 Flash-Lite, 3 runs each)

| Experiment | Result |
|---|---|
| CLEAN | 6.3 of 7 reference decisions on average (6, 6, 7); 84 of 84 columns covered. The original one-call prompt caught 2 of 7 |
| LEAK_AGENT | 5/5 planted leaks caught, 0 false alarms, in every run |
| AGENT_MODEL vs fixed | CV 0.724 vs 0.730 (range 0.723–0.725); test 0.802 vs 0.805, difference −0.003 [−0.021, +0.014]. Agents match, but do not beat, the hand-tuned pipeline; the feature agent adds nothing |
| BRIEFS | 130 briefs: 641/641 driver values exact, 163/163 citations real and on point, 0 claims of proven fraud. With retrieval, the agent's suggested action matches the code's decision in 77% of briefs, but it suggests a full audit for 11 of 12 desk-review cases. Trajectory: 0 tool calls in 65 briefs |
| DASHBOARD | Overview passed in 3/3 runs; provider pages 56, 57, 56 of 57 (failures: wrong provider ID, caught) |
| Grounding gate | 162/162 evidence items re-derived from case evidence; 13 of 506 displayed AI items removed (numbers written in words), none with an unsupported number |
| Retrieval | Hybrid search recall@1 0.90 (BM25 0.85, dense 0.80) on 20 test queries |

## Reproducing the results

**1. Environment** (Python 3.12+; tested on 3.13)

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On macOS, LightGBM needs OpenMP: `brew install libomp`.

**2. Data** (not redistributed; downloads from Kaggle and checks file sizes)

```bash
python scripts/00_get_data.py
```

**3. Models** (~20 min on 2 cores)

```bash
python scripts/01_build_features.py
python scripts/02_run_experiments.py          # EDA_TESTS, TUNE, SIZE, MODELS, FAMILIES, NETWORK, IMBALANCE, NOISE,
                                              # TEST, DEEP, EVASION, FAIRNESS, LEAKY_SPLIT, SUMMARY
python scripts/04_make_figures.py
```

**4. Agents, queue and dashboard**

```bash
cp .env.example .env                          # paste your key after GEMINI_API_KEY=
python scripts/run_agents_local.py            # RETRIEVAL, CLEAN, LEAK_AGENT, AGENT_MODEL, QUEUE, BRIEFS, DASHBOARD
```

Or replay every agent result from the cache, with no key and no cost:

```bash
python scripts/03_run_agents.py --replay
```

Notes:

* About 450 LLM calls for a full run (447 in ours). Gemini's free tier allows roughly that per day; if the daily
  quota runs out, the step stops without writing partial results – run the same command after the
  reset and the cache carries on where it stopped.
* Offline check of all agent code paths (fake LLM, nothing written): `python tests/test_agent_plumbing.py`.
* Replay is exact on the platform that produced the cache (Linux, Python 3.13). On another platform, scripts
  00–02 reproduce every table exactly except last-digit floating-point noise (for example 0.69559 vs 0.69564 in a
  tuning grid). AGENT_MODEL embeds those scores in its prompts, so on Windows its cache keys can miss: with
  `--replay` it then stops with an error instead of calling the API. RETRIEVAL, CLEAN, LEAK_AGENT, QUEUE, BRIEFS
  and DASHBOARD replay identically.

## Repository layout

```
src/fraudrank/
  data.py, features.py, splits.py      cleaning, 63 features, frozen split and folds
  models.py, network.py, metrics.py    models, provider network, PR-AUC / precision@k / CIs
  audit_selection.py                   two-gate audit selection
  config.py                            seeds, split, queue settings
  agents/                              LLM client, agents, schemas, tools, retrieval,
                                       checks, grounding gate, dashboard
scripts/                               00 data · 01 features · 02 models · 03 agents · 04 figures
tests/test_agent_plumbing.py           offline checks with a fake LLM
results/tables/                        every number above (agents/ for the agent layer)
results/dashboard/                     the review dashboard
results/cache/llm/                     cached LLM responses
report/                                report20.docx (editable), report20.pdf, presentation20.pptx
```

## Limitations

* The claims appear to come from CMS's synthetic DE-SynPUF files, and how `PotentialFraud` was
  assigned is undocumented: results describe this benchmark, not real providers.
* The test set has 101 flagged providers, so close models are not separable at one queue size.
* LLM outputs vary between runs; every agent experiment is run three times and the spread reported.
* The evasion attacks edit provider features rather than re-simulating claims, and fairness covers two groupings.
* The labels cannot reward the rule check: duplicate claims and overlapping stays are evidence of a
  different kind. Of the 4 rule referrals the model ranks outside its top 65, 1 was previously flagged.
