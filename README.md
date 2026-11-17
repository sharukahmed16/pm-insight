# PM-Insight

Story point estimation for agile work items using machine learning.

MITM 421 Project — Institute of Information Technology, University of Dhaka.

When a new work item is created, the size of the task is normally estimated by hand in a
planning meeting. This project trains a model on the recorded history of 16 open-source
agile projects and uses it to suggest a story point value from the issue's title and
description.

---

## Results

Final evaluation on the held-out test split (4,634 issues, never used for any decision).
`overall` is weighted by issue count; `macro` is the unweighted average across projects.

| model | overall MAE | overall SA | macro MAE | macro SA |
|---|---:|---:|---:|---:|
| hybrid (model if train_std ≥ 8, else median) | 2.899 | 49.0% | 2.511 | 42.9% |
| **random forest [project]** | **2.920** | **48.6%** | **2.536** | **41.7%** |
| median (per project) | 2.984 | 47.5% | 2.553 | 42.6% |
| mean (per project) | 4.171 | 26.6% | 3.473 | 27.5% |
| random guessing | 5.686 | 0% | 4.657 | 0% |

**Success criterion (set before the models were built):** a lower MAE than both the mean
and median baselines on a majority of the 16 projects, with a positive Standardized
Accuracy.

**Met.** Random Forest beats both baselines on 9 of 16 projects, with SA = 48.6% overall.

### What the numbers actually mean

The criterion is met, but the margin over the median baseline is about 2%. The model earns
its place on projects whose story points vary widely — moodle (12.57 → 6.15 against the
mean) and datamanagement (8.80 → 5.88) — and adds little where the median is already
tight, such as bamboo (0.75) or usergrid (1.17).

The hybrid rule, which uses the model only on projects whose training labels have a
standard deviation of 8 or more, has marginally the best aggregate scores. It is **not
recommended**: it beats both baselines on only 2 of 16 projects (on the other fourteen it
*is* the median, so it ties), and its advantage over the plain Random Forest is 0.7%. The
extra mechanism does not pay for its complexity.

---

## Week 10: board feedback after the midterm

The board asked for (1) better text representations than TF-IDF (word2vec, BERT or a
domain-specific technique) and (2) other models, including LLMs. The split, metrics,
baselines and protocol are unchanged; only the representation and the model vary.

Representations: TF-IDF (week 4), LSA (TF-IDF → 300-dim SVD), Stack Overflow word2vec
(Efstathiou et al., MSR 2018: the domain-specific option) and Sentence-BERT
(all-MiniLM-L6-v2, frozen). Models: Ridge, analogy-based kNN (median story point of the k
most similar training issues of the same project), SVR, Random Forest, XGBoost.

### Validation (used for every choice)

Macro MAE, lower is better. Median baseline 2.965, mean baseline 3.592.

| representation | kNN | RF | Ridge | SVR | XGBoost |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 2.897 | 2.964 | 3.004 | 2.887 | 3.019 |
| LSA | 2.916 | 3.161 | 3.179 | 3.193 | 3.095 |
| Stack Overflow word2vec | 2.871 | 3.149 | 3.084 | 3.130 | 3.004 |
| Sentence-BERT | **2.853** | 3.103 | 2.996 | 3.075 | 2.976 |

The rule fixed in advance (lowest validation macro MAE) selected **Sentence-BERT + kNN (k=80)**.

### Test (run once, `src/test_evaluation_week10.py --final`)

| model | overall MAE | macro MAE | projects beating both baselines |
|---|---:|---:|---:|
| week 9: TF-IDF + Random Forest | **2.920** | **2.536** | 9 / 16 |
| median (per project) | 2.984 | 2.553 | n/a |
| week 10 winner: Sentence-BERT + kNN | 3.028 | 2.656 | **10 / 16** |
| Stack Overflow word2vec + kNN | 3.015 | 2.619 | n/a |

Full table: `results/week10_test_metrics.csv`.

### What the test result means

The week 10 winner meets the success criterion (10 of 16 projects) but its aggregate error is
**higher** than both the week 9 Random Forest and the median baseline. The validation advantage
did not carry over to the test split.

One project accounts for it. On **moodle** the winner's test MAE is 8.49 against 6.48 for the
median, which alone adds about 0.13 to the macro MAE. Moodle's estimates changed over time:

| moodle | actual mean | actual median | kNN prediction (median) |
|---|---:|---:|---:|
| validation period | 18.8 | 8 | 13 |
| test period | 7.1 | 3 | 13 |

Because the splits are in creation order, the test period is the most recent one. The team
began giving much smaller estimates, while the most similar *past* issues still carried large
ones. kNN reproduces the past exactly, which was rewarded on validation and penalised on test.
On the other 15 projects the winner's test MAE is slightly below both the median and the
Random Forest (by about 0.02 on average).

Conclusions:

1. Dense embeddings help **only** with similarity-based estimation (kNN). With RF, SVR, Ridge and
   XGBoost, TF-IDF stays ahead on both validation and test.
2. The domain-specific Stack Overflow word2vec almost matches Sentence-BERT (validation 2.871 vs
   2.853, test 2.619 vs 2.656) while being far cheaper to compute (16 s vs 17 min for all
   23,129 issues on a laptop CPU).
3. None of the representations beats the week 9 model on held-out data. **The application keeps
   the week 9 Random Forest.** This decision was taken after seeing the test result. It does not
   pick a new winner from the test numbers; it keeps the existing model because the challenger
   failed to beat it.
4. Estimation drift over time (moodle) limits every model trained on history. It is reported as
   a limitation.

Not done: word2vec trained on our own issues needs gensim, which has no Python 3.14 wheel yet.
The 43% of tokens missing from the Stack Overflow model are stop words, which that model excludes.
Week 12 covers the LLM comparison.

---

## Week 12: LLM estimation (board feedback, part 2)

A local open LLM (`qwen2.5:7b`, run with Ollama on a laptop CPU, temperature 0) estimates
story points in two settings: **zero-shot** (issue text only) and **few-shot** (plus the 5 most
similar training issues of the same project with their real story points, retrieved with
Sentence-BERT). Nothing was tuned, so the test split is used directly, on a stratified sample
of 20 issues per project (320 issues). Every model is scored on the same 320 issues.
Script: `src/llm_estimate.py`; details: `docs/week12_llm.md`.

| model (test sample, 320 issues) | MAE | projects beating both baselines |
|---|---:|---:|
| median (per project) | **2.416** | n/a |
| week 9: TF-IDF + Random Forest | 2.478 | 8 / 16 |
| LLM zero-shot | 2.541 | 5 / 16 |
| week 10: Sentence-BERT + kNN | 2.628 | 8 / 16 |
| LLM few-shot (5 similar issues) | 2.744 | 5 / 16 |
| mean (per project) | 3.399 | n/a |

Each project contributes 20 issues, so overall and macro MAE are equal here.

What the result means:

1. **The LLM does not beat the per-project median**, in either setting, and it beats both
   baselines on only 5 of 16 projects.
2. **Zero-shot is almost a constant.** The model answered "3" for 247 of the 320 issues (77%).
   Its reasonable MAE comes from 3 being the overall median story point, not from reading the
   issue.
3. **Few-shot does use the examples** (its answers spread over 1, 2, 3, 5, 8, 13), but it
   inherits the same weakness as kNN: on moodle the retrieved past issues carry the old, larger
   estimates, and the MAE there is 10.50 against 7.05 for the median. Without moodle, few-shot
   (2.227) is on par with kNN (2.233) and still behind the median (2.107).
4. **Cost.** 13 s per issue zero-shot and 57 s few-shot on a laptop CPU (about 6 hours for the
   640 calls), against milliseconds for the Random Forest.

Limitations: a sample of 20 issues per project is noisy (on it, even the Random Forest does
not beat the median, although it does on the full test split); one model and one prompt were
tried; the issues are public, so the LLM may have seen them during its own training. That
would favour the LLM, and it still did not win.

**Decision unchanged: the application uses the week 9 Random Forest.**

---

## The web application

A project tracker (projects, sprints, a three-column board, work items with status
history) with the week 9 model inside the work-item form: "Suggest story points" shows the
model's value and the three most similar past issues, and every saved item records both
the model's suggestion and the team's final number. FastAPI, SQLAlchemy, PostgreSQL,
Jinja2 templates; login with hashed passwords; manager and member roles.

Setup: `setup_app.bat` once, then `run_app.bat`. Structure, tables, security decisions and
limitations: `docs/app.md`.

---

## Dataset

Deep-SE story point dataset — Choetkiertikul et al., *A Deep Learning Model for Estimating
Story Points*, IEEE Transactions on Software Engineering, 2019.

23,313 issues from 16 open-source projects across 9 issue repositories (Apache,
Appcelerator, Atlassian, Duraspace, Lsstcorp, Moodle, Mulesoft, Spring, Talendforge).
Each row has `issuekey`, `title`, `description` and `storypoint`, where the story point
value was assigned by the project's own development team.

The data is **not committed to this repository**. To obtain it:

```bash
git clone --depth 1 https://github.com/morakotch/datasets.git
```

Copy the 16 CSV files from `storypoint/IEEE TSE2018/dataset/` into `data/raw/`.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Python 3.14, scikit-learn, XGBoost, pandas, Jupyter; sentence-transformers for the
week 10 embeddings. gensim (for `w2v_own` only) has no Python 3.14 wheel yet; see
`docs/week10_embeddings.md`.

---

## Reproducing the results

Run from the repository root, in this order. Each script writes its output to `results/`.

| # | Command | What it does |
|---|---|---|
| 1 | `python src/prepare.py` | Clean the raw CSVs, split each project 60/20/20 in creation order |
| 2 | `python src/features.py` | Fit the TF-IDF vectoriser on training text only, save the feature matrices |
| 3 | `python src/baselines.py` | Mean, median, random guessing and Ridge, scored on validation |
| 4 | `python src/random_forest.py` | Random Forest grid search on validation |
| 5 | `python src/xgboost_model.py` | XGBoost grid search on validation |
| 6 | `python src/compare.py` | Cross-model comparison against the baselines |
| 7 | `python src/hybrid.py` | Per-project rule: model on high-variance projects, median elsewhere |
| 8 | `python src/test_evaluation.py` | **Final evaluation on the test split** (week 9) |
| 9 | `python src/embeddings.py <name>` | Build one dense text representation: `lsa`, `w2v_own`, `w2v_so`, `sbert`, `sbert_mp` |
| 10 | `python src/compare_representations.py` | Every representation × 5 models (Ridge, kNN, SVR, RF, XGBoost), scored on validation |
| 11 | `python src/test_evaluation_week10.py --final` | **Final evaluation of the week 10 winner on the test split** |
| 12 | `run_week12_llm.bat` (or `python src/llm_estimate.py --split test --n 20`) | LLM zero-shot and few-shot estimation on a test sample; needs Ollama |

Details and run times for steps 9–11: `docs/week10_embeddings.md`.

`notebooks/01_eda.ipynb` holds the exploratory analysis of the raw data.

Steps 4 and 5 are the slow ones — a grid search over the full training matrix takes
several minutes each.

---

## Method notes

These are the decisions most likely to be questioned, recorded here so the reasoning is
not lost.

**Ordering.** The dataset has no creation timestamp. Issue keys inside a tracker are
assigned in increasing order as issues are created, so the numeric part of the key
(`MESOS-313` → 313) is used as the ordering. Splits are taken in that order, so the model
is always trained on earlier issues and tested on later ones.

**Splitting is per project, not pooled.** A "5" means different things to different teams:
moodle averages 15.5 story points, duracloud 2.1. Every model, baseline and split is
computed within a project.

**Cleaning.** JIRA code blocks, `{noformat}` sections, URLs, user mentions and wiki markup
are stripped; text is lowercased and whitespace collapsed. Issues with no story point, a
non-positive story point, or an empty title are dropped. Exact duplicate text within a
project is deduplicated keeping the earliest issue, so the same text cannot appear in both
the training and the test split. 23,313 raw issues → 23,129 after cleaning →
13,870 train / 4,625 validation / 4,634 test.

**Features.** TF-IDF over the combined title and description — unigrams and bigrams,
maximum 10,000 features, `min_df=2`, sublinear term frequency, English stop words — plus a
binary `has_description` flag. The `[project]` model variants add a one-hot project column;
the `[text]` variants do not, and so could be applied to a project the model has never
seen.

**Target scale.** Story points are heavily skewed, so the selected models are fitted on
`log1p(storypoint)` and predictions mapped back with `expm1`, then clipped at a minimum
of 1.

**No test data was used for any decision.** The vectoriser, the models and the baseline
statistics are all fitted on the training split. Hyperparameters and the hybrid threshold
were selected on the validation split. `src/test_evaluation.py` was run once, at the end.

---

## Repository structure

```
data/raw/          16 source CSVs (not committed)
data/processed/    cleaned splits and feature matrices (not committed)
models/            fitted vectoriser and models (not committed)
notebooks/         exploratory analysis
results/           metrics tables, one CSV per stage
src/               the pipeline
```

---

## Status

The estimation system — the primary scope of the project — is complete through final
evaluation. Remaining work: the web application that serves the saved model (database,
authentication, work item module, prediction endpoint and estimation interface), and the
project report.

A customisable report builder is secondary scope and will be attempted only if time
permits once the estimation system is finished.

---

## AI tool use

GitHub Copilot, Claude and ChatGPT were used for boilerplate, debugging, learning current
library versions and documentation drafting, as declared in the project proposal. All code
has been reviewed, tested and is independently understood.
