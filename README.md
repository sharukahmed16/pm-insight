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

Python 3.14, scikit-learn, XGBoost, pandas, Jupyter.

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
| 8 | `python src/test_evaluation.py` | **Final evaluation on the test split** |

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
