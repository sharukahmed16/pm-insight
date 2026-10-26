"""Week 9: final evaluation on the TEST split.

Protocol, fixed before this script is run:

  * The model was selected on the VALIDATION split in week 6: Random Forest with
    the project one-hot feature, log1p target (models/rf_project.joblib).
    XGBoost scored worse on validation macro MAE and is therefore not reported
    here - selecting on validation and reporting the selection is the point.
  * The hybrid threshold was selected on the VALIDATION split in week 8
    (src/hybrid.py) and is hard-coded below. It is NOT re-tuned on test.
  * The test split has not been used for any decision up to this point.

Nothing here is fitted on test data: the vectoriser, the forest and the baseline
statistics all come from the training split.

Run from the repository root:  python src/test_evaluation.py
"""
import os

import joblib
import numpy as np
import pandas as pd

from baselines import per_project_constant, random_rows
from evaluate import score_table
from features import load_splits, to_matrix
from random_forest import add_project

RESULTS = "results"
MODEL_PATH = "models/rf_project.joblib"
MODEL_NAME = "random forest [project] (log, leaf=1, mf=0.05)"
# chosen on validation in week 8; frozen here
HYBRID_THRESHOLD = 8.0
SUMMARY_ROWS = ["overall", "macro"]


def rf_predict(model, X):
    """The saved forest was fitted on log1p targets, so map predictions back."""
    return np.clip(np.expm1(model.predict(X)), 1, None)


def hybrid_predict(test, train, rf_pred, median_pred, threshold):
    """Use the forest on high-variance projects, the median baseline elsewhere.

    The decision uses the spread of each project's TRAINING labels only, so it
    could be made before seeing any test issue.
    """
    train_std = train.groupby("project").storypoint.std()
    use_model = test.project.map(train_std >= threshold).to_numpy(bool)
    return np.where(use_model, rf_pred, median_pred), use_model


def criterion_check(res, model_name):
    """How many of the 16 projects the model beats BOTH baselines on."""
    wide = res[~res.project.isin(SUMMARY_ROWS)].pivot(
        index="project", columns="model", values="MAE")
    wins = (wide[model_name] < wide["mean (per project)"]) & \
           (wide[model_name] < wide["median (per project)"])
    return int(wins.sum()), len(wins), wide


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    splits = load_splits()
    train, test = splits["train"], splits["test"]
    projects = sorted(train.project.unique())

    vec = joblib.load("models/tfidf.joblib")
    X_test = add_project(to_matrix(vec, test), test, projects)

    # --- baselines, computed from training statistics only ---
    mean_pred = per_project_constant(train, test, "mean")
    median_pred = per_project_constant(train, test, "median")

    # --- the model selected on validation ---
    rf = joblib.load(MODEL_PATH)
    rf_pred = rf_predict(rf, X_test)

    # --- the hybrid rule, threshold frozen from week 8 ---
    hyb_pred, use_model = hybrid_predict(train=train, test=test, rf_pred=rf_pred,
                                         median_pred=median_pred,
                                         threshold=HYBRID_THRESHOLD)
    hybrid_name = f"hybrid (model if train_std >= {HYBRID_THRESHOLD:g}, else median)"
    used_on = sorted(test.loc[use_model, "project"].unique())

    tables = [
        score_table(test, mean_pred, train, "mean (per project)"),
        score_table(test, median_pred, train, "median (per project)"),
        random_rows(test, train),
        score_table(test, rf_pred, train, MODEL_NAME),
        score_table(test, hyb_pred, train, hybrid_name),
    ]
    res = pd.concat(tables, ignore_index=True)
    res.to_csv(f"{RESULTS}/week9_test_metrics.csv", index=False)

    pd.set_option("display.width", 250)
    print("=== TEST split, headline scores ===")
    print(res[res.project.isin(SUMMARY_ROWS)][["model", "project", "MAE", "MdAE", "SA"]]
          .sort_values(["project", "MAE"]).round(3).to_string(index=False))

    print(f"\nHybrid used the model on {len(used_on)} of {len(projects)} projects: "
          f"{', '.join(used_on)}")

    print("\n=== Success criterion: lower MAE than BOTH baselines on a majority of projects ===")
    for name in (MODEL_NAME, hybrid_name):
        wins, total, _ = criterion_check(res, name)
        print(f"{name[:52]:<54} {wins:>2} / {total}   "
              f"{'MET' if wins > total / 2 else 'NOT MET'}")

    print("\n=== Per-project MAE ===")
    _, _, wide = criterion_check(res, MODEL_NAME)
    print(wide.round(3).to_string())
    print(f"\nSaved {RESULTS}/week9_test_metrics.csv")
