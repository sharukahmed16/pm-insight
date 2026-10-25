"""Week 8: does a per-project rule beat using one model everywhere?

Week 7 showed the model wins on some projects and loses on others. This asks
whether the winners can be identified in advance from the TRAINING data alone,
using the spread of each project's story points as the signal.

Rule: for a project, use the Random Forest prediction if the standard deviation
of that project's training labels is at or above a threshold; otherwise fall
back to the per-project median baseline. The threshold is swept here and chosen
on the validation split - so it is a selection made on validation, and must be
confirmed once on the test split in week 9.

No leakage: train_std uses training labels only.

Run from the repository root:  python src/hybrid.py
"""
import os

import numpy as np
import pandas as pd

RESULTS = "results"
PROCESSED = "data/processed"
SUMMARY_ROWS = ["overall", "macro"]
MEDIAN = "median (per project)"
THRESHOLDS = [0, 2, 3, 4, 5, 6, 8, 10, 15]


def per_project(path, model_pattern):
    """MAE and issue count per project for the model matching `model_pattern`."""
    df = pd.read_csv(os.path.join(RESULTS, path))
    rows = df[df.model.str.contains(model_pattern, regex=True) &
              ~df.project.isin(SUMMARY_ROWS)]
    return rows.set_index("project")[["n", "MAE"]]


def train_spread():
    """Standard deviation of each project's training story points."""
    train = pd.read_csv(os.path.join(PROCESSED, "train.csv"))
    return train.groupby("project").storypoint.std()


def apply_rule(median_mae, model_mae, std, threshold, weights):
    """Score the rule 'use the model where train_std >= threshold, else median'.

    Returns the issue-weighted (overall) MAE, the project-averaged (macro) MAE,
    and how many projects the model was used on.
    """
    use_model = std.reindex(median_mae.index) >= threshold
    chosen = np.where(use_model, model_mae, median_mae)
    return (float(np.average(chosen, weights=weights)),
            float(np.mean(chosen)),
            int(use_model.sum()))


if __name__ == "__main__":
    median = per_project("week5_validation_metrics.csv", r"^median \(per project\)$")
    model = per_project("week6_rf_validation_metrics.csv", r"random forest \[project\]")
    std = train_spread()

    weights = median.n.to_numpy(float)
    med_mae, mod_mae = median.MAE, model.MAE.reindex(median.index)

    rows = []
    # the two reference points: never use the model, always use the model
    rows.append({"rule": "median everywhere", "threshold": None,
                 "overall_MAE": float(np.average(med_mae, weights=weights)),
                 "macro_MAE": float(med_mae.mean()), "projects_using_model": 0})
    rows.append({"rule": "model everywhere", "threshold": None,
                 "overall_MAE": float(np.average(mod_mae, weights=weights)),
                 "macro_MAE": float(mod_mae.mean()), "projects_using_model": 16})

    for t in THRESHOLDS:
        o, m, k = apply_rule(med_mae, mod_mae, std, t, weights)
        rows.append({"rule": f"model if train_std >= {t}", "threshold": t,
                     "overall_MAE": o, "macro_MAE": m, "projects_using_model": k})

    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))

    # the threshold rows only - pick the best by macro MAE
    swept = out[out.threshold.notna()]
    best = swept.loc[swept.macro_MAE.idxmin()]
    print(f"\nBest threshold on validation: {best.rule}"
          f"  (overall {best.overall_MAE:.3f}, macro {best.macro_MAE:.3f},"
          f" model used on {int(best.projects_using_model)} of 16)")
    print("Selected on validation - confirm once on the test split in week 9.")

    out.to_csv(f"{RESULTS}/week8_hybrid_sweep.csv", index=False)
    print(f"Saved {RESULTS}/week8_hybrid_sweep.csv")
