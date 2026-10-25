"""Week 8: compare the baselines, Random Forest and XGBoost on the validation split.

Reads the result tables already written by weeks 5-7 and answers the success
criterion stated in the proposal: does the selected model beat BOTH the mean and
the median baseline on a majority of the 16 projects?

Nothing is trained here - this only reads CSVs from results/.

Run from the repository root:  python src/compare.py
"""
import os

import pandas as pd

RESULTS = "results"
SOURCES = [
    "week5_validation_metrics.csv",
    "week6_rf_validation_metrics.csv",
    "week7_xgb_validation_metrics.csv",
]
MEAN = "mean (per project)"
MEDIAN = "median (per project)"
SUMMARY_ROWS = ["overall", "macro"]


def load_all():
    """Stack every per-model result table written so far into one frame."""
    frames = [pd.read_csv(os.path.join(RESULTS, f)) for f in SOURCES]
    return pd.concat(frames, ignore_index=True)


def headline(res):
    """One row per model showing the pooled (overall) and averaged (macro) scores."""
    keep = res[res.project.isin(SUMMARY_ROWS)]
    return keep[["model", "project", "MAE", "MdAE", "SA"]].sort_values(["project", "MAE"])


def best_named(res, pattern):
    """Name of the model matching `pattern` with the lowest macro MAE."""
    macro = res[(res.project == "macro") & (res.model.str.contains(pattern, regex=True))]
    return macro.loc[macro.MAE.idxmin(), "model"]


def mae_by_project(res, model_name):
    """MAE for each of the 16 projects for one model, as a Series indexed by project."""
    rows = res[(res.model == model_name) & (~res.project.isin(SUMMARY_ROWS))]
    return rows.set_index("project")["MAE"]


def comparison(res, model_names):
    """Per-project MAE of the two baselines next to each candidate model.

    Adds a beats_both column per candidate: True where that model has a lower
    MAE than the mean AND the median baseline for that project.
    """
    out = pd.DataFrame({
        "n": res[(res.model == MEDIAN) & (~res.project.isin(SUMMARY_ROWS))]
             .set_index("project")["n"],
        "mean": mae_by_project(res, MEAN),
        "median": mae_by_project(res, MEDIAN),
    })
    for name in model_names:
        short = name.split(" (")[0]          # drop the hyperparameters from the label
        out[short] = mae_by_project(res, name)
        out[f"{short} beats both"] = (out[short] < out["mean"]) & (out[short] < out["median"])
    return out.reset_index()


if __name__ == "__main__":
    res = load_all()

    # pick the strongest Random Forest and XGBoost variant by macro MAE
    candidates = [best_named(res, r"random forest"), best_named(res, r"xgboost")]

    print("=== Headline scores (validation split) ===")
    print(headline(res).round(3).to_string(index=False))

    table = comparison(res, candidates)
    print()
    print("=== Per-project MAE vs baselines ===")
    print(table.round(3).to_string(index=False))

    print()
    print("=== Success criterion: beats BOTH baselines on a majority of 16 projects ===")
    for name in candidates:
        short = name.split(" (")[0]
        wins = int(table[f"{short} beats both"].sum())
        verdict = "MET" if wins > 8 else "NOT MET"
        print(f"{short:<26} {wins:>2} / 16   {verdict}")

    table.to_csv(f"{RESULTS}/week8_model_comparison.csv", index=False)
    print(f"\nSaved {RESULTS}/week8_model_comparison.csv")
