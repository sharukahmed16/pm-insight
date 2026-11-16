"""Week 10: final evaluation of the embedding x model comparison on the TEST split.

Protocol, fixed before this script is run with --final:

  * Every choice is read from the VALIDATION results written by
    compare_representations.py - nothing is chosen here:
      - for each representation x model, the setting with the lowest
        validation MAE (results/week10_representation_grid.csv);
      - the WINNER is the configuration with the lowest validation macro MAE
        (results/week10_representation_validation.csv), the same rule that
        picked the Random Forest over XGBoost in week 9.
  * Models are fitted on the TRAINING split only, as in week 9.
  * The week 9 model (models/rf_project.joblib) is scored alongside, so the
    gain from the board's suggestions can be read off directly.

Run from the repository root:
    python src/test_evaluation_week10.py           # check run: scores VALIDATION,
                                                   # must reproduce the week 10 table
    python src/test_evaluation_week10.py --final   # the one-off TEST evaluation
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

from baselines import per_project_constant, random_rows
from compare_representations import representation, run_model, with_project
from evaluate import score_table
from features import load_splits, to_matrix
from random_forest import add_project
from test_evaluation import criterion_check, rf_predict

RESULTS = "results"
SUMMARY_ROWS = ["overall", "macro"]
PARAM_TYPES = {"alpha": float, "k": int, "C": float, "min_samples_leaf": int,
               "max_depth": int, "learning_rate": float}


def best_params(grid, rep, model):
    """The validation-best setting of one representation x model, from the grid log."""
    rows = grid[(grid.representation == rep) & (grid.model == model)]
    row = rows.loc[rows.val_MAE.idxmin()]
    params = {}
    for col in grid.columns.drop(["representation", "model", "val_MAE", "secs"]):
        v = row[col]
        if pd.isna(v):
            continue
        if col == "max_features":           # stored as text: "sqrt" or a fraction
            params[col] = v if v == "sqrt" else float(v)
        else:
            params[col] = PARAM_TYPES[col](v)
    return params


def xgb_available():
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def selected_configs():
    """(winner, all configurations) as chosen on validation."""
    val = pd.read_csv(f"{RESULTS}/week10_representation_validation.csv")
    macro = val[(val.project == "macro") & (val.family != "baseline")]
    winner = macro.loc[macro.MAE.idxmin()]
    return (winner.representation, winner.family), \
        list(macro[["representation", "family"]].itertuples(index=False, name=None))


if __name__ == "__main__":
    final = "--final" in sys.argv
    target = "test" if final else "val"
    os.makedirs(RESULTS, exist_ok=True)
    splits = load_splits()
    train, ev = splits["train"], splits[target]
    ytr = train.storypoint.to_numpy(float)
    projects = sorted(train.project.unique())
    grid = pd.read_csv(f"{RESULTS}/week10_representation_grid.csv")
    (win_rep, win_model), configs = selected_configs()
    print(f"Scoring the {target.upper()} split. Winner chosen on validation: {win_rep} + {win_model}")

    tables = [
        score_table(ev, per_project_constant(train, ev, "mean"), train, "mean (per project)"),
        score_table(ev, per_project_constant(train, ev, "median"), train, "median (per project)"),
        random_rows(ev, train),
    ]
    # the week 9 model, unchanged
    vec = joblib.load("models/tfidf.joblib")
    week9 = rf_predict(joblib.load("models/rf_project.joblib"),
                       add_project(to_matrix(vec, ev), ev, projects))
    tables.append(score_table(ev, week9, train, "week 9: tfidf + rf"))

    names = {}
    for rep, model in configs:
        if model == "xgb" and not xgb_available():
            print(f"  skipped {rep} + xgb (xgboost not installed)")
            continue
        params = best_params(grid, rep, model)
        Xtr, Xev = representation(rep, splits, parts=("train", target))
        pred = run_model(model, params, Xtr, Xev,
                         with_project(Xtr, train, projects), with_project(Xev, ev, projects),
                         train, ev, ytr)
        label = ", ".join(f"{k}={v}" for k, v in params.items())
        names[(rep, model)] = f"{rep} + {model} ({label})"
        tables.append(score_table(ev, pred, train, names[(rep, model)]))
        print(f"  scored {names[(rep, model)]}", flush=True)

    res = pd.concat(tables, ignore_index=True)
    out = f"{RESULTS}/week10_{'test' if final else 'check_val'}_metrics.csv"
    res.to_csv(out, index=False)

    pd.set_option("display.width", 250)
    head = res[res.project.isin(SUMMARY_ROWS)].pivot(index="model", columns="project", values=["MAE", "SA"])
    print(f"\n=== {target.upper()} split: overall and macro ===")
    print(head.sort_values(("MAE", "macro")).round(3).to_string())

    print("\n=== Success criterion: lower MAE than BOTH baselines on a majority of 16 projects ===")
    for name in ["week 9: tfidf + rf", names[(win_rep, win_model)]]:
        wins, total, _ = criterion_check(res, name)
        print(f"{name[:60]:<62} {wins:>2} / {total}   {'MET' if wins > total / 2 else 'NOT MET'}")
    if not final:
        print("\nThis was the check run on VALIDATION. Its macro MAE column must equal the "
              "week 10 table.\nWhen all embeddings are built and compared, run once with --final.")
    print(f"\nSaved {out}")
