"""Week 5: mean / median / random baselines and Ridge regression, scored on the
VALIDATION split (the test split is reserved for the final evaluation in Week 9).

Run from the repository root:  python src/baselines.py
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from evaluate import mae, mdae, random_guess_mae, random_guess_mdae, score_table, standardized_accuracy
from features import load_splits, to_matrix

RESULTS = "results"
ALPHAS = [0.1, 0.3, 1, 3, 10, 30, 100]


def per_project_constant(train, target, stat):
    """Prediction for each row of `target` = statistic of that project's training labels."""
    s = train.groupby("project").storypoint.agg(stat)
    return target.project.map(s).to_numpy()


def random_rows(val, train):
    """Random guessing: MAE is the exact expectation, MdAE a seeded simulation."""
    rows = []
    for proj, g in val.groupby("project"):
        tr = train.loc[train.project == proj, "storypoint"].to_numpy()
        y = g.storypoint.to_numpy()
        rows.append((proj, len(g), random_guess_mae(y, tr), random_guess_mdae(y, tr), 0.0))
    t = pd.DataFrame(rows, columns=["project", "n", "MAE", "MdAE", "SA"])
    w = t.n.to_numpy()
    all_rows = [("overall", int(w.sum()), float(np.average(t.MAE, weights=w)), float(np.average(t.MdAE, weights=w)), 0.0),
                ("macro", int(w.sum()), t.MAE.mean(), t.MdAE.mean(), 0.0)]
    t = pd.concat([t, pd.DataFrame(all_rows, columns=t.columns)], ignore_index=True)
    t.insert(0, "model", "random")
    return t


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    splits = load_splits()
    train, val = splits["train"], splits["val"]
    vec = joblib.load("models/tfidf.joblib")
    Xtr, Xva = to_matrix(vec, train), to_matrix(vec, val)
    ytr = train.storypoint.to_numpy(float)

    tables = [
        score_table(val, per_project_constant(train, val, "mean"), train, "mean (per project)"),
        score_table(val, per_project_constant(train, val, "median"), train, "median (per project)"),
        score_table(val, np.full(len(val), ytr.mean()), train, "mean (global)"),
        score_table(val, np.full(len(val), np.median(ytr)), train, "median (global)"),
        random_rows(val, train),
    ]

    # Ridge: choose alpha by validation MAE
    best = None
    for a in ALPHAS:
        pred = np.clip(Ridge(alpha=a).fit(Xtr, ytr).predict(Xva), 1, None)
        m = mae(val.storypoint.to_numpy(float), pred)
        print(f"ridge alpha={a:<5} val MAE={m:.3f}")
        if best is None or m < best[0]:
            best = (m, a, pred)
    _, alpha, pred = best
    tables.append(score_table(val, pred, train, f"ridge (alpha={alpha})"))

    res = pd.concat(tables, ignore_index=True)
    res.to_csv(f"{RESULTS}/week5_validation_metrics.csv", index=False)
    pd.set_option("display.width", 250)
    summary = res[res.project.isin(["overall", "macro"])].pivot(index="model", columns="project", values=["MAE", "MdAE", "SA"]).round(3)
    print(summary.to_string())
