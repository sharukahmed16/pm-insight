"""Week 6: tune a Random Forest Regressor on the VALIDATION split.

Two feature variants are compared:
  text     - TF-IDF + has_description (deployable for any new project)
  project  - the same plus a one-hot project column (needs a known project)
and two target scales: raw story points, or log1p (predictions mapped back with expm1).

Run from the repository root:  python src/random_forest.py
"""
import itertools
import os
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor

from evaluate import mae, score_table
from features import load_splits, to_matrix

RESULTS = "results"
GRID = {
    "target": ["raw", "log"],
    "min_samples_leaf": [1, 3, 5],
    "max_features": ["sqrt", 0.05],
}
N_TREES = 200


def add_project(X, df, projects):
    onehot = sparse.csr_matrix(pd.get_dummies(df.project).reindex(columns=projects, fill_value=0).to_numpy(float))
    return sparse.hstack([X, onehot], format="csr")


def fit_predict(Xtr, ytr, Xva, target, **params):
    rf = RandomForestRegressor(n_estimators=N_TREES, n_jobs=-1, random_state=42, **params)
    rf.fit(Xtr, np.log1p(ytr) if target == "log" else ytr)
    p = rf.predict(Xva)
    return rf, np.clip(np.expm1(p) if target == "log" else p, 1, None)


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    splits = load_splits()
    train, val = splits["train"], splits["val"]
    vec = joblib.load("models/tfidf.joblib")
    projects = sorted(train.project.unique())
    base_tr, base_va = to_matrix(vec, train), to_matrix(vec, val)
    feats = {
        "text": (base_tr, base_va),
        "project": (add_project(base_tr, train, projects), add_project(base_va, val, projects)),
    }
    ytr, yva = train.storypoint.to_numpy(float), val.storypoint.to_numpy(float)

    log, best = [], {}
    keys = list(GRID)
    for variant, (Xtr, Xva) in feats.items():
        for combo in itertools.product(*GRID.values()):
            cfg = dict(zip(keys, combo))
            target = cfg.pop("target")
            t0 = time.time()
            rf, pred = fit_predict(Xtr, ytr, Xva, target, **cfg)
            m = mae(yva, pred)
            log.append({"variant": variant, "target": target, **cfg, "val_MAE": m, "secs": round(time.time() - t0)})
            print(log[-1], flush=True)
            if variant not in best or m < best[variant][0]:
                best[variant] = (m, target, cfg, pred, rf)

    pd.DataFrame(log).to_csv(f"{RESULTS}/week6_rf_grid.csv", index=False)
    tables = []
    for variant, (m, target, cfg, pred, rf) in best.items():
        name = f"random forest [{variant}] ({target}, leaf={cfg['min_samples_leaf']}, mf={cfg['max_features']})"
        tables.append(score_table(val, pred, train, name))
        joblib.dump(rf, f"models/rf_{variant}.joblib", compress=3)
    res = pd.concat(tables, ignore_index=True)
    res.to_csv(f"{RESULTS}/week6_rf_validation_metrics.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[res.project.isin(["overall", "macro"])].round(3).to_string(index=False))
