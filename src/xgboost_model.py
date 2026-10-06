"""Week 7: tune an XGBoost Regressor on the VALIDATION split.

Same two feature variants as the Random Forest (text / project). Each trial is a
(target scale, objective) pair - raw or log1p labels, squared or absolute error -
crossed with tree depth and min_child_weight. The number of trees is chosen by
early stopping on the validation split.

Run from the repository root:  python src/xgboost_model.py
"""
import itertools
import os
import time

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from evaluate import mae, score_table
from features import load_splits, to_matrix
from random_forest import add_project

RESULTS = "results"
TARGET_OBJECTIVE = [
    ("raw", "reg:squarederror"),
    ("raw", "reg:absoluteerror"),
    ("log", "reg:squarederror"),
    ("log", "reg:absoluteerror"),
]
MAX_DEPTH = [4, 6]
MIN_CHILD_WEIGHT = [1, 5]


def fit_predict(Xtr, ytr, Xva, yva, target, objective, **params):
    f = np.log1p if target == "log" else (lambda v: v)
    model = XGBRegressor(
        objective=objective, n_estimators=1000, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.3, tree_method="hist", n_jobs=-1, random_state=42,
        early_stopping_rounds=50, **params,
    )
    model.fit(Xtr, f(ytr), eval_set=[(Xva, f(yva))], verbose=False)
    p = model.predict(Xva)
    return model, np.clip(np.expm1(p) if target == "log" else p, 1, None)


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
    for variant, (Xtr, Xva) in feats.items():
        for (target, obj), depth, mcw in itertools.product(TARGET_OBJECTIVE, MAX_DEPTH, MIN_CHILD_WEIGHT):
            t0 = time.time()
            model, pred = fit_predict(Xtr, ytr, Xva, yva, target, obj, max_depth=depth, min_child_weight=mcw)
            m = mae(yva, pred)
            log.append({"variant": variant, "target": target, "objective": obj, "max_depth": depth,
                        "min_child_weight": mcw, "trees": model.best_iteration + 1,
                        "val_MAE": m, "secs": round(time.time() - t0)})
            print(log[-1], flush=True)
            if variant not in best or m < best[variant][0]:
                best[variant] = (m, target, obj, depth, mcw, pred, model)

    pd.DataFrame(log).to_csv(f"{RESULTS}/week7_xgb_grid.csv", index=False)
    tables = []
    for variant, (m, target, obj, depth, mcw, pred, model) in best.items():
        name = f"xgboost [{variant}] ({target}, {obj.split(':')[1]}, depth={depth}, mcw={mcw})"
        tables.append(score_table(val, pred, train, name))
        model.save_model(f"models/xgb_{variant}.json")
    res = pd.concat(tables, ignore_index=True)
    res.to_csv(f"{RESULTS}/week7_xgb_validation_metrics.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[res.project.isin(["overall", "macro"])].round(3).to_string(index=False))
