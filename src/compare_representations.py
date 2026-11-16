"""Week 10: which text representation x which model works best?

Board feedback: (1) better embeddings, (2) explore other models. This script
crosses every representation built by embeddings.py (plus the original TF-IDF)
with five regressors, and scores everything on the VALIDATION split with the
same metrics as weeks 5-9. The test split is NOT used here; the winner is
evaluated on test later, the same way week 9 did it.

Models
  ridge  linear regression with L2 penalty
  knn    analogy-based estimation: the median story point of the k most similar
         TRAINING issues of the same project (cosine similarity)
  svr    support vector regression, RBF kernel
  rf     random forest
  xgb    gradient boosting (skipped if xgboost is not installed)

All models except knn get the project one-hot column, as in week 6, and are
trained on log1p(story points); predictions are mapped back with expm1.

Run from the repository root (build the embeddings first):
    python src/compare_representations.py                 # everything found
    python src/compare_representations.py tfidf w2v_own   # only these
"""
import itertools
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVR

from baselines import per_project_constant
from embeddings import BUILDERS, load_embedding
from evaluate import mae, score_table
from features import load_splits, to_matrix

RESULTS = "results"
SEED = 42
SUMMARY_ROWS = ["overall", "macro"]


# --------------------------------------------------------------- features
def project_onehot(df, projects):
    return pd.get_dummies(df.project).reindex(columns=projects, fill_value=0).to_numpy(float)


def representation(name, splits, parts=("train", "val")):
    """Feature matrices for the requested splits, or None if not built yet."""
    if name == "tfidf":
        vec = joblib.load("models/tfidf.joblib")
        return tuple(to_matrix(vec, splits[p]) for p in parts)
    emb = load_embedding(name)
    if emb is None:
        return None
    # scaler fitted on training vectors only
    sc = StandardScaler().fit(emb["train"])
    return tuple(sc.transform(emb[p]) for p in parts)


def with_project(X, df, projects):
    oh = project_onehot(df, projects)
    if sparse.issparse(X):
        return sparse.hstack([X, sparse.csr_matrix(oh)], format="csr")
    return np.hstack([X, oh])


# ----------------------------------------------------------------- models
def back(p):
    return np.clip(np.expm1(p), 1, None)


def fit_ridge(Xtr, ytr, Xva, alpha):
    return back(Ridge(alpha=alpha).fit(Xtr, np.log1p(ytr)).predict(Xva))


def fit_svr(Xtr, ytr, Xva, C):
    return back(SVR(kernel="rbf", C=C, epsilon=0.1, gamma="scale").fit(Xtr, np.log1p(ytr)).predict(Xva))


def fit_rf(Xtr, ytr, Xva, max_features, min_samples_leaf):
    rf = RandomForestRegressor(n_estimators=200, max_features=max_features,
                               min_samples_leaf=min_samples_leaf, n_jobs=-1, random_state=SEED)
    return back(rf.fit(Xtr, np.log1p(ytr)).predict(Xva))


def fit_xgb(Xtr, ytr, Xva, max_depth, learning_rate):
    from xgboost import XGBRegressor
    m = XGBRegressor(n_estimators=400, max_depth=max_depth, learning_rate=learning_rate,
                     subsample=0.8, colsample_bytree=0.5, random_state=SEED, n_jobs=-1)
    return back(m.fit(Xtr, np.log1p(ytr)).predict(Xva))


def knn_predict(Xtr, Xva, train, val, k):
    """Analogy-based estimation, within each project, cosine similarity."""
    Xtr, Xva = normalize(Xtr), normalize(Xva)       # unit length -> dot product = cosine
    ytr = train.storypoint.to_numpy(float)
    pred = np.empty(len(val))
    for proj in val.project.unique():
        tr = np.flatnonzero(train.project.to_numpy() == proj)
        va = np.flatnonzero(val.project.to_numpy() == proj)
        sims = Xva[va] @ Xtr[tr].T
        sims = sims.toarray() if sparse.issparse(sims) else sims
        kk = min(k, len(tr))
        top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        pred[va] = np.median(ytr[tr][top], axis=1)
    return pred


def grids(sparse_input):
    g = {
        "ridge": {"alpha": [1, 10, 100, 1000]},
        "knn": {"k": [10, 20, 40, 80, 160]},
        "svr": {"C": [0.3, 1, 3]},
        "rf": {"max_features": [0.05, "sqrt"] if sparse_input else [0.33, "sqrt"],
               "min_samples_leaf": [1, 5]},
    }
    try:
        import xgboost  # noqa: F401
        g["xgb"] = {"max_depth": [4, 6], "learning_rate": [0.05]}
    except ImportError:
        print("xgboost not installed - skipping xgb")
    return g


def run_model(model, params, Xtr, Xva, Xtr_p, Xva_p, train, val, ytr):
    if model == "knn":
        return knn_predict(Xtr, Xva, train, val, **params)
    fn = {"ridge": fit_ridge, "svr": fit_svr, "rf": fit_rf, "xgb": fit_xgb}[model]
    return fn(Xtr_p, ytr, Xva_p, **params)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    splits = load_splits()
    train, val = splits["train"], splits["val"]
    ytr, yva = train.storypoint.to_numpy(float), val.storypoint.to_numpy(float)
    projects = sorted(train.project.unique())

    wanted = sys.argv[1:] or ["tfidf", *BUILDERS]
    tables = [
        score_table(val, per_project_constant(train, val, "mean"), train, "mean (per project)"),
        score_table(val, per_project_constant(train, val, "median"), train, "median (per project)"),
    ]
    for c in ("representation", "family"):
        for t in tables:
            t.insert(1, c, "baseline")
    log = []

    for rep in wanted:
        Xs = representation(rep, splits)
        if Xs is None:
            print(f"\n[{rep}] not built yet - run: python src/embeddings.py {rep}")
            continue
        Xtr, Xva = Xs
        Xtr_p, Xva_p = with_project(Xtr, train, projects), with_project(Xva, val, projects)
        print(f"\n[{rep}] {Xtr.shape[1]} features")
        for model, grid in grids(sparse.issparse(Xtr)).items():
            best = None
            for combo in itertools.product(*grid.values()):
                params = dict(zip(grid, combo))
                t0 = time.time()
                pred = run_model(model, params, Xtr, Xva, Xtr_p, Xva_p, train, val, ytr)
                m = mae(yva, pred)
                log.append({"representation": rep, "model": model, **params,
                            "val_MAE": round(m, 4), "secs": round(time.time() - t0, 1)})
                print(f"  {model:<5} {params}  val MAE={m:.3f}  ({log[-1]['secs']}s)", flush=True)
                if best is None or m < best[0]:
                    best = (m, params, pred)
            _, params, pred = best
            label = ", ".join(f"{k}={v}" for k, v in params.items())
            t = score_table(val, pred, train, f"{rep} + {model} ({label})")
            t.insert(1, "representation", rep)
            t.insert(2, "family", model)
            tables.append(t)

    res = pd.concat(tables, ignore_index=True)
    res.to_csv(f"{RESULTS}/week10_representation_validation.csv", index=False)
    pd.DataFrame(log).to_csv(f"{RESULTS}/week10_representation_grid.csv", index=False)

    # ---- summary: macro MAE, representation x model
    pd.set_option("display.width", 250)
    macro = res[(res.project == "macro") & (res.family != "baseline")]
    print("\n=== VALIDATION macro MAE (lower is better) ===")
    print(macro.pivot(index="representation", columns="family", values="MAE").round(3).to_string())
    base = res[(res.project == "macro") & (res.family == "baseline")][["model", "MAE"]]
    print("\nbaselines:", ", ".join(f"{m} {v:.3f}" for m, v in base.itertuples(index=False)))

    # ---- how many projects each configuration beats BOTH baselines on
    per = res[~res.project.isin(SUMMARY_ROWS)].pivot(index="project", columns="model", values="MAE")
    floor = per[["mean (per project)", "median (per project)"]].min(axis=1)
    wins = {m: int((per[m] < floor).sum()) for m in per.columns if "per project" not in m}
    print("\n=== projects (of 16) where the model beats BOTH baselines ===")
    print(pd.Series(wins).sort_values(ascending=False).to_string())
    print(f"\nSaved {RESULTS}/week10_representation_validation.csv and week10_representation_grid.csv")
