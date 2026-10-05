"""Evaluation metrics shared by all models: MAE, MdAE and Standardized Accuracy."""
import numpy as np
import pandas as pd


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def mdae(y, p):
    return float(np.median(np.abs(y - p)))


def random_guess_mae(y, train_y):
    """Exact expected MAE of guessing a random training label for every issue."""
    return float(np.mean(np.abs(y[:, None] - train_y[None, :])))


def random_guess_mdae(y, train_y, repeats=200, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.mean([mdae(y, rng.choice(train_y, size=len(y))) for _ in range(repeats)]))


def standardized_accuracy(mae_model, mae_random):
    return (1 - mae_model / mae_random) * 100


def score_table(df, pred, train_df, name):
    """Per-project and overall MAE / MdAE / SA for one set of predictions.

    df, train_df: frames with columns project, storypoint (df rows align with pred).
    The random-guess reference for each project is drawn from that project's
    training labels; the overall row pools all issues, and 'macro' averages projects.
    """
    rows = []
    err = pd.DataFrame({"project": df.project.to_numpy(), "y": df.storypoint.to_numpy(), "p": pred})
    rand_by_project = {}
    for proj, g in err.groupby("project"):
        tr = train_df.loc[train_df.project == proj, "storypoint"].to_numpy()
        r = random_guess_mae(g.y.to_numpy(), tr)
        rand_by_project[proj] = r
        m = mae(g.y.to_numpy(), g.p.to_numpy())
        rows.append((proj, len(g), m, mdae(g.y.to_numpy(), g.p.to_numpy()), standardized_accuracy(m, r)))
    out = pd.DataFrame(rows, columns=["project", "n", "MAE", "MdAE", "SA"])
    w = out.n.to_numpy()
    pooled_mae = float(np.average(out.MAE, weights=w))
    pooled_random = float(np.average(list(rand_by_project.values()), weights=w))
    overall = ("overall", int(w.sum()), pooled_mae, mdae(err.y.to_numpy(), err.p.to_numpy()),
               standardized_accuracy(pooled_mae, pooled_random))
    macro = ("macro", int(w.sum()), out.MAE.mean(), out.MdAE.mean(),
             float(np.mean([standardized_accuracy(m, rand_by_project[p]) for p, m in zip(out.project, out.MAE)])))
    out = pd.concat([out, pd.DataFrame([overall, macro], columns=out.columns)], ignore_index=True)
    out.insert(0, "model", name)
    return out
