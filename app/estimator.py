"""Story point suggestion, served from the week 9 model.

The week 10 and week 12 experiments did not beat this model on held-out data,
so it is the one the application uses (see README).

The model and the training issues are loaded once, on the first suggestion,
not when the server starts. Pages that do not estimate stay fast, and the
tests can run without the 45 MB model file.
"""
import html
import os
import re
import sys
from functools import lru_cache

import numpy as np

from app.config import ROOT

sys.path.insert(0, os.path.join(ROOT, "src"))

# planning-poker values; the model's number is rounded to the nearest one
SCALE = [1, 2, 3, 5, 8, 13, 20, 40, 100]

# the 16 teams of the Deep-SE dataset the model was trained on
REFERENCE_PROJECTS = [
    "appceleratorstudio", "aptanastudio", "bamboo", "clover", "datamanagement",
    "duracloud", "jirasoftware", "mesos", "moodle", "mule", "mulestudio",
    "springxd", "talenddataquality", "talendesb", "titanium", "usergrid",
]

FILES = {
    "vectorizer": os.path.join(ROOT, "models", "tfidf.joblib"),
    "model": os.path.join(ROOT, "models", "rf_project.joblib"),
    "train": os.path.join(ROOT, "data", "processed", "train.csv"),
}


class ModelUnavailable(Exception):
    """The model files are missing, so no suggestion can be made."""


@lru_cache(maxsize=1)
def _load():
    missing = [p for p in FILES.values() if not os.path.exists(p)]
    if missing:
        raise ModelUnavailable("Model files not found: " + ", ".join(missing))
    import joblib
    import pandas as pd

    vec = joblib.load(FILES["vectorizer"])
    model = joblib.load(FILES["model"])
    train = pd.read_csv(FILES["train"])
    projects = sorted(train.project.unique())
    assert projects == REFERENCE_PROJECTS, "training data does not match REFERENCE_PROJECTS"
    # TF-IDF rows are unit length, so a dot product between two rows is their cosine
    return vec, model, train, projects, vec.transform(train.text)


def plain(text, limit=160):
    """Training text without HTML tags, shortened for display."""
    t = re.sub(r"<[^>]+>", " ", html.unescape(str(text)))
    t = re.sub(r"\bhtml\b", " ", t)        # the export wraps descriptions in "html ... html"
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) <= limit else t[:limit] + "..."


def suggest(title, description, reference_project):
    """{"points", "raw", "similar": [3 most similar past issues]} for one new item."""
    from features import to_matrix          # the same feature code as training
    from prepare import clean_text
    from random_forest import add_project

    vec, model, train, projects, train_tfidf = _load()
    if reference_project not in projects:
        raise ValueError(f"unknown reference project: {reference_project}")

    title, description = clean_text(title), clean_text(description)
    text = (title + " " + description).strip()
    import pandas as pd
    row = pd.DataFrame([{"project": reference_project, "text": text,
                         "has_description": int(description != "")}])
    X = add_project(to_matrix(vec, row), row, projects)

    # the model was trained on log1p(story points), so undo that
    raw = float(np.clip(np.expm1(model.predict(X))[0], 1, None))
    points = min(SCALE, key=lambda s: abs(s - raw))

    same = (train.project == reference_project).to_numpy()
    scores = (train_tfidf[same] @ vec.transform([text]).T).toarray().ravel()
    rows = train[same].reset_index(drop=True)
    top = np.argsort(-scores)[:3]
    similar = [{"key": rows.issuekey[i], "text": plain(rows.text[i]),
                "points": int(rows.storypoint[i])} for i in top]
    return {"points": points, "raw": round(raw, 2), "similar": similar}
