"""PM-Insight demo: suggest a story point value for a new task.

Run from the repository root:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000
"""
import sys

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# reuse the same functions the pipeline used, so the features match training
sys.path.insert(0, "src")
from features import to_matrix          # noqa: E402
from prepare import clean_text          # noqa: E402
from random_forest import add_project   # noqa: E402

SCALE = [1, 2, 3, 5, 8, 13, 20, 40, 100]

# loaded once when the server starts
vec = joblib.load("models/tfidf.joblib")
model = joblib.load("models/rf_project.joblib")
train = pd.read_csv("data/processed/train.csv")
projects = sorted(train.project.unique())
train_tfidf = vec.transform(train.text)

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")


def estimate(title, description, project):
    """Predicted story points, plus the 3 most similar past issues."""
    title, description = clean_text(title), clean_text(description)
    text = (title + " " + description).strip()

    # one row shaped exactly like a training row
    row = pd.DataFrame([{"project": project, "text": text,
                         "has_description": int(description != "")}])
    X = add_project(to_matrix(vec, row), row, projects)

    # the model was trained on log1p(storypoint), so undo that
    raw = float(np.clip(np.expm1(model.predict(X))[0], 1, None))
    points = min(SCALE, key=lambda s: abs(s - raw))

    # cosine similarity against this project's training issues
    # (TF-IDF vectors are already normalised, so a dot product is the cosine)
    same = train.project == project
    scores = (train_tfidf[same.to_numpy()] @ vec.transform([text]).T).toarray().ravel()
    rows = train[same].reset_index(drop=True)
    top = np.argsort(-scores)[:3]
    similar = [{"text": rows.text[i], "storypoint": int(rows.storypoint[i])}
               for i in top]

    return {"points": points, "raw": round(raw, 2), "similar": similar}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"projects": projects})


@app.post("/", response_class=HTMLResponse)
def submit(request: Request,
           project: str = Form(...),
           title: str = Form(...),
           description: str = Form("")):
    result = estimate(title, description, project) if title.strip() else None
    return templates.TemplateResponse(request, "index.html", {
        "projects": projects, "project": project,
        "title": title, "description": description, "result": result,
    })
