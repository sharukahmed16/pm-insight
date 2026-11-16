"""Week 12: can a large language model estimate story points?

Board feedback after the midterm: "explore other models and LLM models".
A local open model is run through Ollama (free, offline, nothing leaves the
laptop) in two settings:

  zero     the issue text only; the model must guess the team's scale itself
  fewshot  the issue text plus the 5 most similar TRAINING issues of the same
           project with their real story points. Similarity = Sentence-BERT
           cosine, the same retrieval as the week 10 kNN.

Nothing is tuned: the prompts, k=5, temperature 0 and the sample size are
fixed in this file before any test result is seen, so the test split can be
used directly. Use --split val (with a small --n) only to check that the
pipeline works.

Because LLM calls are slow on a laptop CPU, the evaluation uses a stratified
sample: the same N randomly chosen issues per project (seed 42). On exactly
the same issues the script also scores the median and mean baselines, the
week 9 Random Forest and the week 10 SBERT + kNN, so every row of the result
table is comparable.

Every LLM answer is appended to a cache file as soon as it arrives, so an
interrupted run continues where it stopped.

Run from the repository root, with Ollama running:
    ollama pull qwen2.5:7b
    python src/llm_estimate.py --split val --n 3            # quick pipeline check
    python src/llm_estimate.py --split test --n 20          # the real run
    python src/llm_estimate.py --split test --n 20 --model llama3.2:3b   # faster model
"""
import argparse
import json
import os
import re
import time
import urllib.request

import joblib
import numpy as np
import pandas as pd

from baselines import per_project_constant
from compare_representations import knn_predict, representation
from embeddings import load_embedding, plain_text
from evaluate import score_table
from features import load_splits, to_matrix
from random_forest import add_project
from test_evaluation import criterion_check, rf_predict

RESULTS = "results"
SEED = 42
K_EXAMPLES = 5
MAX_CHARS = 700          # issue text shown to the model is cut to this length
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
SUMMARY_ROWS = ["overall", "macro"]


# ------------------------------------------------------------------ prompts
def short(text):
    t = plain_text(text)
    return t if len(t) <= MAX_CHARS else t[:MAX_CHARS] + " ..."


def prompt_zero(project, text):
    return (
        f"You are an experienced developer on the '{project}' open-source project's agile team.\n"
        "Estimate the size of the following work item in story points, as your team would.\n\n"
        f"Work item:\n{short(text)}\n\n"
        'Reply with JSON only, in the form {"story_points": <number>}.'
    )


def prompt_fewshot(project, text, examples):
    shown = "\n\n".join(
        f"Example {i + 1} - story points: {sp}\n{short(t)}" for i, (t, sp) in enumerate(examples)
    )
    return (
        f"You are an experienced developer on the '{project}' open-source project's agile team.\n"
        "Below are past work items from this team, the most similar ones to the new item, "
        "with the story points the team actually gave them.\n\n"
        f"{shown}\n\n"
        "Estimate the new work item in story points on the same scale as the examples.\n\n"
        f"New work item:\n{short(text)}\n\n"
        'Reply with JSON only, in the form {"story_points": <number>}.'
    )


# ------------------------------------------------------------- ollama client
def ask(model, prompt):
    """One call to the local Ollama server. Returns the raw reply text."""
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,                       # no hidden reasoning, for models that have it
        "format": {"type": "object",
                   "properties": {"story_points": {"type": "number"}},
                   "required": ["story_points"]},
        "options": {"temperature": 0, "seed": SEED, "num_predict": 30},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["response"]


def parse_points(reply):
    """Story points from the reply, or None if the reply is unusable."""
    try:
        v = float(json.loads(reply)["story_points"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        m = re.search(r"\d+(\.\d+)?", str(reply))
        v = float(m.group()) if m else None
    if v is None or not np.isfinite(v) or v <= 0:
        return None
    return v


# ------------------------------------------------------------------ sample
def stratified_sample(df, n, seed=SEED):
    """n random issues per project (all of them if a project has fewer)."""
    parts = [g.sample(min(n, len(g)), random_state=seed) for _, g in df.groupby("project")]
    return pd.concat(parts).sort_index()


def nearest_examples(train, sample, pos, split, k=K_EXAMPLES):
    """For each sampled issue, its k most similar TRAINING issues of the same project.

    pos[j] is the row position of sample row j inside the full split, which is
    where its cached Sentence-BERT vector lives. Returns {sample index: [(text, sp), ...]}.
    """
    emb = load_embedding("sbert")
    if emb is None:
        raise SystemExit("Sentence-BERT vectors not found - run: python src/embeddings.py sbert")
    unit = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)   # so dot product = cosine
    tr_vecs, ev_vecs = unit(emb["train"]), unit(emb[split][pos])
    tr_project = train.project.to_numpy()
    out = {}
    for j, (i, row) in enumerate(sample.iterrows()):
        cand = np.flatnonzero(tr_project == row.project)
        top = cand[np.argsort(-(tr_vecs[cand] @ ev_vecs[j]))[:k]]
        out[i] = [(train.text.iat[t], int(train.storypoint.iat[t])) for t in top]
    return out


# -------------------------------------------------------------------- main
def run_llm(mode, model, sample, examples, cache_path):
    done = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done[rec["issuekey"]] = rec
    todo = [i for i in sample.index if sample.issuekey[i] not in done]
    print(f"\n[{mode}] {len(sample) - len(todo)} cached, {len(todo)} to ask", flush=True)
    t0 = time.time()
    with open(cache_path, "a", encoding="utf-8") as f:
        for n, i in enumerate(todo, 1):
            row = sample.loc[i]
            prompt = (prompt_zero(row.project, row.text) if mode == "zero"
                      else prompt_fewshot(row.project, row.text, examples[i]))
            reply = ask(model, prompt)
            rec = {"issuekey": row.issuekey, "project": row.project,
                   "reply": reply, "points": parse_points(reply)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[row.issuekey] = rec
            if n % 10 == 0 or n == len(todo):
                rate = (time.time() - t0) / n
                print(f"  {n}/{len(todo)}  {rate:.1f}s per item, "
                      f"about {rate * (len(todo) - n) / 60:.0f} min left", flush=True)
    return [done[k]["points"] for k in sample.issuekey]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--n", type=int, default=20, help="issues per project")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--modes", default="zero,fewshot")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    splits = load_splits()
    train, ev = splits["train"], splits[args.split]
    projects = sorted(train.project.unique())
    sample = stratified_sample(ev, args.n)
    tag = f"{args.split}_n{args.n}"
    sample[["project", "issuekey", "storypoint"]].to_csv(f"{RESULTS}/week12_sample_{tag}.csv", index=False)
    print(f"{args.split.upper()} sample: {len(sample)} issues ({args.n} per project), model {args.model}")

    # positions of the sampled rows inside the full split, for the cached vectors
    pos = ev.index.get_indexer(sample.index)
    tables = [
        score_table(sample, per_project_constant(train, sample, "mean"), train, "mean (per project)"),
        score_table(sample, per_project_constant(train, sample, "median"), train, "median (per project)"),
    ]
    vec = joblib.load("models/tfidf.joblib")
    rf = rf_predict(joblib.load("models/rf_project.joblib"),
                    add_project(to_matrix(vec, sample), sample, projects))
    tables.append(score_table(sample, rf, train, "week 9: tfidf + rf"))
    Xtr, Xev = representation("sbert", splits, parts=("train", args.split))
    knn = knn_predict(Xtr, Xev[pos], train, sample.reset_index(drop=True), 80)
    tables.append(score_table(sample, knn, train, "week 10: sbert + knn (k=80)"))

    examples = None
    median_pred = per_project_constant(train, sample, "median")
    for mode in args.modes.split(","):
        if mode == "fewshot" and examples is None:
            examples = nearest_examples(train, sample, pos, args.split)
        safe = args.model.replace(":", "-").replace("/", "-")
        pts = run_llm(mode, args.model, sample, examples,
                      f"{RESULTS}/week12_llm_{safe}_{mode}_{tag}.jsonl")
        failed = sum(p is None for p in pts)
        if failed:
            print(f"  {failed} unusable replies -> replaced by the project median")
        pred = np.array([m if p is None else p for p, m in zip(pts, median_pred)], dtype=float)
        tables.append(score_table(sample, pred, train, f"LLM {args.model} {mode}"))

    res = pd.concat(tables, ignore_index=True)
    out = f"{RESULTS}/week12_llm_{tag}.csv"
    res.to_csv(out, index=False)

    pd.set_option("display.width", 250)
    head = res[res.project.isin(SUMMARY_ROWS)].pivot(index="model", columns="project", values=["MAE", "SA"])
    print(f"\n=== {args.split.upper()} sample ({len(sample)} issues): overall and macro ===")
    print(head.sort_values(("MAE", "macro")).round(3).to_string())
    print("\n=== projects where the model beats BOTH baselines (on this sample) ===")
    for name in res.model.unique():
        if "per project" in name:
            continue
        wins, total, _ = criterion_check(res, name)
        print(f"  {name:<40} {wins:>2} / {total}")
    print(f"\nSaved {out}")
