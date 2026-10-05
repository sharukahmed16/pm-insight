"""Week 3: clean the Deep-SE CSVs and split each project 60/20/20 in creation order.

Run from the repository root:  python src/prepare.py
Writes data/processed/{train,val,test}.csv and data/processed/cleaning_report.csv
"""
import glob
import os
import re

import pandas as pd

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2


def clean_text(s):
    """Normalise text for TF-IDF: drop markup/URLs/code fences, collapse whitespace."""
    s = "" if pd.isna(s) else str(s)
    s = re.sub(r"\{code[^}]*\}.*?\{code\}", " ", s, flags=re.S)  # JIRA code blocks
    s = re.sub(r"\{noformat\}.*?\{noformat\}", " ", s, flags=re.S)
    s = re.sub(r"https?://\S+", " URL ", s)
    s = re.sub(r"\[~[^\]]*\]", " ", s)  # JIRA user mentions
    s = re.sub(r"[{}\[\]|]|\*{2,}|_{2,}|h[1-6]\.", " ", s)  # wiki markup
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def load_project(path):
    project = os.path.basename(path)[:-4]
    d = pd.read_csv(path)
    d.insert(0, "project", project)
    # Creation date is not in the dataset; the numeric part of the key grows with
    # creation time inside a tracker, so it is used as the ordering.
    d["key_num"] = d.issuekey.str.extract(r"-(\d+)$")[0].astype(int)
    return d.sort_values("key_num", kind="stable").reset_index(drop=True)


def clean_project(d):
    stats = {"project": d.project.iloc[0], "raw": len(d)}
    d = d.copy()
    d["title"] = d.title.map(clean_text)
    d["description"] = d.description.map(clean_text)

    d = d[d.storypoint.notna() & (d.storypoint > 0)]
    d = d[d.title != ""]
    stats["after_incomplete"] = len(d)

    d["has_description"] = (d.description != "").astype(int)
    d["text"] = (d.title + " " + d.description).str.strip()

    # exact duplicate text inside a project: keep the earliest issue only, so the
    # same text cannot appear in both the training and the test split
    d = d.drop_duplicates(subset="text", keep="first")
    stats["after_dedup"] = len(d)
    return d.reset_index(drop=True), stats


def split_project(d):
    n = len(d)
    i_train = int(n * TRAIN_FRAC)
    i_val = int(n * (TRAIN_FRAC + VAL_FRAC))
    return d.iloc[:i_train], d.iloc[i_train:i_val], d.iloc[i_val:]


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    parts = {"train": [], "val": [], "test": []}
    report = []
    for path in sorted(glob.glob(f"{RAW_DIR}/*.csv")):
        d, stats = clean_project(load_project(path))
        for name, part in zip(parts, split_project(d)):
            parts[name].append(part)
            stats[name] = len(part)
        report.append(stats)

    cols = ["project", "issuekey", "key_num", "text", "has_description", "storypoint"]
    for name, frames in parts.items():
        pd.concat(frames)[cols].to_csv(f"{OUT_DIR}/{name}.csv", index=False)
    rep = pd.DataFrame(report)
    rep.to_csv(f"{OUT_DIR}/cleaning_report.csv", index=False)
    pd.set_option("display.width", 250)
    print(rep.to_string(index=False))
    print(rep.drop(columns="project").sum().to_string())
