"""Week 2: descriptive statistics per project for the Deep-SE story point CSVs."""
import glob
import os

import pandas as pd

SCALE = [1, 2, 3, 5, 8, 13, 20, 40, 100]


def summarise(path):
    d = pd.read_csv(path)
    text = d.title.fillna("") + " " + d.description.fillna("")
    sp = d.storypoint
    return {
        "project": os.path.basename(path)[:-4],
        "n": len(d),
        "no_description": int((d.description.fillna("").str.strip() == "").sum()),
        "duplicate_text": int(text.duplicated().sum()),
        "mean": round(sp.mean(), 2),
        "median": sp.median(),
        "std": round(sp.std(), 2),
        "min": sp.min(),
        "max": sp.max(),
        "distinct_values": sp.nunique(),
        "off_scale": int((~sp.isin(SCALE)).sum()),
    }


if __name__ == "__main__":
    files = sorted(glob.glob("data/raw/*.csv"))
    summary = pd.DataFrame([summarise(f) for f in files])
    pd.set_option("display.width", 250)
    print(summary.to_string(index=False))
    print("total issues:", summary.n.sum())
