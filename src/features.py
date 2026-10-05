"""Week 4: TF-IDF feature pipeline.

The vectoriser is fitted on the training split only and reused unchanged for the
validation and test splits (and later by the application).

Run from the repository root:  python src/features.py
"""
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

PROCESSED = "data/processed"
MODELS = "models"


def make_vectorizer():
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10_000,
        min_df=2,
        sublinear_tf=True,
        stop_words="english",
    )


def to_matrix(vectorizer, df):
    """TF-IDF of the text plus the has_description flag as one extra column."""
    tfidf = vectorizer.transform(df.text)
    flag = sparse.csr_matrix(df.has_description.to_numpy(dtype=float).reshape(-1, 1))
    return sparse.hstack([tfidf, flag], format="csr")


def load_splits():
    return {n: pd.read_csv(f"{PROCESSED}/{n}.csv") for n in ("train", "val", "test")}


if __name__ == "__main__":
    splits = load_splits()
    vec = make_vectorizer().fit(splits["train"].text)
    joblib.dump(vec, f"{MODELS}/tfidf.joblib")
    for name, df in splits.items():
        X = to_matrix(vec, df)
        sparse.save_npz(f"{PROCESSED}/X_{name}.npz", X)
        np.save(f"{PROCESSED}/y_{name}.npy", df.storypoint.to_numpy(dtype=float))
        print(name, X.shape, f"nnz/row={X.nnz / X.shape[0]:.1f}")
    vocab = vec.get_feature_names_out()
    print("vocabulary:", len(vocab))
    idf = dict(zip(vocab, vec.idf_))
    print("sample terms:", [vocab[i] for i in range(0, len(vocab), len(vocab) // 12)])
    print("lowest idf (most common):", sorted(idf, key=idf.get)[:8])
    print("bigram share: %.0f%%" % (100 * np.mean([" " in t for t in vocab])))
