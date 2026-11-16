"""Week 10: dense text representations, as an alternative to TF-IDF.

Board feedback after the midterm: "use better embeddings (word2vec, BERT, or a
domain-specific technique)". Each function below turns the issue text of every
split into one fixed-length vector per issue. Everything that is *learned*
(Word2Vec weights, the SVD) is fitted on the TRAINING split only, exactly like
the TF-IDF vectoriser in features.py.

  lsa       TF-IDF -> TruncatedSVD (300 dims). No new library; a classic
            "dense TF-IDF" reference point.
  w2v_own   Word2Vec trained here on our own training issues (gensim).
  w2v_so    Word2Vec pre-trained on 15 GB of Stack Overflow posts
            (Efstathiou et al., MSR 2018) - the domain-specific option.
            Download SO_vectors_200.bin (1.5 GB) from
            https://zenodo.org/records/1199620 into models/
  sbert     Sentence-BERT (all-MiniLM-L6-v2), used frozen, no fine-tuning.
  sbert_mp  Sentence-BERT (all-mpnet-base-v2), larger and slower.

Vectors are cached to data/processed/emb_<name>_<split>.npy, so the slow
ones are computed once.

Run from the repository root, one representation at a time:
    python src/embeddings.py lsa
    python src/embeddings.py w2v_own
    python src/embeddings.py w2v_so
    python src/embeddings.py sbert
"""
import html
import os
import re
import sys
import time

import joblib
import numpy as np

from features import load_splits

PROCESSED = "data/processed"
MODELS = "models"
SPLITS = ("train", "val", "test")
SO_PATH = f"{MODELS}/SO_vectors_200.bin"
SEED = 42


# ---------------------------------------------------------------- text helpers
def plain_text(s):
    """The cleaned text still holds HTML from the Jira export ("<div><p>...").
    TF-IDF did not care, but embeddings should see words, not tags."""
    s = html.unescape(str(s))
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\bhtml\b", " ", s)          # the export wraps text in "html ... html"
    return re.sub(r"\s+", " ", s).strip()


TOKEN = re.compile(r"[a-z][a-z0-9_.#+-]*[a-z0-9#+]|[a-z]")


def tokenize(s):
    """Lower-case word tokens. Keeps software words such as c#, c++, node.js."""
    return TOKEN.findall(plain_text(s).lower())


def mean_vectors(token_lists, kv, dim):
    """Average the word vectors of each issue. Issues with no known word get a
    zero vector; the returned `coverage` says how many tokens were found."""
    out = np.zeros((len(token_lists), dim), dtype=np.float32)
    found = total = 0
    for i, toks in enumerate(token_lists):
        vecs = [kv[t] for t in toks if t in kv]
        found += len(vecs)
        total += len(toks)
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out, found / max(total, 1)


# ------------------------------------------------------------ representations
def build_lsa(splits, dims=300):
    from sklearn.decomposition import TruncatedSVD
    vec = joblib.load(f"{MODELS}/tfidf.joblib")
    svd = TruncatedSVD(n_components=dims, random_state=SEED)
    svd.fit(vec.transform(splits["train"].text))
    print(f"LSA explained variance: {svd.explained_variance_ratio_.sum():.1%}")
    return {n: svd.transform(vec.transform(d.text)).astype(np.float32) for n, d in splits.items()}


def build_w2v_own(splits, dim=200):
    from gensim.models import Word2Vec
    toks = {n: [tokenize(t) for t in d.text] for n, d in splits.items()}
    # skip-gram, trained on TRAINING issues only; workers=1 makes it reproducible
    w2v = Word2Vec(sentences=toks["train"], vector_size=dim, window=5, min_count=2,
                   sg=1, epochs=15, seed=SEED, workers=1)
    w2v.save(f"{MODELS}/w2v_own.model")
    print(f"vocabulary: {len(w2v.wv):,} words")
    for probe in ("bug", "crash", "deploy", "ui"):
        if probe in w2v.wv:
            print(f"  nearest to '{probe}':", [w for w, _ in w2v.wv.most_similar(probe, topn=5)])
    out = {}
    for n in SPLITS:
        out[n], cov = mean_vectors(toks[n], w2v.wv, dim)
        print(f"  {n}: token coverage {cov:.1%}")
    return out


def read_word2vec_bin(path, keep):
    """Read a file in the original word2vec binary format, without gensim.

    Layout: a text header "<vocab size> <dims>\\n", then for every word the
    word itself, one space, and <dims> float32 numbers (4 bytes each).
    Only the words in `keep` (our own vocabulary) are held in memory.
    """
    vectors = {}
    with open(path, "rb") as f:
        n_words, dim = map(int, f.readline().split())
        n_bytes = 4 * dim
        for _ in range(n_words):
            word = bytearray()
            while True:
                ch = f.read(1)
                if ch == b" ":
                    break
                if ch == b"":
                    raise EOFError(f"file ended early in {path}")
                if ch != b"\n":          # newline left over from the previous vector
                    word += ch
            raw = f.read(n_bytes)
            w = word.decode("utf-8", errors="ignore")
            if w in keep:
                vectors[w] = np.frombuffer(raw, dtype=np.float32).copy()
    return vectors, dim, n_words


def build_w2v_so(splits):
    if not os.path.exists(SO_PATH):
        sys.exit(f"Missing {SO_PATH}. Download SO_vectors_200.bin from "
                 "https://zenodo.org/records/1199620 and put it in models/")
    toks = {n: [tokenize(t) for t in d.text] for n, d in splits.items()}
    our_vocab = {w for lists in toks.values() for ts in lists for w in ts}
    t0 = time.time()
    kv, dim, n_words = read_word2vec_bin(SO_PATH, our_vocab)
    print(f"Stack Overflow model: {n_words:,} words, {dim} dims "
          f"(read in {time.time() - t0:.0f}s); {len(kv):,} of our {len(our_vocab):,} words found")
    out = {}
    for n in SPLITS:
        out[n], cov = mean_vectors(toks[n], kv, dim)
        print(f"  {n}: token coverage {cov:.1%}")
    return out


def build_sbert(splits, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")
    out = {}
    for n, d in splits.items():
        texts = [plain_text(t) for t in d.text]
        out[n] = model.encode(texts, batch_size=64, show_progress_bar=True,
                              normalize_embeddings=True).astype(np.float32)
    return out


BUILDERS = {
    "lsa": build_lsa,
    "w2v_own": build_w2v_own,
    "w2v_so": build_w2v_so,
    "sbert": build_sbert,
    "sbert_mp": lambda s: build_sbert(s, "sentence-transformers/all-mpnet-base-v2"),
}


def cache_path(name, split):
    return f"{PROCESSED}/emb_{name}_{split}.npy"


def load_embedding(name):
    """{split: array} for a representation built earlier, or None."""
    if not all(os.path.exists(cache_path(name, s)) for s in SPLITS):
        return None
    return {s: np.load(cache_path(name, s)) for s in SPLITS}


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in BUILDERS:
        sys.exit(f"usage: python src/embeddings.py {{{'|'.join(BUILDERS)}}}")
    name = sys.argv[1]
    splits = load_splits()
    t0 = time.time()
    vecs = BUILDERS[name](splits)
    for s in SPLITS:
        assert len(vecs[s]) == len(splits[s])
        np.save(cache_path(name, s), vecs[s])
        print(f"saved {cache_path(name, s)}  shape={vecs[s].shape}")
    print(f"done in {time.time() - t0:.0f}s")
