# Week 10 — Embeddings and more models (board feedback)

Board feedback after the midterm:
1. Use better embeddings (word2vec, BERT, or a domain-specific technique).
2. Explore other models and LLMs.

This week covers (1) and the classical half of (2). The LLM part comes in Week 12.

## What was added

| File | Purpose |
|---|---|
| `src/embeddings.py` | Builds one vector per issue for each representation and caches it to `data/processed/emb_<name>_<split>.npy` |
| `src/compare_representations.py` | Crosses every representation with 5 models and scores them on the **validation** split |
| `src/test_evaluation_week10.py` | Re-fits every validation-chosen configuration on train and scores it; `--final` scores the **test** split once |

The protocol has not changed: the vocabulary, Word2Vec weights, SVD and scalers are all fitted on **train** only. Model settings are chosen on **validation**. **Test** stays untouched until the final evaluation.

## Representations

| Name | What it is | Library |
|---|---|---|
| `tfidf` | the original (Week 4) | scikit-learn |
| `lsa` | TF-IDF compressed to 300 dense dimensions (TruncatedSVD) | scikit-learn |
| `w2v_own` | Word2Vec skip-gram, 200 dims, trained on our own training issues; issue = mean of its word vectors | gensim |
| `w2v_so` | Word2Vec pre-trained on Stack Overflow (Efstathiou et al., MSR 2018), 200 dims — **domain-specific** | none (own reader for the word2vec .bin format) + 1.5 GB file |
| `sbert` | Sentence-BERT all-MiniLM-L6-v2, frozen (no fine-tuning), 384 dims | sentence-transformers |
| `sbert_mp` | Sentence-BERT all-mpnet-base-v2, 768 dims, slower | sentence-transformers |

## Models

`ridge`, `knn` (analogy-based: the median of the k most similar past issues of the same project), `svr`, `rf`, `xgb`.

## How to run (Windows, from `D:\MIT\Project\pm-insight`, venv active)

```bat
REM install separately: if one package fails, pip installs neither
pip install sentence-transformers

python src\embeddings.py lsa
python src\embeddings.py sbert

REM domain-specific: download SO_vectors_200.bin from
REM https://zenodo.org/records/1199620 into the models\ folder (no gensim needed)
python src\embeddings.py w2v_so

python src\compare_representations.py
```

### `w2v_own` needs gensim, which has no Python 3.14 wheel

`pip install gensim` tries to compile from source and fails with
"Microsoft Visual C++ 14.0 or greater is required". Two ways around it:

- **A. A second venv on Python 3.13, used only to build this one embedding** (recommended, smaller download):
  ```bat
  py install 3.13
  py -3.13 -m venv .venv-w2v
  .venv-w2v\Scripts\activate
  pip install gensim pandas scikit-learn joblib
  python src\embeddings.py w2v_own
  deactivate
  .venv\Scripts\activate
  ```
  (If `py install` is not recognised, install Python 3.13 from python.org.)
  The vectors are saved as `.npy` files, so `compare_representations.py` keeps running in the normal venv.
- **B.** Install "Microsoft C++ Build Tools" (several GB), then `pip install gensim` works in the normal venv.

Approximate times on a laptop CPU: lsa < 1 min, w2v_own 2–5 min, sbert 10–20 min, w2v_so 1–3 min, compare (everything) 45–90 min.
`compare_representations.py` skips any representation that has not been built.

Output:
- `results/week10_representation_validation.csv`: per-project and overall MAE / MdAE / SA for the best setting of every representation × model
- `results/week10_representation_grid.csv`: every setting tried

`models/` and `data/processed/` are already in `.gitignore`, so the 1.5 GB file and the cached vectors will not be committed.

## Result so far (validation, macro MAE)

| | kNN | RF | Ridge | SVR | XGB |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 2.897 | 2.964 | 3.004 | 2.887 | 3.019 |
| LSA | 2.916 | 3.161 | 3.179 | 3.193 | 3.095 |
| SBERT | **2.853** | 3.103 | 2.996 | 3.075 | 2.976 |

Baselines: median 2.965, mean 3.592. SBERT + kNN (k=80) beats both baselines on 10 of 16 projects.

## Final test evaluation (run once, at the end)

1. Build every embedding you intend to report, then run `python src\compare_representations.py`.
2. Check run: `python src\test_evaluation_week10.py`. It scores **validation** again from scratch;
   its macro MAE must match the table printed by step 1. (Tiny differences in kNN can come
   from ties in similarity.)
3. Only then: `python src\test_evaluation_week10.py --final`, **once**. Do not change any
   setting after seeing the test numbers.

The winner is not picked by hand: it is the configuration with the lowest validation macro MAE,
read from `results\week10_representation_validation.csv`. The week 9 model is scored alongside it.
