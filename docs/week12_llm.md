# Week 12 — LLM story point estimation (board feedback, part 2)

Board feedback: "explore other models and LLM models". This step asks a local open LLM
to estimate story points and compares it with everything built so far.

## Design

| | |
|---|---|
| Model | an open model run locally with Ollama (default `qwen2.5:7b`): free, offline, reproducible |
| `zero` | the issue text only |
| `fewshot` | the issue text plus the 5 most similar **training** issues of the same project, with their real story points. Similarity is Sentence-BERT cosine, the same retrieval as the week 10 kNN |
| Sample | the same N random issues per project (seed 42), because LLM calls are slow on a CPU |
| Compared on the same issues | mean, median, week 9 TF-IDF + RF, week 10 SBERT + kNN |
| Settings | temperature 0, fixed seed, JSON-only answer, issue text cut to 700 characters |

Nothing is tuned (the prompts, k = 5 and the sample size are fixed in the script before any
result is seen), so the test split is used directly. A reply that cannot be read as a number is
replaced by the project median and counted.

## How to run

1. Install Ollama from https://ollama.com/download and start it.
2. In a terminal: `ollama pull qwen2.5:7b` (about 4.7 GB).
3. From `D:\MIT\Project\pm-insight` with the venv active:

```bat
python src\llm_estimate.py --split val --n 3
python src\llm_estimate.py --split test --n 20
```

The first command is a pipeline check (48 issues). The second is the real run: 320 issues × 2
settings = 640 calls. Each answer is saved the moment it arrives
(`results\week12_llm_<model>_<mode>_<split>_n<N>.jsonl`), so if the run is interrupted, the same
command continues where it stopped.

If one call takes more than about 30 seconds on your laptop, use a smaller model:
`ollama pull llama3.2:3b`, then add `--model llama3.2:3b`.

## Limitations to state in the report

- **Data contamination.** These Jira issues are public, so the LLM may have seen them, with
  their story points, during its own training. Its scores may be optimistic.
- **Sample, not the full split.** 20 issues per project; results on small projects are noisy.
- **One model, one prompt.** No prompt tuning was done.
