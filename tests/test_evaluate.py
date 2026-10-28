"""Unit tests for the evaluation metrics in src/evaluate.py.

These are the functions every reported number depends on, so they are worth
pinning down. The important one is test_random_guess_mae_matches_simulation:
random_guess_mae computes the exact expected MAE of guessing a random training
label by comparing every test label against every training label, instead of
sampling. This test confirms that shortcut agrees with actually simulating it.

No test framework needed. Run from the repository root:

    python tests/test_evaluate.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate import (  # noqa: E402
    mae,
    mdae,
    random_guess_mae,
    score_table,
    standardized_accuracy,
)


def test_mae_and_mdae():
    y = np.array([3.0, 5.0, 2.0, 8.0])
    p = np.array([3.0, 4.0, 4.0, 4.0])
    # absolute errors are 0, 1, 2, 4
    assert mae(y, p) == 1.75
    assert mdae(y, p) == 1.5          # median of 0, 1, 2, 4


def test_perfect_prediction_scores_zero_error():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0
    assert mdae(y, y) == 0.0
    assert standardized_accuracy(0.0, 5.0) == 100.0


def test_standardized_accuracy_formula():
    # a model half as wrong as random guessing scores 50%
    assert standardized_accuracy(2.0, 4.0) == 50.0
    # a model exactly as wrong as random guessing scores 0%
    assert standardized_accuracy(4.0, 4.0) == 0.0
    # a model worse than random guessing scores negative
    assert standardized_accuracy(8.0, 4.0) == -100.0


def test_random_guess_mae_matches_simulation():
    """The closed-form expectation must agree with sampling random labels."""
    rng = np.random.default_rng(0)
    train_y = rng.integers(1, 20, size=300).astype(float)
    y = rng.integers(1, 20, size=120).astype(float)

    exact = random_guess_mae(y, train_y)
    simulated = np.mean([
        mae(y, rng.choice(train_y, size=len(y))) for _ in range(4000)
    ])
    assert abs(exact - simulated) < 0.05, (exact, simulated)


def _toy_frames():
    """Two projects of different sizes, so overall and macro must differ."""
    train = pd.DataFrame({
        "project": ["big"] * 8 + ["small"] * 4,
        "storypoint": [1.0, 2, 3, 2, 1, 3, 2, 2, 10, 12, 8, 10],
    })
    test = pd.DataFrame({
        "project": ["big"] * 6 + ["small"] * 2,
        "storypoint": [2.0, 2, 3, 1, 2, 2, 10, 12],
    })
    return train, test


def test_score_table_has_a_row_per_project_plus_summaries():
    train, test = _toy_frames()
    out = score_table(test, test.storypoint.to_numpy(), train, "perfect")
    assert list(out.project) == ["big", "small", "overall", "macro"]
    assert out.model.unique().tolist() == ["perfect"]
    # a perfect prediction is zero error everywhere
    assert (out.MAE == 0).all()
    assert (out.SA == 100).all()


def test_overall_is_issue_weighted_and_macro_is_not():
    train, test = _toy_frames()
    # predict 2 everywhere: exact for most of 'big', far off for 'small'
    pred = np.full(len(test), 2.0)
    out = score_table(test, pred, train, "constant 2").set_index("project")

    big, small = out.loc["big", "MAE"], out.loc["small", "MAE"]
    n_big, n_small = out.loc["big", "n"], out.loc["small", "n"]

    expected_overall = (big * n_big + small * n_small) / (n_big + n_small)
    expected_macro = (big + small) / 2

    assert abs(out.loc["overall", "MAE"] - expected_overall) < 1e-9
    assert abs(out.loc["macro", "MAE"] - expected_macro) < 1e-9
    # the small project is much worse, so weighting by issue count flatters the model
    assert out.loc["overall", "MAE"] < out.loc["macro", "MAE"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
