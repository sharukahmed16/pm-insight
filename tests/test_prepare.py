"""Unit tests for the text cleaning and splitting in src/prepare.py.

clean_text runs over every issue before anything else sees it, so a mistake here
corrupts the input to the whole pipeline silently. These tests pin down what it
actually does, including one known limitation (see
test_known_limitation_h_pattern_matches_inside_words).

No test framework needed. Run from the repository root:

    python tests/test_prepare.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prepare import clean_text, split_project  # noqa: E402


def test_lowercases_and_collapses_whitespace():
    assert clean_text("Fix NullPointerException") == "fix nullpointerexception"
    assert clean_text("Multiple    spaces\tand\nnewlines") == "multiple spaces and newlines"
    assert clean_text("   padded   ") == "padded"


def test_missing_values_become_empty_string():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text("") == ""


def test_jira_code_blocks_are_removed():
    assert clean_text("See {code:java}int x = 1;{code} for detail") == "see for detail"
    assert clean_text("Log {noformat}ERROR 500{noformat} here") == "log here"


def test_urls_become_a_single_token():
    # URLs carry no size signal, so they collapse to one placeholder term
    assert clean_text("Docs at https://example.com/a/b now") == "docs at url now"
    assert clean_text("http://x.io/y") == "url"


def test_user_mentions_are_removed():
    assert clean_text("Ping [~jsmith] about this") == "ping about this"


def test_wiki_markup_characters_are_stripped():
    assert clean_text("A [link] and |pipe| and __under__") == "a link and pipe and under"
    assert clean_text("h2. Heading then text") == "heading then text"


def test_known_limitation_h_pattern_matches_inside_words():
    """The h1.-h6. heading rule also fires in the middle of a word.

    Measured over the raw dataset: 11 of 23,313 issues (0.05%) are affected,
    and no issue in the dataset contains a real h1.-h6. heading, so the rule
    currently removes nothing useful. Real examples that get mangled:
    'OAuth2.0', 'libssh2.so.1', 'oauth1.AbstractOAuth1ApiBinding'.

    Left unchanged deliberately: fixing it would alter the cleaned text and
    require re-running the whole pipeline including the one-shot test
    evaluation, for a 0.05% change. Recorded as a limitation instead.

    This test asserts the CURRENT behaviour. If the regex is ever tightened to
    require a line start, this test should be updated to the fixed behaviour.
    """
    assert clean_text("graph1. shows the curve") == "grap shows the curve"
    assert clean_text("supports OAuth2.0 tokens") == "supports oaut tokens"


def test_split_is_sixty_twenty_twenty_in_order():
    d = pd.DataFrame({"x": range(10)})
    train, val, test = split_project(d)
    assert (len(train), len(val), len(test)) == (6, 2, 2)
    # the split preserves order: train is the earliest rows, test the latest
    assert list(train.x) == [0, 1, 2, 3, 4, 5]
    assert list(val.x) == [6, 7]
    assert list(test.x) == [8, 9]


def test_split_handles_sizes_that_do_not_divide_evenly():
    for n, expected in [(7, (4, 1, 2)), (16, (9, 3, 4)), (3, (1, 1, 1))]:
        d = pd.DataFrame({"x": range(n)})
        got = tuple(len(p) for p in split_project(d))
        assert got == expected, (n, got, expected)


def test_split_never_loses_or_duplicates_rows():
    for n in (1, 3, 7, 10, 16, 100):
        d = pd.DataFrame({"x": range(n)})
        parts = split_project(d)
        assert sum(len(p) for p in parts) == n
        combined = pd.concat(parts).x.tolist()
        assert combined == list(range(n))


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
