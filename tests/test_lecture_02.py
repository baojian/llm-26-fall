"""Check the lecture-02 notebook against the numbers shown on the slides."""

import builtins
import json
import math
import re
from fractions import Fraction
from pathlib import Path

import nbformat
import pytest


LECTURE = Path(__file__).resolve().parents[1] / "slides/lecture-02"
NOTEBOOK = json.loads((LECTURE / "lecture-02-exercise.ipynb").read_text())
SLIDES = (LECTURE / "slides.md").read_text()
WORDS = ["i", "want", "to", "eat", "chinese", "food", "lunch", "spend"]


def execute_cell(cell, namespace):
    exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)


@pytest.fixture(scope="module")
def lesson(request):
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)

    def no_network(*args, **kwargs):
        pytest.fail("The offline notebook attempted a network request")

    original_import = builtins.__import__

    def no_model_imports(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "datasets", "kenlm", "numpy"}:
            pytest.fail(f"The offline notebook imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("urllib.request.OpenerDirector.open", no_network)
    monkeypatch.setattr(builtins, "__import__", no_model_imports)
    nbformat.validate(nbformat.from_dict(NOTEBOOK))
    namespace = {}
    for cell in NOTEBOOK["cells"]:
        if cell["cell_type"] == "code":
            execute_cell(cell, namespace)
    return namespace


def slide_table(slide_id):
    """Return the 8x8 numeric table of one slide as floats keyed by row word."""
    start = SLIDES.index(f'id="{slide_id}"')
    end = SLIDES.index("\n---\n", start)
    section = SLIDES[start:end]
    rows = {}
    for row in re.findall(r"<tr><th>(\w+)</th>(.*?)</tr>", section):
        values = re.findall(r"<td[^>]*>(?:<[^>]+>)*([0-9.]+)(?:</[^>]+>)*</td>", row[1])
        assert len(values) == 8, (slide_id, row[0])
        rows[row[0]] = [float(value) for value in values]
    assert list(rows) == WORDS
    return rows


def test_toy_bigram_estimates_match_the_slide(lesson):
    mle = lesson["bigram_mle"]
    assert mle("BOS", "I") == Fraction(2, 3)
    assert mle("BOS", "Sam") == Fraction(1, 3)
    assert mle("I", "am") == Fraction(2, 3)
    assert mle("I", "do") == Fraction(1, 3)
    assert mle("Sam", "EOS") == Fraction(1, 2)
    assert mle("am", "Sam") == Fraction(1, 2)
    assert sum(mle("I", word) for word in ["am", "do"]) == 1


def test_unseen_bigram_gives_minus_infinity(lesson):
    assert lesson["sentence_log_prob"]("I am Sam") == pytest.approx(math.log(2 / 3 * 2 / 3 * 1 / 2 * 1 / 2))
    assert lesson["sentence_log_prob"]("Sam do not like ham") == float("-inf")


def test_uniform_digits_have_perplexity_ten(lesson):
    assert lesson["digit_perplexity"] == pytest.approx(10.0)
    assert lesson["perplexity"](math.log(1.0), 7) == pytest.approx(1.0)


def test_laplace_tables_match_the_slides(lesson):
    probabilities = slide_table("exercise-04")
    counts = slide_table("reconstituted-counts")
    for previous in WORDS:
        for index, word in enumerate(WORDS):
            assert lesson["laplace_table"][previous][word] == pytest.approx(probabilities[previous][index], rel=0.05)
            assert lesson["reconstituted_table"][previous][word] == pytest.approx(counts[previous][index], rel=0.05)
    assert lesson["laplace_probability"]("i", "want") == pytest.approx(828 / 3979)


def test_raw_counts_on_the_slide_match_the_notebook(lesson):
    raw = slide_table("laplace-counts")
    plus_one = slide_table("laplace-plus-one")
    for previous in WORDS:
        assert raw[previous] == lesson["BIGRAM_COUNTS"][previous]
        assert plus_one[previous] == [value + 1 for value in raw[previous]]


def test_good_turing_estimates_form_a_distribution(lesson):
    assert lesson["good_turing_total"] == 1
    assert lesson["good_turing"]["Bob"] == Fraction(1, 10)
    assert lesson["good_turing"]["do"] == Fraction(2, 15)
    assert lesson["good_turing"]["Sam"] == Fraction(3, 20)
    assert lesson["good_turing"]["I"] == 0


def test_sampled_sentences_use_training_bigrams(lesson):
    import random

    bigrams = lesson["BIGRAMS"]
    for seed in range(20):
        sentence = lesson["sample_sentence"](random.Random(seed))
        tokens = ["BOS"] + sentence.split() + ["EOS"]
        assert all(pair in bigrams for pair in zip(tokens, tokens[1:-1]))


def test_manifest_and_assets_are_consistent():
    manifest = json.loads((LECTURE / "lecture.json").read_text())
    assert manifest["notebook"] == "lecture-02-exercise.ipynb"
    assert (LECTURE / manifest["notebook"]).exists()
    for asset in re.findall(r'(?:src|data-plotly)="(assets/[^"]+)"', SLIDES):
        assert (LECTURE / asset).exists(), asset
    assert (LECTURE / "assets/predicting-next-word.mp4").stat().st_size < 10_000_000
    assert SLIDES.rstrip().endswith("index.html#/33.")
    assert 'id="references"' in SLIDES
