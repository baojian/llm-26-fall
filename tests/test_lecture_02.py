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


def test_laplace_example_matches_the_textbook(lesson):
    """Jurafsky and Martin, Chapter 3: P_Lap(want | i) and the reconstituted counts."""
    assert lesson["laplace_probability"]("i", "want") == pytest.approx(828 / 3979)
    assert lesson["reconstituted_count"]("i", "want") == pytest.approx(527, abs=0.5)
    assert lesson["reconstituted_count"]("want", "to") == pytest.approx(238, abs=0.5)
    assert lesson["reconstituted_count"]("to", "spend") == pytest.approx(133, abs=0.5)
    assert lesson["reconstituted_count"]("chinese", "want") == pytest.approx(0.098, abs=0.001)
    for previous in WORDS:
        row_total = sum(lesson["laplace_probability"](previous, word) for word in WORDS)
        assert 0 < row_total < 1
    assert sum(lesson["BIGRAM_COUNTS"]["i"]) == 843


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
    assert SLIDES.count('<!-- .slide:') == 38
    assert 'id="smoothing"' in SLIDES and "Katz-backoff" not in SLIDES
    notebook_text = "".join("".join(cell["source"]) for cell in NOTEBOOK["cells"])
    for label in re.findall(r"(?:Exercise|Notebook|practices?) ([EP]\d\d)", SLIDES):
        assert label in notebook_text, label
