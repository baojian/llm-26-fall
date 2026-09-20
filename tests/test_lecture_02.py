"""Check the lecture-02 notebook against the numbers shown on the slides."""

import builtins
import json
import math
import re
import random
from collections import Counter
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


def test_interpolation_tunes_lambda_on_held_out_text(lesson):
    assert lesson["best_lambda"] == 0.6
    assert math.isfinite(lesson["interpolated_perplexity"])
    assert 5.5 < lesson["interpolated_perplexity"] < 6.5
    assert 0.8 < lesson["bits_per_byte"] < 0.95
    # pure bigram and pure unigram both lose to the mixture on the held-out sentence
    assert lesson["scores"][0.6] > lesson["scores"][1.0] and lesson["scores"][0.6] > lesson["scores"][0.0]


def test_self_training_loop_reports_finite_losses_for_all_rounds(lesson):
    rounds = lesson["loop_rounds"]
    assert [r["round"] for r in rounds] == list(range(6))
    assert all(math.isfinite(r["loss"]) and r["loss"] >= 0 for r in rounds)


def test_notebook_mixtures_normalize_on_fixed_vocabulary_and_unknown_histories(lesson):
    prob, _, _ = lesson["loop_model"](lesson["LOOP_CORPUS"])
    for previous in (lesson["BOS"], "cat", "unseen", lesson["EOS"]):
        assert sum(prob(previous, w) for w in lesson["LOOP_VOCAB"]) == pytest.approx(1)
        assert sum(lesson["interpolated_prob"](previous, w, 0.6) for w in lesson["P03_VOCAB"]) == pytest.approx(1)


def test_self_training_sampler_uses_the_scored_distribution(lesson):
    prob, sample, _ = lesson["loop_model"](lesson["LOOP_CORPUS"])
    rng = random.Random(2026)
    draws = Counter(sample(rng, max_length=2) or lesson["EOS"] for _ in range(12000))
    for word in lesson["LOOP_VOCAB"]:
        assert draws[word] / 12000 == pytest.approx(prob(lesson["BOS"], word), abs=0.015)


def test_notebook_practices_follow_the_stated_order():
    ids = [match.group(1) for cell in NOTEBOOK["cells"]
           if (match := re.match(r"## (P\d\d) ", "".join(cell["source"]))) ]
    assert ids == ["P01", "P02", "P03", "P04"]


def test_demonstration_metrics_and_charts_use_consistent_units():
    results = json.loads((LECTURE / "assets/lecture02-results.json").read_text())
    for source in results["held_out"].values():
        assert source["test_tokens"] == source["test_content_tokens"] + source["test_docs"]
        for row in source["orders"].values():
            bpb = source["test_tokens"] / source["test_bytes"] * row["loss_nats_per_token"] / math.log(2)
            assert bpb == pytest.approx(row["bits_per_byte"], abs=0.0006)
    traces = json.loads((LECTURE / "assets/lm-filter.json").read_text())["data"]
    assert traces[0]["x"] == traces[1]["x"]
    assert all(sum(trace["y"]) == pytest.approx(100) for trace in traces)
    loop = json.loads((LECTURE / "assets/self-training-loop.json").read_text())["data"][0]
    assert loop["customdata"] == [r["corpus_tokens"] for r in results["self_training"]["rounds"]]


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
