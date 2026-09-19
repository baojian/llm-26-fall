"""Check the lecture-03 notebook against the numbers shown on the slides."""

import json
import math
import re
from pathlib import Path

import nbformat
import pytest


LECTURE = Path(__file__).resolve().parents[1] / "slides/lecture-03"
NOTEBOOK = json.loads((LECTURE / "lecture-03-exercise.ipynb").read_text())
SLIDES = (LECTURE / "slides.md").read_text()


def execute_cell(cell, namespace):
    exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)


@pytest.fixture(scope="module")
def lesson(request):
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)

    def no_network(*args, **kwargs):
        pytest.fail("The offline notebook attempted a network request")

    monkeypatch.setattr("urllib.request.OpenerDirector.open", no_network)
    monkeypatch.setattr("socket.create_connection", no_network)
    nbformat.validate(nbformat.from_dict(NOTEBOOK))
    namespace = {}
    for cell in NOTEBOOK["cells"]:
        if cell["cell_type"] == "code":
            execute_cell(cell, namespace)
    return namespace


def test_notebook_has_no_stored_outputs():
    for cell in NOTEBOOK["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_ppmi_matches_the_textbook_table(lesson):
    """Jurafsky and Martin's word-context counts: totals and PPMI(information, data) = 0.0944."""
    assert lesson["TOTAL"] == 11716
    assert lesson["ROW_TOTAL"]["information"] == 7703
    assert lesson["COLUMN_TOTAL"]["data"] == 5673
    assert lesson["ppmi"]("information", "data") == pytest.approx(0.0944, abs=5e-5)
    assert lesson["ppmi"]("cherry", "pie") == pytest.approx(4.38, abs=5e-3)
    assert lesson["pmi"]("strawberry", "computer") == float("-inf")
    assert lesson["pmi"]("digital", "pie") < 0 and lesson["ppmi"]("digital", "pie") == 0


def test_skipgram_step_by_hand_matches_autograd(lesson):
    assert lesson["s_pos"] == pytest.approx(0.7311, abs=5e-5)
    assert lesson["s_neg"] == pytest.approx(0.6225, abs=5e-5)
    assert lesson["e02_loss"] == pytest.approx(1.2873, abs=5e-5)
    assert lesson["e02_grads"]["c_pos"] == pytest.approx([-0.2689, -0.1345], abs=5e-5)
    assert lesson["e02_grads"]["c_neg"] == pytest.approx([0.6225, 0.3112], abs=5e-5)
    assert lesson["e02_grads"]["w"] == pytest.approx([0.4880, -0.8914], abs=5e-5)
    for name, grad in lesson["e02_grads"].items():
        assert lesson["e02_autograd"][name] == pytest.approx(grad, abs=1e-6)
    assert lesson["after"] < lesson["e02_loss"]


def test_lookup_shapes_and_uniform_loss(lesson):
    assert lesson["e03_shapes"] == {"weight": (10, 4), "x": (2, 3, 4), "one_hot": (2, 3, 10)}
    assert lesson["e03_lookup_equals_onehot"] is True
    assert lesson["e03_logits_shape"] == (2, 3, 10)
    assert lesson["e03_uniform_loss"] == pytest.approx(math.log(10), abs=1e-6)
    assert lesson["e03_rows_with_gradient"] == [0, 1, 5, 7]


def test_parameter_counts_with_and_without_weight_sharing(lesson):
    assert (lesson["e04_untied"], lesson["e04_tied"]) == (80, 40)
    assert lesson["e04_tables"] == {"GPT-2 small": 38_597_376, "Qwen3-0.6B": 155_582_464}
    assert 38_597_376 / 124_439_808 == pytest.approx(0.31, abs=0.005)


def test_toy_skipgram_learns_the_template_groups(lesson):
    history = lesson["p01_history"]
    assert history[-1] < history[0] < 6 * math.log(2)
    groups = [
        {"king", "queen", "prince", "princess", "man", "woman", "boy", "girl"},
        {"cat", "dog", "horse", "bird"},
        {"rice", "bread", "fish", "apple"},
        {"beijing", "paris", "london", "tokyo"},
    ]
    for group in groups:
        for word in group:
            neighbours = {neighbour for neighbour, _ in lesson["nearest"](word, k=2)}
            assert neighbours & group, f"{word}: {neighbours}"
    assert "queen" in lesson["analogy"]("man", "woman", "king")


def test_training_loop_memorizes_one_batch(lesson):
    losses = lesson["p03_losses"]
    assert losses[0] == pytest.approx(math.log(10), abs=0.01)
    assert losses[-1] < 0.5


def test_notebook_exercises_follow_the_slide_order():
    ids = [match.group(1) for cell in NOTEBOOK["cells"]
           if (match := re.match(r"## ([EP]\d\d) ", "".join(cell["source"])))]
    assert ids == ["E01", "E02", "E03", "E04", "P01", "P02", "P03"]
    slide_ids = re.findall(r"Exercise (E\d\d) ·", SLIDES)
    assert slide_ids == ["E01", "E02", "E03", "E04"]


@pytest.mark.parametrize("number", [
    "0.0944", "0.7311", "0.6225", "1.2873", "38,597,376", "124,439,808", "155,582,464",
])
def test_slide_numbers_come_from_the_notebook(number):
    assert number in SLIDES
