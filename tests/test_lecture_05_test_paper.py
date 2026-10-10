"""Check the pilot's encoder-count exercise against independently built modules."""

import json
import re
from pathlib import Path

import pytest
import torch


DECK = Path(__file__).resolve().parents[1] / "slides/lecture-05-test"
NOTEBOOK = json.loads((DECK / "practice.ipynb").read_text())


@pytest.fixture(scope="module")
def lesson():
    namespace = {}
    for cell in NOTEBOOK["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
    return namespace


@pytest.mark.parametrize("layers,width,heads,vocabulary", [(12, 768, 12, 30522), (2, 8, 1, 11), (2, 8, 4, 11)])
def test_encoder_count_matches_pytorch_shapes(lesson, layers, width, heads, vocabulary):
    # Meta tensors retain the library's real parameter shapes without allocating weights.
    embedding = torch.nn.Embedding(vocabulary, width, device="meta")
    layer = torch.nn.TransformerEncoderLayer(width, heads, dim_feedforward=4 * width, device="meta")
    expected = sum(p.numel() for p in embedding.parameters()) + layers * sum(p.numel() for p in layer.parameters())
    assert lesson["encoder_parameter_counts"](layers, width, heads, vocabulary)["total"] == expected


def test_exercise_order_and_clean_notebook():
    ids = [match.group(1) for cell in NOTEBOOK["cells"] if cell["cell_type"] == "markdown"
           if (match := re.match(r"## Practice (E\d\d)", "".join(cell["source"])))]
    assert ids == ["E01", "E02", "E03", "E04", "E05", "E06"]
    slides = (DECK / "slides.md").read_text()
    assert re.findall(r"Practice (E\d\d) ·", slides) == ids
    for cell in NOTEBOOK["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_training_flop_comparators_are_distinct(lesson):
    assert lesson["single_cost_ratio"] == pytest.approx(150 / 23)
    assert lesson["ensemble_cost_ratio"] == pytest.approx(1200 / 23)
