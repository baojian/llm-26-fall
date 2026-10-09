"""Execute the offline lesson and verify its feedforward, recurrent, and alignment claims."""

import json
import math
import re
from pathlib import Path

import nbformat
import pytest


LECTURE = Path(__file__).resolve().parents[1] / "slides/lecture-04"
NOTEBOOK = json.loads((LECTURE / "lecture-04-exercise.ipynb").read_text())


@pytest.fixture(scope="module")
def lesson(request):
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)

    def no_network(*args, **kwargs):
        pytest.fail("The offline lecture attempted a network request")

    monkeypatch.setattr("urllib.request.OpenerDirector.open", no_network)
    monkeypatch.setattr("socket.socket.connect", no_network)
    monkeypatch.setattr("socket.create_connection", no_network)
    monkeypatch.chdir(LECTURE)
    nbformat.validate(nbformat.from_dict(NOTEBOOK))
    namespace = {}
    for cell in NOTEBOOK["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
    return namespace


def test_notebook_is_clean_and_exercise_order_matches_slides():
    for cell in NOTEBOOK["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
    ids = [match.group(1) for cell in NOTEBOOK["cells"] if cell["cell_type"] == "markdown"
           if (match := re.match(r"## ([EP]\d\d) ·", "".join(cell["source"])))]
    assert ids == ["E01", "E02", "E03", "E04", "E05"]
    slides = (LECTURE / "slides.md").read_text()
    assert re.findall(r"Exercise (E\d\d) ·", slides) == ids[:5]


def test_windows_stay_within_sentence_boundaries(lesson):
    assert lesson["train_contexts"].tolist() == [[0, 1], [1, 3], [3, 4], [0, 2], [2, 3], [3, 5]]
    assert lesson["train_targets"].tolist() == [3, 4, 6, 3, 5, 6]
    contexts, targets = lesson["make_windows"]([[1, 2], [3, 4, 5]], 2)
    assert contexts.tolist() == [[3, 4]] and targets.tolist() == [5]
    empty_contexts, empty_targets = lesson["make_windows"]([[1]], 2)
    assert tuple(empty_contexts.shape) == (0, 2)
    assert tuple(empty_targets.shape) == (0,)
    with pytest.raises(ValueError, match="positive"):
        lesson["make_windows"]([[1, 2]], 0)


def test_context_and_target_shapes(lesson):
    assert lesson["e01_shapes"] == {
        "ids": (2, 2), "lookup": (2, 2, 4), "concatenated": (2, 8),
        "hidden": (2, 8), "logits": (2, 7),
    }
    with pytest.raises(ValueError, match="context_size"):
        lesson["model"](lesson["train_contexts"][:, :1])


def test_stable_loss_and_uniform_baseline(lesson):
    assert math.isinf(lesson["naive_normalizer"].item())
    expected = math.log(1 + math.exp(-1) + math.exp(-2))
    assert lesson["stable_loss"].item() == pytest.approx(expected, abs=1e-12)
    assert lesson["builtin_loss"].item() == pytest.approx(expected, abs=1e-12)
    assert lesson["uniform_loss"] == pytest.approx(math.log(7), abs=1e-6)


def test_repaired_loop_learns_earlier_context(lesson):
    torch = lesson["torch"]
    assert lesson["detached_loss"].requires_grad is False
    assert "does not require grad" in lesson["detached_error"]
    assert len(lesson["e02_losses"]) == 201
    assert all(math.isfinite(value) for value in lesson["e02_losses"] + lesson["e02_gradient_norms"])
    assert lesson["e02_losses"][-1] < 0.03
    assert torch.equal(lesson["e02_predictions"], lesson["train_targets"])
    assert not torch.equal(lesson["e02_initial_embedding"], lesson["model"].embedding.weight)
    # The last token is identical, but the learned two-token predictions differ.
    with torch.no_grad():
        predictions = lesson["model"](torch.tensor([[1, 3], [2, 3]])).argmax(-1)
    assert predictions.tolist() == [4, 5]


def test_loss_figure_matches_the_executed_notebook(lesson):
    figure = json.loads((LECTURE / "assets/tiny-lm-loss.json").read_text())
    trace = figure["data"][0]
    assert trace["x"][0] == 0 and trace["x"][-1] == 200
    assert trace["y"] == pytest.approx([lesson["e02_losses"][step] for step in trace["x"]], abs=1e-5)
    assert figure["data"][1]["y"] == pytest.approx([math.log(7)] * 2)


def test_recurrent_memory_and_shared_parameter_derivatives(lesson):
    assert lesson["recurrent_states"].tolist() == [1, 0.5, 1.25]
    assert lesson["input_derivatives"].tolist() == [0.25, 0.5, 1]
    assert lesson["shared_derivative"].item() == 1
    assert lesson["changed_states"].tolist() == [2, 1, 1.5]
    torch = lesson["torch"]
    # A zero recurrent coefficient discards every earlier input.
    values = torch.tensor([5., -3., 2.], dtype=torch.float64)
    assert torch.equal(lesson["linear_recurrence"](values, 0), values)


@pytest.mark.parametrize("length", [1, 4])
def test_tanh_recurrence_matches_library_for_outputs_and_gradients(lesson, length):
    torch = lesson["torch"]
    model = lesson["rnn"]
    inputs = lesson["rnn_inputs"][:, :length].clone().requires_grad_()
    initial = lesson["rnn_initial"].clone().requires_grad_()
    actual, final = lesson["tanh_recurrence"](
        inputs, initial, model.weight_ih_l0, model.weight_hh_l0,
        model.bias_ih_l0 + model.bias_hh_l0)
    expected, expected_final = model(inputs, initial.unsqueeze(0))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(final, expected_final[0], rtol=1e-12, atol=1e-12)
    parameters = (inputs, initial, *model.parameters())
    grad_actual = torch.autograd.grad(actual.square().sum(), parameters)
    grad_expected = torch.autograd.grad(expected.square().sum(), parameters)
    for left, right in zip(grad_actual, grad_expected):
        torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-11)


def test_lstm_cell_arithmetic_and_gate_extremes(lesson):
    assert lesson["next_cell"].item() == 1.25
    assert lesson["next_hidden"].item() == pytest.approx(0.5 * math.tanh(1.25))
    assert lesson["direct_cell_derivative"].item() == 0.75
    update = lesson["supplied_gate_update"]
    old = lesson["previous_cell"]
    assert update(old, 0, 0.5, -0.5, 0.5)[0].item() == -0.25
    assert update(old, 1, 0.5, -0.5, 0.5)[0].item() == 1.75
    stored, hidden = update(old, 0.75, 0.5, -0.5, 0)
    assert stored.item() == 1.25 and hidden.item() == 0


def test_complete_lstm_cell_matches_library_outputs_and_gradients(lesson):
    torch = lesson["torch"]
    cell = lesson["lstm_cell"]
    inputs, hidden, previous = [lesson[name].clone().requires_grad_()
                               for name in ("lstm_input", "lstm_hidden", "lstm_previous")]
    actual_cell, actual_hidden = lesson["explicit_lstm_cell"](inputs, hidden, previous, cell)
    expected_hidden, expected_cell = cell(inputs, (hidden, previous))
    torch.testing.assert_close(actual_cell, expected_cell, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_hidden, expected_hidden, rtol=1e-12, atol=1e-12)
    variables = (inputs, hidden, previous, *cell.parameters())
    actual_grad = torch.autograd.grad(actual_cell.square().sum() + actual_hidden.square().sum(), variables)
    expected_grad = torch.autograd.grad(expected_cell.square().sum() + expected_hidden.square().sum(), variables)
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-11)


def test_decoder_steps_select_different_source_contexts(lesson):
    assert lesson["alignment_weights"].tolist() == pytest.approx([0.25, 0.75])
    assert lesson["alignment_context"].tolist() == pytest.approx([2.5, 0.5])
    assert lesson["next_alignment_weights"].tolist() == pytest.approx([0.75, 0.25])
    assert lesson["next_alignment_context"].tolist() == pytest.approx([1.5, 1.5])


def test_additive_alignment_matches_scalar_reference_and_finite_differences(lesson):
    torch = lesson["torch"]
    state, source, weight_s, weight_h, vector = lesson["additive_inputs"]
    # Independent scalar expansion of the two affine projections and tanh score.
    scores = []
    for annotation in source.tolist():
        score = 0
        for row_s, row_h, coefficient in zip(weight_s.tolist(), weight_h.tolist(), vector.tolist()):
            score += coefficient * math.tanh(
                sum(a * b for a, b in zip(row_s, state.tolist()))
                + sum(a * b for a, b in zip(row_h, annotation)))
        scores.append(score)
    weights = [math.exp(s) / sum(math.exp(x) for x in scores) for s in scores]
    context = [sum(w * row[d] for w, row in zip(weights, source.tolist())) for d in range(2)]
    assert lesson["additive_scores"].tolist() == pytest.approx(scores)
    assert lesson["additive_weights"].tolist() == pytest.approx(weights)
    assert lesson["additive_context"].tolist() == pytest.approx(context)
    assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in lesson["additive_gradients"])
    assert torch.autograd.gradcheck(lambda *args: lesson["additive_attention"](*args)[2],
                                   lesson["additive_inputs"])


def test_one_source_annotation_receives_all_weight(lesson):
    torch = lesson["torch"]
    state, source, ws, wh, vector = lesson["additive_inputs"]
    _, weights, context = lesson["additive_attention"](state, source[:1], ws, wh, vector)
    assert weights.tolist() == [1]
    assert torch.equal(context, source[0])
