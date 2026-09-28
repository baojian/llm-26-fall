"""Execute the offline lesson and verify its numerical and causal claims."""

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
    assert ids == ["E01", "E02", "E03", "E04", "E05", "P01", "P02"]
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


def test_hand_attention_example_and_browser_fixture(lesson):
    weights = [math.e / (2 * math.e + 1), 1 / (2 * math.e + 1), math.e / (2 * math.e + 1)]
    assert lesson["toy_weights"][0, 1].tolist() == pytest.approx(weights, abs=1e-12)
    assert lesson["toy_outputs"][0, 1].tolist() == pytest.approx([1.689275193006, 0.733043605245], abs=1e-12)
    fixture = json.loads((LECTURE / "assets/attention-values.json").read_text())
    for field, tensor in [("query", "toy_q"), ("key", "toy_k"), ("value", "toy_v")]:
        assert fixture[field] == lesson[tensor][0].tolist()
    figure = json.loads((LECTURE / "assets/attention-demo.json").read_text())
    for actual, expected in zip(figure["data"][0]["z"], lesson["causal_weights"][0].tolist()):
        assert actual == pytest.approx(expected, abs=1e-12)


def test_masking_normalizes_only_the_allowed_keys(lesson):
    torch = lesson["torch"]
    weights = lesson["causal_weights"]
    assert torch.equal(weights.masked_select(~lesson["allowed"]), torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(weights.sum(-1), torch.ones(1, 3, dtype=torch.float64))
    assert lesson["causal_outputs"][0, 1].tolist() == pytest.approx([0.731058578630, 0.537882842740], abs=1e-12)
    assert lesson["post_softmax_mask"][0, 1].sum().item() == pytest.approx(0.577681201748, abs=1e-12)
    assert torch.equal(lesson["causal_changed"][:, :2], lesson["causal_outputs"][:, :2])
    assert (lesson["unmasked_changed"] - lesson["toy_outputs"])[0, 1].tolist() == pytest.approx([4.223187982515] * 2)


@pytest.mark.parametrize("mask_mode", ["none", "shared", "per_batch"])
def test_batched_attention_matches_pytorch_with_unequal_lengths_and_widths(lesson, mask_mode):
    torch = lesson["torch"]
    generator = torch.Generator().manual_seed(37)
    q = torch.randn(2, 3, 4, dtype=torch.float64, generator=generator)
    k = torch.randn(2, 5, 4, dtype=torch.float64, generator=generator)
    v = torch.randn(2, 5, 2, dtype=torch.float64, generator=generator)
    allowed = None
    if mask_mode != "none":
        allowed = torch.tensor([[True, False, True, False, False],
                                [True, True, False, False, True],
                                [False, False, True, True, False]])
        if mask_mode == "per_batch":
            allowed = torch.stack([allowed, allowed.flip(-1)])
    actual, weights = lesson["scaled_attention"](q, k, v, allowed)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    assert tuple(actual.shape) == (2, 3, 2)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 3, dtype=torch.float64))


def test_single_allowed_key_returns_its_value(lesson):
    torch = lesson["torch"]
    mask = torch.tensor([[False, True, False]]).expand(3, 3)
    output, weights = lesson["scaled_attention"](lesson["toy_q"], lesson["toy_k"], lesson["toy_v"], mask)
    assert torch.equal(output, lesson["toy_v"][:, 1:2].expand(1, 3, 2))
    assert torch.equal(weights[:, :, 1], torch.ones(1, 3, dtype=torch.float64))


@pytest.mark.parametrize("failure", ["rank", "key_width", "value_length", "dtype", "mask_type", "mask_shape", "empty_row"])
def test_invalid_attention_inputs_fail_clearly(lesson, failure):
    torch = lesson["torch"]
    q, k, v = (lesson[name] for name in ["toy_q", "toy_k", "toy_v"])
    mask = lesson["allowed"].clone()
    if failure == "rank":
        q = q[0]
    elif failure == "key_width":
        k = k[:, :, :1]
    elif failure == "value_length":
        v = v[:, :2]
    elif failure == "dtype":
        q = q.float()
    elif failure == "mask_type":
        mask = mask.float()
    elif failure == "mask_shape":
        mask = torch.ones(4, 4, dtype=torch.bool)
    elif failure == "empty_row":
        mask[1] = False
    with pytest.raises(ValueError):
        lesson["scaled_attention"](q, k, v, mask)


def test_gradient_reference_and_finite_differences_agree(lesson):
    assert lesson["forward_error"] < 1e-10
    assert max(lesson["gradient_errors"].values()) < 1e-10
    assert lesson["gradcheck_ok"] is True


def test_causality_before_learned_projections_and_negative_control(lesson):
    assert lesson["prefix_error"] == 0
    assert lesson["future_gradient_max"] == 0
    assert lesson["prefix_gradient"][:, :2].abs().max().item() > 0
    assert lesson["unmasked_prefix_change"] > 1e-3
    torch = lesson["torch"]
    output, _ = lesson["head"](lesson["head_inputs"].detach())
    gradients = torch.autograd.grad(output.square().sum(), tuple(lesson["head"].parameters()))
    assert all(torch.isfinite(gradient).all() and gradient.abs().max() > 0 for gradient in gradients)


def test_resource_arithmetic_and_paired_permutation(lesson):
    assert lesson["score_mib"] == {512: 1, 1024: 4, 2048: 16}
    assert lesson["dense_head_flops"][1024] == 268_435_456
    assert lesson["dense_head_flops"][2048] == 4 * lesson["dense_head_flops"][1024]
    assert lesson["pair_permutation_error"] < 1e-12
