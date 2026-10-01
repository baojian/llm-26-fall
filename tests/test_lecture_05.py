"""Execute the offline Lecture 05 notebook and verify its Transformer claims."""

import json
import hashlib
import math
import platform
import re
from pathlib import Path

import nbformat
import pytest
import torch


LECTURE = Path(__file__).resolve().parents[1] / "slides/lecture-05"
NOTEBOOK = json.loads((LECTURE / "lecture-05-exercise.ipynb").read_text())
EXERCISES = {"E01": 5, "E02": 5, "E03": 4, "E04": 5, "E05": 6}
CHECKPOINTS = [0, 1, 5, 10, 20, 50, 100, 200]


@pytest.fixture(scope="module")
def lesson(request):
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(threads))
    torch.set_num_threads(1)

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


def flat(rows):
    return [value for row in rows for value in row]


def markdown_text(cell_id):
    cell = next(cell for cell in NOTEBOOK["cells"] if cell["id"] == cell_id)
    return "".join(cell["source"])


def test_notebook_is_clean_and_exercises_follow_their_concepts():
    nbformat.validate(nbformat.from_dict(NOTEBOOK))
    assert NOTEBOOK["metadata"]["kernelspec"]["name"] == "python3"
    ids = [cell["id"] for cell in NOTEBOOK["cells"]]
    assert len(ids) == len(set(ids))
    for cell in NOTEBOOK["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
    headings = [match.group(1) for cell in NOTEBOOK["cells"] if cell["cell_type"] == "markdown"
                if (match := re.match(r"## (E\d\d) ·", "".join(cell["source"])))]
    assert headings == list(EXERCISES)
    for exercise, minutes in EXERCISES.items():
        prompt, answer = ids.index(f"{exercise.lower()}-prompt"), ids.index(f"{exercise.lower()}-answer")
        assert markdown_text(ids[prompt]).startswith(f"## {exercise} ·")
        assert f"**{minutes} minutes.**" in markdown_text(ids[prompt])
        assert f"| {exercise} | {minutes} min |" in markdown_text("introduction")
        between = NOTEBOOK["cells"][prompt + 1:answer]
        assert any(cell["cell_type"] == "code" for cell in between)
        assert markdown_text(ids[answer]).startswith("**Check:**")
    text = "".join("".join(cell["source"]) for cell in NOTEBOOK["cells"]).lower()
    assert "todo" not in text and "ungraded" in text and "same expectations for all students" in text


def test_exercise_order_matches_slides():
    slides = LECTURE / "slides.md"
    found = list(dict.fromkeys(re.findall(r"Exercise (E\d\d) ·", slides.read_text())))
    assert found == list(EXERCISES)


def test_toy_data_is_shifted_and_floor_is_log2_over_4(lesson):
    assert lesson["vocabulary"] == ["BOS", "red", "blue", "key", "opens", "closes", "EOS"]
    assert lesson["documents"].tolist() == [[0, 1, 3, 4, 6], [0, 2, 3, 5, 6]]
    assert lesson["X"].tolist() == [[0, 1, 3, 4], [0, 2, 3, 5]]
    assert lesson["y"].tolist() == [[1, 3, 4, 6], [2, 3, 5, 6]]
    assert (lesson["B"], lesson["T"], lesson["d"], lesson["n_heads"], lesson["d_ff"],
            lesson["n_layers"], lesson["V"], lesson["max_length"]) == (2, 4, 16, 4, 32, 2, 7, 4)
    assert lesson["loss_floor"] == pytest.approx(0.173286795, abs=1e-9)
    # The floor is attained by p=1/2 after BOS and certainty elsewhere.
    probabilities = [0.5, 1, 1, 1, 0.5, 1, 1, 1]
    assert -sum(map(math.log, probabilities)) / 8 == pytest.approx(lesson["loss_floor"])


def test_alignment_example_matches_the_independent_hand_calculation(lesson):
    assert lesson["alignment_scores"].tolist() == [0.0, 1.0]
    weight = math.e / (1 + math.e)
    assert lesson["alignment_weights"].tolist() == pytest.approx([1 - weight, weight])
    assert lesson["alignment_context"].tolist() == pytest.approx([weight, 1 - weight])


def test_single_head_worked_example_matches_closed_form(lesson):
    denominator = 2 * math.e + 1
    assert lesson["head_scores"].tolist() == pytest.approx([1, 0, 1])
    assert lesson["head_weights"].tolist() == pytest.approx(
        [math.e / denominator, 1 / denominator, math.e / denominator]
    )
    assert lesson["head_output"].tolist() == pytest.approx(
        [3 * math.e / denominator, (math.e + 2) / denominator]
    )
    allowed_weight = math.e / (math.e + 1)
    assert lesson["head_masked_weights"].tolist() == pytest.approx(
        [allowed_weight, 1 - allowed_weight, 0]
    )
    assert lesson["head_masked_output"].tolist() == pytest.approx(
        [allowed_weight, 2 * (1 - allowed_weight)]
    )


def test_introductory_mask_blocks_future_values_and_gradients(lesson):
    torch.testing.assert_close(lesson["head_masked_changed"], lesson["head_masked_output"],
                               rtol=0, atol=0)
    assert (lesson["head_unmasked_changed"] - lesson["head_output"]).abs().max() > 10
    for gradients in (lesson["head_key_grad"], lesson["head_value_grad"]):
        assert torch.count_nonzero(gradients[2]) == 0
        assert gradients[:2].abs().sum() > 0
    # Dropping probability mass after softmax leaves the allowed row unnormalized.
    assert lesson["head_wrong_weights"].sum().item() == pytest.approx(
        (math.e + 1) / (2 * math.e + 1)
    )
    assert lesson["head_masked_weights"].sum().item() == pytest.approx(1)


def test_split_and_merge_heads_move_feature_chunks(lesson):
    x = torch.arange(2 * 3 * 8, dtype=torch.float64).reshape(2, 3, 8)
    split = lesson["split_heads"](x, 4)
    assert tuple(split.shape) == (2, 4, 3, 2)
    for head in range(4):
        assert torch.equal(split[:, head], x[..., 2 * head:2 * head + 2])
    assert torch.equal(lesson["merge_heads"](split), x)
    assert lesson["e01_shapes"] == {
        "input": (2, 4, 16), "projected": (2, 4, 16), "split heads": (2, 4, 4, 4),
        "scores": (2, 4, 4, 4), "merged": (2, 4, 16), "output": (2, 4, 16),
    }
    with pytest.raises(ValueError, match="divisible"):
        lesson["split_heads"](torch.zeros(1, 2, 10), 4)
    with pytest.raises(ValueError, match="divisible"):
        lesson["MultiHeadAttention"](10, 4)
    assert "divisible" in lesson["e01_divisibility_error"]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("length", [1, 4])
def test_multi_head_attention_matches_pytorch_reference(lesson, causal, length):
    torch.manual_seed(3)
    attention = lesson["MultiHeadAttention"](16, 4).double()
    assert all(layer.bias is None for layer in
               (attention.query, attention.key, attention.value, attention.output))
    reference = torch.nn.MultiheadAttention(16, 4, bias=False, batch_first=True).double()
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.cat(
            [attention.query.weight, attention.key.weight, attention.value.weight]))
        reference.out_proj.weight.copy_(attention.output.weight)
    x = torch.randn(2, length, 16, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    blocked = torch.ones(length, length, dtype=torch.bool).triu(1) if causal else None
    actual, weights = attention(x, causal)
    expected, _ = reference(reference_x, reference_x, reference_x, attn_mask=blocked, need_weights=False)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    probe = torch.randn_like(actual)
    grad = torch.autograd.grad((actual * probe).sum(), x)[0]
    expected_grad = torch.autograd.grad((expected * probe).sum(), reference_x)[0]
    torch.testing.assert_close(grad, expected_grad, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, length, dtype=torch.float64))
    if causal:
        assert torch.equal(weights.triu(1), torch.zeros_like(weights))
    loop = lesson["heads_loop_reference"](attention, x.detach(), causal)
    torch.testing.assert_close(actual, loop, rtol=1e-10, atol=1e-10)


def test_e01_reference_agreement_and_wrong_reshape_control(lesson):
    assert lesson["e01_forward_error"] < 1e-10
    assert lesson["e01_gradient_error"] < 1e-10
    assert lesson["e01_wrong_shape"] == (2, 4, 16)
    assert lesson["e01_wrong_error"] > 0.1
    assert lesson["correct_change"] > 1e-3
    assert lesson["wrong_change"] == 0


def test_e02_permutation_fixture_is_independently_reproduced(lesson):
    table = [[math.sin(p / 10000 ** (i / 4)) if i % 2 == 0 else math.cos(p / 10000 ** ((i - 1) / 4))
              for i in range(4)] for p in range(4)]
    assert flat(lesson["e02_pe"].tolist()) == pytest.approx(flat(table), abs=1e-15)
    assert table[1] == pytest.approx([math.sin(1), math.cos(1), math.sin(0.01), math.cos(0.01)])

    def attend(rows):
        output = []
        for query in rows:
            scores = [sum(a * b for a, b in zip(query, key)) / 2 for key in rows]
            total = sum(math.exp(score) for score in scores)
            output.append([sum(math.exp(s) / total * key[j] for s, key in zip(scores, rows))
                           for j in range(4)])
        return output

    identity = [[float(i == j) for j in range(4)] for i in range(4)]
    permutation = [3, 1, 2, 0]
    fixture = lesson["e02_fixture"]
    assert fixture["tokens"] == ["bank", "of", "the", "river"]
    assert fixture["permutation"] == permutation
    plain = attend(identity)
    assert plain[0][0] == pytest.approx(math.exp(0.5) / (math.exp(0.5) + 3), abs=1e-12)
    assert flat(fixture["plain_output"]) == pytest.approx(flat(plain), abs=1e-12)
    positioned = attend([[a + b for a, b in zip(row, pe)] for row, pe in zip(identity, table)])
    assert flat(fixture["positioned_output"]) == pytest.approx(flat(positioned), abs=1e-12)
    shuffled = attend([[a + b for a, b in zip(identity[source], pe)]
                       for source, pe in zip(permutation, table)])
    restored = [shuffled[permutation.index(i)] for i in range(4)]
    assert flat(fixture["positioned_restored"]) == pytest.approx(flat(restored), abs=1e-12)
    assert fixture["plain_error"] < 1e-12
    assert fixture["positioned_error"] > 0.1
    with pytest.raises(ValueError, match="even"):
        lesson["sinusoidal_positions"](3, 5)


def test_e03_layer_norm_residual_and_axes(lesson):
    assert lesson["e03_normalized"].tolist() == pytest.approx([-1 / math.sqrt(1 + 1e-5), 1 / math.sqrt(1 + 1e-5)], abs=1e-15)
    assert lesson["e03_residual"].tolist() == [3.0, 2.0]
    assert lesson["e03_feature_means"].abs().max().item() < 1e-12
    torch.testing.assert_close(lesson["e03_feature_variances"],
                               torch.ones(2, 4, dtype=torch.float64), rtol=0, atol=1e-5)
    assert lesson["e03_ffn_change"].tolist()[0] == 0
    assert lesson["e03_ffn_change"][2:].tolist() == [0, 0]
    assert lesson["e03_ffn_change"][1] > 0
    assert lesson["e03_attention_change"][0] == 0
    assert (lesson["e03_attention_change"][1:] > 1e-3).all()
    assert lesson["e03_pre_post_difference"] > 1
    assert abs(lesson["e03_post_output"].mean().item()) < 1e-12
    assert lesson["e03_pre_output"].mean().item() == pytest.approx(5, abs=0.5)


@pytest.mark.parametrize("norm_first", [True, False])
def test_block_matches_its_pre_or_post_formula(lesson, norm_first):
    torch.manual_seed(5)
    block = lesson["Block"](16, 4, 32, norm_first).double()
    norms = [block.attention_norm, block.ffn_norm]
    assert all(isinstance(norm, torch.nn.LayerNorm) and norm.eps == 1e-5
               and norm.elementwise_affine and norm.normalized_shape == (16,) for norm in norms)
    assert block.ffn.up.bias is None and block.ffn.down.bias is None
    with torch.no_grad():
        for norm in norms:
            norm.weight.uniform_(0.5, 1.5)
            norm.bias.uniform_(-0.5, 0.5)
    x = torch.randn(2, 4, 16, dtype=torch.float64)

    def layer_norm(value, norm):
        mean = value.mean(-1, keepdim=True)
        variance = value.var(-1, unbiased=False, keepdim=True)
        return (value - mean) / torch.sqrt(variance + 1e-5) * norm.weight + norm.bias

    def ffn(value):
        return torch.relu(value @ block.ffn.up.weight.T) @ block.ffn.down.weight.T

    with torch.no_grad():
        if norm_first:
            h = x + block.attention(layer_norm(x, block.attention_norm))[0]
            expected = h + ffn(layer_norm(h, block.ffn_norm))
        else:
            h = layer_norm(x + block.attention(x)[0], block.attention_norm)
            expected = layer_norm(h + ffn(h), block.ffn_norm)
        torch.testing.assert_close(block(x), expected, rtol=1e-10, atol=1e-10)


def test_e04_counts_are_independent_of_notebook_arithmetic(lesson):
    model = lesson["DecoderLM"]()
    unique, registered = {}, 0
    for _, parameter in model.named_parameters(remove_duplicate=False):
        registered += parameter.numel()
        unique[id(parameter)] = parameter.numel()
    assert sum(unique.values()) == 4432
    assert registered == 4544
    block = model.blocks[0]
    by_part = {name: sum(p.numel() for p in module.parameters()) for name, module in
               [("attention", block.attention), ("ffn", block.ffn)]}
    assert by_part == {"attention": 1024, "ffn": 1024}
    assert sum(p.numel() for p in block.attention_norm.parameters()) == 32
    assert lesson["e04_formula"] == {"attention": 1024, "ffn": 1024, "layer norms": 64}
    assert lesson["e04_counts"] == {
        "block": 2112, "blocks": 4224, "token table": 112, "position table": 64,
        "final norm": 32, "model tied": 4432, "model untied": 4544,
    }
    assert lesson["e04_tied"] is True and lesson["e04_copied_is_tied"] is False
    assert model.lm_head.weight is model.token_embedding.weight
    assert model.lm_head.bias is None
    assert isinstance(model.position_embedding, torch.nn.Embedding)
    assert tuple(model.position_embedding.weight.shape) == (4, 16)
    assert not any(isinstance(module, torch.nn.Dropout) for module in model.modules())
    assert sum(isinstance(module, torch.nn.LayerNorm) for module in model.modules()) == 5
    assert sum(p.numel() for p in lesson["DecoderLM"](n_heads=2).parameters()) == 4432


def test_initialization_policy(lesson):
    torch.manual_seed(0)
    model = lesson["DecoderLM"](width=64, max_length=64, vocab_size=64, tie_weights=False)
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            assert torch.equal(module.weight, torch.ones_like(module.weight))
            assert torch.equal(module.bias, torch.zeros_like(module.bias))
        elif isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            assert module.weight.std().item() == pytest.approx(0.02, rel=0.25)
            assert module.weight.mean().abs().item() < 0.01


def test_e05_causality_and_negative_controls(lesson):
    assert tuple(lesson["e05_logits"].shape) == (2, 4, 7)
    assert lesson["e05_prefix_change"] == 0
    assert lesson["e05_last_change"] > 1e-3
    assert lesson["e05_unmasked_prefix_change"] > 1e-4
    gradient = lesson["e05_state_gradient"]
    assert gradient[3].item() == 0 and (gradient[:3] > 0).all()
    assert lesson["e05_unmasked_state_gradient"][3].item() > 0
    weights = lesson["e05_weights"]
    assert torch.equal(weights.triu(1), torch.zeros_like(weights))
    assert torch.equal(weights[..., 0, 0], torch.ones_like(weights[..., 0, 0]))
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 4))


def test_full_model_causality_in_float64(lesson):
    torch.manual_seed(11)
    model = lesson["DecoderLM"]().double()
    ids = lesson["X"]
    states = model.embed(ids).detach().requires_grad_()
    logits = model.lm_head(model.forward_hidden(states))
    for position in range(4):
        gradient = torch.autograd.grad(logits[:, position].square().sum(), states, retain_graph=True)[0]
        per_position = gradient.abs().amax(dim=(0, 2))
        assert (per_position[position + 1:] == 0).all()
        assert (per_position[:position + 1] > 0).all()
    changed = ids.clone()
    changed[:, 2] = 6
    with torch.no_grad():
        assert torch.equal(model(changed)[:, :2], model(ids)[:, :2])
        assert (model(changed, causal=False) - model(ids, causal=False))[:, :2].abs().max() > 0


def test_model_input_edge_cases(lesson):
    model = lesson["DecoderLM"]()
    X = lesson["X"]
    with torch.no_grad():
        single = model(X[:, :1])
        assert tuple(single.shape) == (2, 1, 7)
        assert torch.isfinite(single).all()
        # One-token logits equal the first position of the longer sequence.
        torch.testing.assert_close(single, model(X)[:, :1], rtol=1e-6, atol=1e-6)
    for bad in [lesson["documents"], X[:, :0], X[0], X.float(), torch.tensor([[0, 7]]),
                torch.tensor([[-1, 0]])]:
        with pytest.raises(ValueError):
            model(bad)
    assert "between 1 and 4" in lesson["e05_length_error"]


def test_tiny_batch_fit_meets_criterion_without_beating_the_floor(lesson):
    for losses, norms in [(lesson["pre_losses"], lesson["pre_gradient_norms"]),
                          (lesson["post_losses"], lesson["post_gradient_norms"])]:
        assert len(losses) == len(norms) == 201
        assert all(math.isfinite(value) for value in losses + norms)
        assert all(loss >= lesson["loss_floor"] - 1e-6 for loss in losses)
        assert losses[0] == pytest.approx(math.log(7), abs=0.1)
    assert lesson["pre_losses"][-1] < 0.20
    assert lesson["checkpoints"] == CHECKPOINTS
    y = lesson["y"]
    assert torch.equal(lesson["e05_predictions"][:, 1:], y[:, 1:])
    bos = lesson["e05_bos_probabilities"]
    assert bos.sum(-1).min().item() > 0.95
    assert (bos - 0.5).abs().max().item() < 0.05
    assert torch.equal(lesson["post_predictions"][:, 1:], y[:, 1:])
    trained = lesson["model"]
    assert all(torch.isfinite(p.grad).all() for p in trained.parameters())
    assert trained.lm_head.weight is trained.token_embedding.weight


def test_pre_post_comparison_changes_only_norm_order(lesson):
    initial = lesson["initial_state"]
    pre, post = lesson["DecoderLM"](), lesson["DecoderLM"](norm_first=False)
    pre.load_state_dict(initial)
    post.load_state_dict(initial)
    assert pre.state_dict().keys() == post.state_dict().keys() == initial.keys()
    assert all(torch.equal(pre.state_dict()[k], post.state_dict()[k]) for k in initial)
    assert [b.norm_first for b in pre.blocks] == [True, True]
    assert [b.norm_first for b in post.blocks] == [False, False]
    assert lesson["post_model"].final_norm is not None
    # A fresh short run from the same state reproduces the recorded history.
    losses, norms = lesson["train"](pre, steps=5)
    assert losses == pytest.approx(lesson["pre_losses"][:6], rel=1e-5, abs=1e-6)
    assert norms == pytest.approx(lesson["pre_gradient_norms"][:6], rel=1e-5, abs=1e-6)


def test_teaching_curve_preserves_every_update_and_records_its_environment(lesson):
    figure = json.loads((LECTURE / "assets/norm-comparison.json").read_text())
    metadata = figure["layout"]["meta"]
    assert metadata["seed"] == 7 and metadata["updates_per_variant"] == 200
    assert metadata["device"] == "cpu" and metadata["threads"] == 1
    assert metadata["timezone"] == "Asia/Shanghai"
    assert len(metadata["git_revision"]) == 40 and len(metadata["notebook_sha256"]) == 64
    assert metadata["notebook_sha256"] == hashlib.sha256(
        (LECTURE / "lecture-05-exercise.ipynb").read_bytes()).hexdigest()
    for trace, prefix in zip(figure["data"][:2], ["pre", "post"]):
        assert trace["x"] == list(range(201))
        losses = trace["y"]
        assert len(losses) == 201 and all(math.isfinite(value) for value in losses)
        assert min(losses) >= lesson["loss_floor"] - 1e-6
        assert losses[-1] < 0.20
        assert losses[:6] == pytest.approx(lesson[f"{prefix}_losses"][:6], rel=1e-4, abs=1e-5)
        # The float32 plateau exit can move across BLAS/platform versions.
        # On the recorded environment, verify the entire unsmoothed history.
        if metadata["platform"] == platform.platform() and metadata["torch"] == torch.__version__:
            assert losses == pytest.approx(lesson[f"{prefix}_losses"], rel=1e-4, abs=1e-5)
    assert figure["data"][2]["y"] == pytest.approx([lesson["loss_floor"]] * 2)


def test_browser_fixture_and_frequency_plot_agree_with_the_notebook(lesson):
    fixture = json.loads((LECTURE / "assets/position-values.json").read_text())
    assert fixture["tokens"] == lesson["e02_fixture"]["tokens"]
    assert fixture["permutation"] == lesson["e02_fixture"]["permutation"]
    assert fixture["vectors"] == torch.eye(4).tolist()
    figure = json.loads((LECTURE / "assets/position-frequencies.json").read_text())
    for trace, frequency in zip(figure["data"], [1, 0.1, 0.01]):
        assert trace["x"] == list(range(32))
        assert trace["y"] == pytest.approx([math.sin(p * frequency) for p in range(32)])
