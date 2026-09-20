"""Execute the offline lecture and check the mathematical teaching examples."""

import json
import math
import re
import shutil
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
    monkeypatch.setattr("socket.socket.connect", no_network)
    monkeypatch.setattr("socket.create_connection", no_network)
    monkeypatch.chdir(LECTURE)
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


def test_lookup_shapes_and_one_hot_equivalence(lesson):
    assert lesson["e01_shapes"] == {
        "weight": (10, 4), "ids": (2, 3), "x": (2, 3, 4), "one_hot": (2, 3, 10),
    }
    assert lesson["e01_same"] is True


def test_classical_bridge_checks_association_and_low_rank_approximation(lesson):
    torch = lesson["torch"]
    assert lesson["bridge_total"].item() == 35
    assert lesson["bridge_pmi"][0, 0].item() == pytest.approx(math.log2(4 / 3))
    assert lesson["bridge_pmi"][2, 0].item() == float("-inf")
    assert lesson["bridge_ppmi"][2, 0].item() == 0
    assert torch.isfinite(lesson["bridge_vectors"]).all()
    assert tuple(lesson["bridge_vectors"].shape) == (3, 2)
    # The error of an optimal rank-2 approximation is the discarded singular value.
    residual = torch.linalg.norm(lesson["bridge_ppmi"] - lesson["bridge_reconstruction"])
    assert residual.item() == pytest.approx(lesson["bridge_s"][2].item(), abs=1e-12)


def test_lookup_visual_uses_the_notebook_initialization(lesson):
    figure = json.loads((LECTURE / "assets/lookup-values.json").read_text())
    expected = lesson["embedding"].weight.detach().tolist()
    assert len(figure["values"]) == len(expected) == 10
    for actual_row, expected_row in zip(figure["values"], expected):
        assert actual_row == pytest.approx(expected_row, abs=1e-7)


def test_ppmi_heatmap_matches_the_executable_counts(lesson):
    figure = json.loads((LECTURE / "assets/counts-ppmi.json").read_text())
    for trace, tensor_name in zip(figure["data"], ["bridge_counts", "bridge_ppmi"]):
        assert trace["x"] == ["drink", "hot", "drive"]
        assert trace["y"] == ["tea", "coffee", "car"]
        for actual, expected in zip(trace["z"], lesson[tensor_name].tolist()):
            assert actual == pytest.approx(expected, abs=1e-12)


def test_skipgram_hand_gradient_matches_autograd(lesson):
    assert lesson["e02_loss"] == pytest.approx(1.2873, abs=5e-5)
    assert lesson["e02_grads"]["c_pos"] == pytest.approx([-0.2689, -0.1345], abs=5e-5)
    assert lesson["e02_grads"]["c_neg"] == pytest.approx([0.6225, 0.3112], abs=5e-5)
    assert lesson["e02_grads"]["w"] == pytest.approx([0.4880, -0.8914], abs=5e-5)
    for name, gradient in lesson["e02_grads"].items():
        assert lesson["e02_autograd"][name] == pytest.approx(gradient, abs=1e-6)
    assert lesson["after"] == pytest.approx(0.6509, abs=5e-5)


def test_targets_are_shifted_within_each_sequence(lesson):
    assert lesson["inputs"].tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert lesson["targets"].tolist() == [[2, 3, 4, 5], [6, 7, 8, 9]]
    assert not lesson["targets"].is_contiguous()
    assert lesson["e03_shapes"] == {"embeddings": (2, 4, 4), "logits": (2, 4, 10)}


def test_tensor_contract_and_flattening_preserve_target_alignment(lesson):
    assert lesson["e01_dtypes"] == ["torch.int64", "torch.float32", "torch.float32"]
    assert lesson["token_ids"].device == lesson["embedding"].weight.device
    assert lesson["e03_view_failed"] is True
    assert lesson["e03_flat_targets"].tolist() == [2, 3, 4, 5, 6, 7, 8, 9]


def test_cross_entropy_equals_direct_nll_for_uniform_logits(lesson):
    assert lesson["e03_uniform_loss"] == pytest.approx(math.log(10), abs=1e-6)
    assert lesson["e03_manual_loss"] == pytest.approx(lesson["e03_uniform_loss"], abs=1e-6)


def test_stable_loss_handles_a_saturated_sigmoid(lesson):
    assert lesson["stability"][1.0]["literal"] == pytest.approx(lesson["stability"][1.0]["stable"])
    assert math.isinf(lesson["stability"][100.0]["literal"])
    assert lesson["stability"][100.0]["stable"] == pytest.approx(100.0)


def test_training_fits_a_compatible_batch_and_updates_embeddings(lesson):
    torch = lesson["torch"]
    assert len(lesson["e04_losses"]) == 201  # before training, then after each update
    assert lesson["e04_losses"][0] == pytest.approx(math.log(10), abs=1e-6)
    assert lesson["e04_losses"][-1] < 0.1
    assert torch.equal(lesson["e04_predictions"], lesson["targets"])
    assert not torch.equal(lesson["e04_initial_embedding"], lesson["model"].embedding.weight)


def test_heldout_probe_does_not_train_or_modify_gradients(lesson):
    torch = lesson["torch"]
    assert set(lesson["val_inputs"].flatten().tolist()).isdisjoint(lesson["inputs"].flatten().tolist())
    assert math.isfinite(lesson["e04_validation_loss"])
    assert lesson["e04_validation_requires_grad"] is False
    assert lesson["e04_mode_restored"] is True
    for kind in ["parameters", "gradients"]:
        before = lesson[f"e04_eval_{kind}_before"]
        after = lesson[f"e04_eval_{kind}_after"]
        assert all(torch.equal(left, right) for left, right in zip(before, after, strict=True))
    assert torch.equal(lesson["model"].embedding.weight.detach()[[0, 9]],
                       lesson["e04_initial_embedding"][[0, 9]])


def test_bigram_model_cannot_distinguish_earlier_context(lesson):
    torch = lesson["torch"]
    model = lesson["model"]
    with torch.no_grad():
        logits = model(torch.tensor([[1, 5], [7, 5]]))
    assert torch.equal(logits[0, 1], logits[1, 1])


def test_tying_preserves_values_but_combines_gradient_paths(lesson):
    assert lesson["e05_same_logits"] is True
    assert lesson["e05_untied_rows"] == [0, 1, 5, 7]
    assert lesson["e05_tied_rows"] == list(range(10))
    assert lesson["e05_gradient_sum"] is True
    assert lesson["tied"].output.weight is lesson["tied"].embedding.weight
    assert lesson["untied"].output.weight is not lesson["untied"].embedding.weight
    assert lesson["e06_toy_counts"] == {"untied": 80, "tied": 40}


def test_zero_output_initialization_has_a_first_step_learning_signal(lesson):
    torch = lesson["torch"]
    model = lesson["TinyLM"](10, 4)
    lesson["next_token_loss"](model(lesson["inputs"]), lesson["targets"]).backward()
    assert torch.count_nonzero(model.embedding.weight.grad) == 0
    assert torch.count_nonzero(model.output.weight.grad) > 0


def test_parameter_and_logit_storage_arithmetic(lesson):
    assert lesson["e06_exercise"] == {
        "one_table": 16_384_000, "untied": 32_768_000,
        "bf16_bytes": 32_768_000, "bf16_mib": 31.25,
    }
    assert lesson["e06_models"] == {
        "Qwen/Qwen3-0.6B": {
            "one_table": 155_582_464, "unique_parameters": 155_582_464, "bf16_mib": 296.75,
        },
        "Qwen/Qwen3-8B": {
            "one_table": 622_329_856, "unique_parameters": 1_244_659_712, "bf16_mib": 2374.0,
        },
    }
    assert lesson["e06_logit_mib"] == 593.5
    assert lesson["e06_projection"] == {"parameters": 40, "positions": 8, "forward_flops": 640}


def test_memory_ledger_includes_initialized_optimizer_moments(lesson):
    assert lesson["e06_training_mib"] == {
        "parameters": 62.5, "gradients": 62.5, "adam_moments": 125.0, "subtotal": 250.0,
    }
    measured = lesson["e06_measured_bytes"]
    assert measured["parameters"] == measured["gradients"] == 320
    assert measured["moments"] == 640
    assert measured["scalar_counters"] > 0
    assert measured["optimizer_state"] == measured["moments"] + measured["scalar_counters"]


def test_model_evidence_distinguishes_ids_from_allocated_rows(lesson):
    models = lesson["model_evidence"]["models"]
    for model in models:
        assert model["tokenizer_entries"] == 151_669
        assert model["max_token_id"] == 151_668
        assert model["embedding_rows"] == 151_936
        assert model["embedding_rows"] > model["max_token_id"]
        assert re.fullmatch(r"[0-9a-f]{40}", model["revision"])
        for source in model["sources"].values():
            assert f"/{model['revision']}/" in source["url"]
            assert re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
    assert models[0]["sources"]["tokenizer"]["sha256"] == models[1]["sources"]["tokenizer"]["sha256"]


@pytest.mark.parametrize("from_repo_root", [False, True])
def test_assets_load_from_a_student_copy_or_repository(lesson, tmp_path, monkeypatch, from_repo_root):
    relative = "slides/lecture-03/assets" if from_repo_root else "assets"
    assets = tmp_path / relative
    assets.mkdir(parents=True)
    shutil.copyfile(LECTURE / "assets/model-configs.json", assets / "model-configs.json")
    monkeypatch.chdir(tmp_path)
    namespace = {"Path": Path, "json": json}
    cell = next(cell for cell in NOTEBOOK["cells"] if cell["id"] == "e06-model-costs")
    execute_cell(cell, namespace)
    assert namespace["e06_models"] == lesson["e06_models"]


def test_missing_assets_give_a_setup_instruction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cell = next(cell for cell in NOTEBOOK["cells"] if cell["id"] == "e06-model-costs")
    with pytest.raises(FileNotFoundError, match="course launcher"):
        execute_cell(cell, {"Path": Path, "json": json})


def test_teaching_figure_matches_the_executable_example(lesson):
    figure = json.loads((LECTURE / "assets/tiny-lm-loss.json").read_text())
    trace = figure["data"][0]
    assert trace["x"][0] == 0 and trace["x"][-1] == 200
    assert trace["y"] == pytest.approx([lesson["e04_losses"][step] for step in trace["x"]], abs=1e-5)


def test_ppmi_handles_zero_and_negative_associations(lesson):
    assert lesson["TOTAL"] == 11716
    assert lesson["ppmi"]("information", "data") == pytest.approx(0.0944, abs=5e-5)
    assert lesson["ppmi"]("cherry", "pie") == pytest.approx(4.38, abs=5e-3)
    assert lesson["pmi"]("strawberry", "computer") == float("-inf")
    assert lesson["ppmi"]("digital", "pie") == 0


def test_optional_skipgram_learns_the_template_contexts(lesson):
    history = lesson["p02_history"]
    assert history[-1] < history[0] < 6 * math.log(2)
    for group in [
        {"king", "queen", "prince", "princess", "man", "woman", "boy", "girl"},
        {"cat", "dog", "horse", "bird"},
        {"rice", "bread", "fish", "apple"},
        {"beijing", "paris", "london", "tokyo"},
    ]:
        for word in group:
            assert {neighbor for neighbor, _ in lesson["nearest"](word, k=2)} & group


def test_notebook_exercises_follow_the_slide_order():
    ids = [match.group(1) for cell in NOTEBOOK["cells"]
           if (match := re.match(r"## ([EP]\d\d) ", "".join(cell["source"])))]
    assert ids == ["E01", "E02", "E03", "E04", "E05", "E06", "P01", "P02", "P03"]
    assert re.findall(r"Exercise (E\d\d) ·", SLIDES) == ids[:6]
