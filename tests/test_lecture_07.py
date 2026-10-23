"""Numerical and offline-execution checks for the synthetic scaling lesson."""

import csv
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean

import pytest

DECK = Path(__file__).resolve().parents[1] / "slides/lecture-07"
spec = importlib.util.spec_from_file_location("lecture07_scaling", DECK / "assets/scaling.py")
scaling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scaling)


def test_compute_memory_and_time_units():
    assert scaling.training_flops(50e6, 4e9) == scaling.training_flops(200e6, 1e9) == 1.2e18
    assert scaling.persistent_bytes(50e6) == 800_000_000
    assert scaling.persistent_bytes(200e6) == 3_200_000_000
    assert scaling.training_flops(50e6, 4e9) / 100e12 / 3600 == pytest.approx(10 / 3)


@pytest.mark.parametrize("multiple", [0.25, 1, 4, 16, 64])
def test_optimum_against_feasible_alternatives(multiple):
    n, d = scaling.optimal_allocation(multiple)
    budget = multiple * scaling.REFERENCE_COMPUTE
    assert scaling.training_flops(n, d) == pytest.approx(budget)
    best_loss = scaling.toy_loss(n, d)
    for factor in [0.1, 0.5, 0.9, 1.1, 2, 10]:
        assert scaling.toy_loss(n * factor, d / factor) > best_loss
    assert best_loss == pytest.approx(1.5 + 1.6 * multiple ** -0.25)


def test_fit_uses_reserved_scale_only_for_validation():
    with (DECK / "assets/synthetic-runs.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15
    assert {row["kind"] for row in rows} == {"synthetic"}
    fit = [row for row in rows if row["split"] == "fit"]
    holdout = [row for row in rows if row["split"] == "holdout"]
    levels = sorted({float(row["compute_multiple"]) for row in fit})
    assert levels == [0.25, 1, 4, 16]
    points = [(x, mean(float(row["loss_nats"]) for row in fit
                       if float(row["compute_multiple"]) == x)) for x in levels]
    k, gamma = scaling.fit_power_law(points, floor=1.5)
    assert k == pytest.approx(1.6)
    assert gamma == pytest.approx(0.25)
    prediction = scaling.predict_loss(64, k, gamma, floor=1.5)
    assert prediction == pytest.approx(1.5 + 1.6 / math.sqrt(8))
    assert {float(row["compute_multiple"]) for row in holdout} == {64}
    residual = mean(float(row["loss_nats"]) for row in holdout) - prediction
    assert residual == pytest.approx(0.16)
    assert residual > 0.05


def test_fitting_rejects_unidentified_or_invalid_inputs():
    for points, floor in [([], 1), ([(1, 2)], 1), ([(1, 2), (1, 3)], 1),
                          ([(0, 2), (2, 3)], 1), ([(1, 1), (2, 3)], 1),
                          ([(1, float("nan")), (2, 3)], 1),
                          ([(1, 2), (2, 3)], float("inf"))]:
        with pytest.raises(ValueError):
            scaling.fit_power_law(points, floor)
    for value in [0, -1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            scaling.training_flops(value, 100)


def test_prepared_browser_states_match_compute_and_loss():
    states = json.loads((DECK / "assets/budget-states.json").read_text())
    for state in states:
        n, d = state["parameters"], state["tokens"]
        assert scaling.training_flops(n, d) == pytest.approx(state["multiple"] * 1.2e18)
        assert state["loss"] == pytest.approx(scaling.toy_loss(n, d))
        for size_m, loss in zip(state["data"][0]["x"], state["data"][0]["y"]):
            size = size_m * 1e6
            tokens = state["multiple"] * 1.2e18 / (6 * size)
            assert loss == pytest.approx(scaling.toy_loss(size, tokens))


def test_notebook_runs_from_fresh_student_copy(tmp_path, monkeypatch):
    import shutil
    import sys

    shutil.copytree(DECK / "assets", tmp_path / "assets", ignore=shutil.ignore_patterns("__pycache__"))
    notebook = json.loads((DECK / "practice.ipynb").read_text())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"lecture07-cell-{index}", "exec"), namespace)
    assert namespace["total_compute"] == pytest.approx(2.52e19)
