"""Check the scoring and uncertainty examples used by Lecture 08."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

DECK = Path(__file__).resolve().parents[1] / "slides/lecture-08"
spec = importlib.util.spec_from_file_location("lecture08_evaluation", DECK / "assets/evaluation.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_exact_audit_detects_unicode_and_cross_split_duplicates():
    records = json.loads((DECK / "assets/toy-documents.json").read_text())["records"]
    assert evaluation.exact_duplicate_groups(records) == [["train-1", "train-2"], ["train-4", "dev-1"]]
    assert evaluation.normalize_document("a  b") != evaluation.normalize_document("a b")
    assert evaluation.normalize_document("Noa") != evaluation.normalize_document("noa")
    assert evaluation.jaccard(evaluation.word_shingles(records[0]["text"]),
                              evaluation.word_shingles(records[2]["text"])) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        evaluation.exact_duplicate_groups([records[0], records[0]])
    with pytest.raises(ValueError):
        evaluation.jaccard(set(), set())


def test_completion_scoring_and_length_normalization_change_the_decision():
    short, long = [-1.1, -1.3], [-0.7, -0.8, -0.8, -0.9]
    assert evaluation.completion_score(short) == pytest.approx(-2.4)
    assert evaluation.completion_score(long) == pytest.approx(-3.2)
    assert evaluation.completion_score(short) > evaluation.completion_score(long)
    assert evaluation.completion_score(short, per_token=True) == pytest.approx(-1.2)
    assert evaluation.completion_score(long, per_token=True) == pytest.approx(-0.8)
    for invalid in ([], [0.1], [float("nan")], [-float("inf")]):
        with pytest.raises(ValueError):
            evaluation.completion_score(invalid)


def test_pass_at_k_matches_enumerated_subsets():
    from itertools import combinations

    for n in range(1, 7):
        for c in range(n + 1):
            for k in range(1, n + 1):
                subsets = list(combinations(range(n), k))
                exact = sum(any(index < c for index in subset) for subset in subsets) / len(subsets)
                assert evaluation.pass_at_k(n, c, k) == pytest.approx(exact)
    assert evaluation.pass_at_k(10, 2, 5) == pytest.approx(7 / 9)
    for invalid in ((0, 0, 1), (4, 5, 1), (4, 2, 5), (4, -1, 1), (4, 2, 0), (True, 0, 1)):
        with pytest.raises(ValueError):
            evaluation.pass_at_k(*invalid)


def test_synthetic_contamination_reverses_the_observed_order():
    baseline, candidate = evaluation.comparison_predictions()
    assert evaluation.accuracy(baseline) == 0.6
    assert evaluation.accuracy(candidate) == 0.64
    assert evaluation.accuracy(baseline + [1] * 19 + [0]) > evaluation.accuracy(candidate + [1] * 11 + [0] * 9)
    assert sum(a != b for a, b in zip(baseline, candidate)) == 24


def test_bootstrap_preserves_pairing_and_reports_limited_evidence():
    same = [0, 1, 1, 0]
    result = evaluation.paired_bootstrap(same, same, draws=100)
    assert (result["difference"], result["low"], result["high"]) == (0, 0, 0)
    result = evaluation.paired_bootstrap([0] * 8, [1] * 8, draws=100)
    assert (result["difference"], result["low"], result["high"]) == (1, 1, 1)
    baseline, candidate = evaluation.comparison_predictions()
    result = evaluation.paired_bootstrap(baseline, candidate)
    assert result["difference"] == pytest.approx(0.04)
    assert result["low"] < 0 < result["high"]
    reversed_result = evaluation.paired_bootstrap(candidate, baseline)
    assert result["low"] == pytest.approx(-reversed_result["high"])
    with pytest.raises(ValueError):
        evaluation.paired_bootstrap([1], [0, 1])
    with pytest.raises(ValueError):
        evaluation.paired_bootstrap([], [])


def test_notebook_runs_offline_from_a_fresh_working_copy(tmp_path, monkeypatch):
    import shutil
    import sys

    shutil.copytree(DECK / "assets", tmp_path / "assets", ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    notebook = json.loads((DECK / "practice.ipynb").read_text())
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"lecture08-cell-{index}", "exec"), namespace)
    assert namespace["bootstrap_result"]["difference"] == pytest.approx(0.04)
    assert math.isclose(namespace["pass5"], 7 / 9)
