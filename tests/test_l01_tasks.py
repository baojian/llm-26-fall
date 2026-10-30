"""Regression checks for the Lecture 01 public task checkers."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["digit-grouping", "gpt2-pretokenizer", "han-runs"])
def checker(request):
    path = ROOT / "tasks/l01-tokenization" / request.param / "tests/test_task.py"
    spec = importlib.util.spec_from_file_location(request.param.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_personal_cases_reject_duplicate_inputs(checker, tmp_path):
    candidate = tmp_path / "duplicate.py"
    candidate.write_text(
        "def solve(text):\n    return [text]\n"
        "MY_CASES = [('uniquereviewcase', ['uniquereviewcase'])] * 2\n"
    )
    with pytest.raises(AssertionError, match="distinct inputs"):
        checker.test_own_cases_are_new_and_pass(candidate)


def test_personal_cases_accept_two_distinct_new_inputs(checker, tmp_path):
    candidate = tmp_path / "distinct.py"
    candidate.write_text(
        "def solve(text):\n    return [text]\n"
        "MY_CASES = [('uniquereviewcase', ['uniquereviewcase']), "
        "('anotherreviewcase', ['anotherreviewcase'])]\n"
    )
    checker.test_own_cases_are_new_and_pass(candidate)
