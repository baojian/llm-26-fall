"""Checker for every file in submissions/. Task authors edit FUNCTION and CASES only.

Run one student:  uv run python scripts/tasks.py check tasks/<lecture>/<slug> <username>
Run everyone:     uv run python scripts/tasks.py check tasks/<lecture>/<slug>
"""

import importlib.util
from pathlib import Path

import pytest

FUNCTION = "solve"
CASES = [
    # (input, expected output)
    ("REPLACE", ["REPLACE"]),
    ("", []),
]

SUBMISSIONS = sorted(p for p in (Path(__file__).resolve().parents[1] / "submissions").glob("*.py"))


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, FUNCTION)


@pytest.mark.parametrize("submission", SUBMISSIONS, ids=[f"@{p.stem}" for p in SUBMISSIONS])
def test_filename_is_a_lowercase_username(submission):
    assert submission.stem == submission.stem.lower(), "name the file <username>.py in lowercase"


@pytest.mark.parametrize("submission", SUBMISSIONS, ids=[f"@{p.stem}" for p in SUBMISSIONS])
@pytest.mark.parametrize("text,expected", CASES)
def test_cases(submission, text, expected):
    assert load(submission)(text) == expected
