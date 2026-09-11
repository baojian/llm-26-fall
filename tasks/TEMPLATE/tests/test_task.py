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
PREDICTION_INPUTS = ["REPLACE 1", "REPLACE 2", "REPLACE 3"]  # listed in instruction.md; students predict these
MIN_OWN_CASES = 2
MIN_NOTES_CHARS = 200

SUBMISSIONS = sorted(p for p in (Path(__file__).resolve().parents[1] / "submissions").glob("*.py") if not p.name.startswith((".", "_")))


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(path):
    return getattr(load_module(path), FUNCTION)


@pytest.mark.parametrize("submission", SUBMISSIONS, ids=[f"user.{p.stem}" for p in SUBMISSIONS])
def test_filename_is_a_lowercase_username(submission):
    assert submission.stem == submission.stem.lower(), "name the file <username>.py in lowercase"


@pytest.mark.parametrize("submission", SUBMISSIONS, ids=[f"user.{p.stem}" for p in SUBMISSIONS])
@pytest.mark.parametrize("text,expected", CASES)
def test_cases(submission, text, expected):
    assert load(submission)(text) == expected


@pytest.mark.parametrize("submission", SUBMISSIONS, ids=[f"user.{p.stem}" for p in SUBMISSIONS])
def test_predictions_match_solve(submission):
    module = load_module(submission)
    predictions = getattr(module, "PREDICTIONS", {})
    assert set(predictions) == set(PREDICTION_INPUTS), "PREDICTIONS must cover exactly the three inputs in instruction.md"
    for text, expected in predictions.items():
        assert getattr(module, FUNCTION)(text) == expected, f"solve({text!r}) differs from your prediction"


@pytest.mark.parametrize("submission", SUBMISSIONS, ids=[f"user.{p.stem}" for p in SUBMISSIONS])
def test_own_cases_are_new_and_pass(submission):
    module = load_module(submission)
    own = getattr(module, "MY_CASES", [])
    assert len(own) >= MIN_OWN_CASES, f"add at least {MIN_OWN_CASES} cases of your own to MY_CASES"
    given = {text for text, _ in CASES} | set(PREDICTION_INPUTS)
    for text, expected in own:
        assert text not in given, f"{text!r} is one of the given inputs; find a new one"
        assert getattr(module, FUNCTION)(text) == expected, f"solve({text!r}) fails your own case"


@pytest.mark.parametrize("submission", SUBMISSIONS, ids=[f"user.{p.stem}" for p in SUBMISSIONS])
def test_notes_answer_the_question(submission):
    notes = getattr(load_module(submission), "NOTES", "")
    assert len(notes.strip()) >= MIN_NOTES_CHARS and "REPLACE" not in notes, "write 3–5 sentences in NOTES"
