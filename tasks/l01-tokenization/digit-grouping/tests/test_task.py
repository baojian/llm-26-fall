"""Public digit-grouping checks for every file in submissions/.

Run one student:  uv run python scripts/tasks.py check tasks/l01-tokenization/digit-grouping <username>
Run everyone:     uv run python scripts/tasks.py check tasks/l01-tokenization/digit-grouping
"""

import importlib.util
import json
from pathlib import Path

import pytest

FUNCTION = "solve"
DATA = Path(__file__).resolve().parents[1] / "data"
CASES = [
    # (input, expected output)
    ('2026', ['202', '6']),
    ('Year 2026: 1,000,000', ['Year ', '202', '6', ': ', '1', ',', '000', ',', '000']),
    ('', []),
    ('12345678', ['123', '456', '78']),
    ('no digits', ['no digits']),
    ('0 12 345 6789', ['0', ' ', '12', ' ', '345', ' ', '678', '9']),
    ('0000123', ['000', '012', '3']),
    ('A1234B', ['A', '123', '4', 'B']),
    ('no  digits\n', ['no  digits\n']),
    ('\t  \r\n', ['\t  \r\n']),
    ('Code １２３４.', ['Code ', '１２３', '４', '.']),
    ('x²=½; n=1234', ['x²=½; n=', '123', '4']),
    ('3.14159', ['3', '.', '141', '59']),
    ('Room 101', ['Room ', '101']),
    ('2024-09-16', ['202', '4', '-', '09', '-', '16']),
    (DATA.joinpath('sample.txt').read_text(encoding='utf-8'),
     json.loads(DATA.joinpath('sample-chunks.json').read_text(encoding='utf-8'))),
]
PREDICTION_INPUTS = ['3.14159', 'Room 101', '2024-09-16']  # listed in instruction.md; students predict these
MIN_OWN_CASES = 2
MIN_NOTES_CHARS = 200

SUBMISSIONS = sorted(p for p in (Path(__file__).resolve().parents[1] / "submissions").glob("*.py") if not p.name.startswith((".", "_")))


def test_raw_text_fixture_preserves_every_character():
    text, expected = CASES[-1]
    assert text.endswith("\n")
    assert "".join(expected) == text


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
    assert len({text for text, _ in own}) >= MIN_OWN_CASES, "MY_CASES must contain at least two distinct inputs"
    given = {text for text, _ in CASES} | set(PREDICTION_INPUTS)
    for text, expected in own:
        assert text not in given, f"{text!r} is one of the given inputs; find a new one"
        assert getattr(module, FUNCTION)(text) == expected, f"solve({text!r}) fails your own case"


@pytest.mark.parametrize("submission", SUBMISSIONS, ids=[f"user.{p.stem}" for p in SUBMISSIONS])
def test_notes_answer_the_question(submission):
    notes = getattr(load_module(submission), "NOTES", "")
    assert len(notes.strip()) >= MIN_NOTES_CHARS and "REPLACE" not in notes, "write 3–5 sentences in NOTES"
