"""Public checks, compatible with scripts/tasks.py and task_feedback.py."""

import copy
import importlib.util
import json
import math
import reprlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DSIR = True
PROJECTION = "selected_ids" if DSIR else "contamination_pairs"
CASES = json.loads((ROOT / "data/checks.json").read_text())
PREDICTIONS = json.loads((ROOT / "data/examples.json").read_text())
SUBMISSIONS = sorted(path for path in (ROOT / "submissions").glob("*.py")
                     if not path.name.startswith((".", "_")))
USERS = [f"user.{path.stem}" for path in SUBMISSIONS]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(getattr(module, "solve", None)), "define solve(case)"
    return module


def canonical(case):
    return json.dumps(case, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def preview(value):
    return reprlib.repr(value)[:1200]


def assert_output(actual, expected, path="output"):
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: return a dictionary"
        assert actual.keys() == expected.keys(), f"{path}: return exactly the required fields {list(expected)}"
        for key in expected:
            assert_output(actual[key], expected[key], f"{path}[{key!r}]")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: return a list"
        assert len(actual) == len(expected), f"{path}: expected {len(expected)} items, got {len(actual)}"
        for index, (a, b) in enumerate(zip(actual, expected)):
            assert_output(a, b, f"{path}[{index}]")
    elif isinstance(expected, float):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual), f"{path}: return a finite number"
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10), f"{path}: {actual!r} != {expected!r} (rel_tol=1e-9, abs_tol=1e-10)"
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def check_case(module, case, expected):
    actual = module.solve(copy.deepcopy(case))
    try:
        assert_output(actual, expected)
    except AssertionError as error:
        raise AssertionError(
            f"Input: {preview(case)}\nExpected: {preview(expected)}\n"
            f"Actual: {preview(actual)}\nDetail: {error}"
        ) from None


@pytest.mark.parametrize("submission", SUBMISSIONS, ids=USERS)
def test_filename_is_a_lowercase_username(submission):
    assert submission.stem == submission.stem.lower(), "use your lowercase GitHub username"


@pytest.mark.parametrize("submission", SUBMISSIONS, ids=USERS)
@pytest.mark.parametrize("record", CASES, ids=[record["name"] for record in CASES])
def test_cases(submission, record):
    check_case(load(submission), record["case"], record["expected"])


@pytest.mark.parametrize("submission", SUBMISSIONS, ids=USERS)
def test_cases_input_order(submission):
    module = load(submission)
    for record in CASES:
        case = copy.deepcopy(record["case"])
        case["pool" if DSIR else "documents"].reverse()
        check_case(module, case, record["expected"])


@pytest.mark.parametrize("submission", SUBMISSIONS, ids=USERS)
def test_cases_invalid_values(submission):
    changes = ([{"num_buckets": 0}, {"alpha": 0}, {"alpha": float("nan")}, {"alpha": float("inf")},
                {"k": -1}, {"k": 99}, {"raw_fit": [[]]}, {"target_fit": []},
                {"uniforms": {"a": 0, "b": 0.5}}, {"uniforms": {"a": 0.5, "b": 1}},
                {"uniforms": {"a": 0.5}}]
               if DSIR else
               [{"n": 0}, {"num_hashes": 0}, {"num_bands": 0}, {"num_bands": 3},
                {"jaccard_threshold": 0}, {"jaccard_threshold": float("nan")},
                {"containment_threshold": float("inf")}, {"containment_threshold": 1.1}])
    module = load(submission)
    for change in changes:
        case = {**copy.deepcopy(CASES[0]["case"]), **change}
        with pytest.raises(ValueError):
            module.solve(case)
    case = copy.deepcopy(CASES[0]["case"])
    records = case["pool" if DSIR else "documents"]
    records[1]["id"] = records[0]["id"]
    with pytest.raises(ValueError):
        module.solve(case)


@pytest.mark.parametrize("submission", SUBMISSIONS, ids=USERS)
def test_predictions_match_solve(submission):
    module = load(submission)
    values = getattr(module, "PREDICTIONS", {})
    assert isinstance(values, dict) and values.keys() == PREDICTIONS.keys(), "cover the three named prediction inputs"
    for name, case in PREDICTIONS.items():
        assert_output(module.solve(copy.deepcopy(case))[PROJECTION], values[name])


@pytest.mark.parametrize("submission", SUBMISSIONS, ids=USERS)
def test_own_cases_are_new_and_pass(submission):
    module = load(submission)
    own = getattr(module, "MY_CASES", [])
    assert isinstance(own, list) and len(own) >= 2, "add two (case, expected_projection) pairs"
    assert all(isinstance(item, (list, tuple)) and len(item) == 2 for item in own)
    assert all(isinstance(case, dict) for case, _ in own), "each case must be a JSON-compatible dictionary"
    encoded = [canonical(case) for case, _ in own]
    assert len(set(encoded)) == len(encoded), "use distinct inputs"
    supplied = {canonical(record["case"]) for record in CASES} | {canonical(case) for case in PREDICTIONS.values()}
    for (case, expected), key in zip(own, encoded):
        assert key not in supplied, "design an input beyond the supplied cases"
        assert_output(module.solve(copy.deepcopy(case))[PROJECTION], expected)


@pytest.mark.parametrize("submission", SUBMISSIONS, ids=USERS)
def test_notes_are_present(submission):
    notes = getattr(load(submission), "NOTES", "")
    assert isinstance(notes, str) and len(notes.strip()) >= 200 and "REPLACE" not in notes, "Complete NOTES in your own words: 3–5 sentences, at least 200 characters; replace the placeholder."
    # A reviewer checks reasoning, experiment evidence, and the counterexamples.
    # Text length establishes presence only.
