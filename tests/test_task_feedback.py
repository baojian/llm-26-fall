"""Exercise student feedback end to end, including failed and unsafe PR shapes."""

import ast
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import task_feedback as feedback

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "tasks/example/word-count"
EXAMPLE_SOURCE = (EXAMPLE / "submissions/octocat.py").read_bytes()


def test_published_example_gets_complete_milestones():
    result = feedback.run_checks(EXAMPLE, "alice", EXAMPLE_SOURCE)
    assert result.passed
    report = feedback.markdown(result)
    for group in ("Examples and edge cases", "Your predictions", "Your own tests", "Explanation completeness"):
        assert group in report
    assert "not a grade" in report
    assert "Passing checks never triggers a merge" in report


@pytest.mark.parametrize("slug", ["digit-grouping", "gpt2-pretokenizer", "han-runs"])
def test_wrong_solution_gets_counterexamples_for_each_released_task(slug):
    result = feedback.run_checks(ROOT / "tasks/l01-tokenization" / slug, "alice",
                                 b"def solve(text):\n    return []\n")
    assert not result.passed
    details = "\n".join(c.detail for c in result.checks if c.group == "Examples and edge cases")
    assert "Input:" in details and "Expected:" in details and "Actual:" in details
    report = feedback.markdown(result)
    assert "Your predictions" in report and "Your own tests" in report and "Explanation completeness" in report
    assert "same PR branch" in report
    assert any(c.status == "passed" for c in result.checks)  # e.g. the empty-string case


def test_syntax_error_is_reported_once_without_running_tests():
    result = feedback.run_checks(EXAMPLE, "alice", b"def solve(:\n    pass\n")
    assert not result.passed and len(result.checks) == 1
    assert result.checks[0].name == "Python syntax"
    assert "Line 1" in result.checks[0].detail


def test_import_error_has_actionable_feedback_and_escaped_markup():
    result = feedback.run_checks(EXAMPLE, "alice", b"raise RuntimeError('<script>broken</script>')\n")
    assert not result.passed
    report = feedback.markdown(result)
    assert "RuntimeError" in report and "&lt;script&gt;" in report
    assert "<script>" not in report


def test_infinite_loop_times_out():
    result = feedback.run_checks(EXAMPLE, "alice", b"while True:\n    pass\n", timeout=0.5)
    assert not result.passed
    assert "stopped after 0.5 seconds" in result.problems[0]


def test_early_successful_process_exit_is_not_a_passing_submission():
    result = feedback.run_checks(EXAMPLE, "alice", b"import os\nos._exit(0)\n")
    assert not result.passed
    assert any("No checks completed" in p for p in result.problems)


def test_no_tests_or_skipped_tests_are_not_passes(tmp_path):
    folder = tmp_path / "task"
    shutil.copytree(EXAMPLE, folder)
    test_file = folder / "tests/test_task.py"
    test_file.write_text("")
    empty = feedback.run_checks(folder, "alice", EXAMPLE_SOURCE)
    assert not empty.passed and any("No checks completed" in p for p in empty.problems)
    test_file.write_text("def test_data_only(): assert True\n")
    fixtures_only = feedback.run_checks(folder, "alice", EXAMPLE_SOURCE)
    assert not fixtures_only.passed and any("No checks ran against your submission" in p for p in fixtures_only.problems)
    test_file.write_text("import pytest\n@pytest.mark.skip(reason='not checked')\ndef test_cases(): pass\n")
    skipped = feedback.run_checks(folder, "alice", EXAMPLE_SOURCE)
    assert not skipped.passed and skipped.checks[0].status == "skipped"


@pytest.mark.parametrize("username", ["Alice", "../alice", "alice.py", "_alice", ""])
def test_invalid_usernames_are_not_executed(username):
    result = feedback.run_checks(EXAMPLE, username, b"raise RuntimeError('must not run')\n")
    assert not result.passed and "lowercase GitHub username" in result.problems[0]


def test_oversized_submission_is_not_executed():
    result = feedback.run_checks(EXAMPLE, "alice", b" " * (feedback.MAX_SOURCE_BYTES + 1))
    assert not result.passed and "128 KiB" in result.problems[0]


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE).strip()


@pytest.fixture(params=["l01-tokenization", "l02-ngram"])
def pull_request(tmp_path, request):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Feedback test")
    git(repo, "config", "user.email", "feedback-test@example.invalid")
    git(repo, "config", "commit.gpgsign", "false")
    task_id = f"{request.param}/demo"
    folder = repo / "tasks" / task_id
    shutil.copytree(EXAMPLE, folder)
    manifest = folder / "task.toml"
    text = manifest.read_text().replace('id = "example/word-count"', f'id = "{task_id}"')
    text = text.replace('issue = 0', 'issue = 99')
    manifest.write_text(text)
    # Another student's bad submission must not affect this student's report.
    (folder / "submissions/bob.py").write_text("raise RuntimeError('do not run another student')\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "publish example task")
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "switch", "-c", "student")
    (folder / "submissions/alice.py").write_bytes(EXAMPLE_SOURCE)
    git(repo, "add", ".")
    git(repo, "commit", "-m", "submit exercise")
    head = git(repo, "rev-parse", "HEAD")
    git(repo, "switch", "main")
    event = {"pull_request": {"title": f"{task_id}: alice", "body": "Related to #99",
                              "user": {"login": "Alice"}, "head": {"sha": head}, "base": {"sha": base}}}
    return repo, event


def test_pr_runs_only_its_own_file_against_base_tests(pull_request):
    repo, event = pull_request
    result = feedback.pr_feedback(repo, event, "student")
    assert result.passed, feedback.markdown(result)
    task_id = event["pull_request"]["title"].split(":", 1)[0]
    assert not (repo / "tasks" / task_id / "submissions/alice.py").exists()


@pytest.mark.parametrize("title,body,message", [
    ("wrong title", "Related to #99", "Use PR title"),
    (None, "", "Include Related to"),
    (None, "Related to #99\nFixes #99", "not Fixes/Closes/Resolves"),
])
def test_pr_metadata_gives_specific_fixes(pull_request, title, body, message):
    repo, event = pull_request
    event["pull_request"].update(title=title or event["pull_request"]["title"], body=body)
    result = feedback.pr_feedback(repo, event, "student")
    assert not result.passed and any(message in p for p in result.problems)


@pytest.mark.parametrize("change", ["checker", "wrong-user", "symlink", "deleted"])
def test_pr_rejects_invalid_scope_and_nonregular_files(pull_request, change):
    repo, event = pull_request
    git(repo, "switch", "student")
    folder = repo / "tasks" / event["pull_request"]["title"].split(":", 1)[0]
    submission = folder / "submissions/alice.py"
    if change == "checker":
        (folder / "tests/test_task.py").write_text("def test_always_pass(): pass\n")
    elif change == "wrong-user":
        submission.rename(folder / "submissions/charlie.py")
    elif change == "symlink":
        submission.unlink()
        submission.symlink_to("octocat.py")
    else:
        # A deletion of an existing student's file must never execute.
        submission.unlink()
        (folder / "submissions/octocat.py").unlink()
    git(repo, "add", "-A")
    git(repo, "commit", "-m", "change submission")
    event["pull_request"]["head"]["sha"] = git(repo, "rev-parse", "HEAD")
    git(repo, "switch", "main")
    result = feedback.pr_feedback(repo, event, "student")
    assert not result.passed and result.problems and not result.checks


def test_changed_head_does_not_report_on_stale_code(pull_request):
    repo, event = pull_request
    event["pull_request"]["head"]["sha"] = "0" * 40
    result = feedback.pr_feedback(repo, event, "student")
    assert not result.passed and "latest commit" in result.problems[0]


def test_survey_pr_is_outside_exercise_feedback(pull_request):
    repo, event = pull_request
    git(repo, "switch", "-c", "survey")
    response = repo / "tasks/l01-tokenization/llm-app-survey/responses/alice.md"
    response.parent.mkdir(parents=True)
    response.write_text("survey response\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "survey response")
    event["pull_request"].update(title="survey: alice", body="Related to #6")
    event["pull_request"]["head"]["sha"] = git(repo, "rev-parse", "HEAD")
    git(repo, "switch", "main")
    result = feedback.pr_feedback(repo, event, "survey")
    assert result.skipped and "No student exercise solution" in feedback.markdown(result)


@pytest.fixture(params=["dsir-data-selection", "ngram-contamination"])
def l02_checker(request):
    path = ROOT / "tasks/l02-ngram" / request.param / "tests/test_task.py"
    spec = importlib.util.spec_from_file_location(request.param.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gbk_text_default(monkeypatch):
    """Simulate a Chinese Windows locale without changing explicit encodings."""
    original_open = Path.open

    def open_with_gbk(path, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if "b" not in mode and encoding in (None, "locale"):
            encoding = "gbk"
        return original_open(path, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, "open", open_with_gbk)


@pytest.mark.parametrize("slug", ["dsir-data-selection", "ngram-contamination"])
def test_l02_fixtures_preserve_unicode_with_gbk_default(slug, gbk_text_default):
    folder = ROOT / "tasks/l02-ngram" / slug
    spec = importlib.util.spec_from_file_location("locale_checker", folder / "tests/test_task.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.CASES == json.loads((folder / "data/checks.json").read_bytes())
    assert module.PREDICTIONS == json.loads((folder / "data/examples.json").read_bytes())


@pytest.mark.parametrize("slug", ["dsir-data-selection", "ngram-contamination"])
def test_l02_reports_round_trip_utf8_with_gbk_default(slug, tmp_path, gbk_text_default):
    path = ROOT / "tasks/l02-ngram" / slug / "experiment_utils.py"
    spec = importlib.util.spec_from_file_location("locale_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = {"text": "café, 中文, 🌍"}
    output = tmp_path / "report.json"
    module.write_report(output, report)
    assert json.loads(output.read_bytes()) == report


def test_wrong_output_gets_named_cases_and_actionable_feedback(l02_checker):
    result = feedback.run_checks(l02_checker.ROOT, "alice", b"def solve(case):\n    return {}\n")
    assert not result.passed
    details = "\n".join(check.detail for check in result.checks)
    assert all(label in details for label in ("Input:", "Expected:", "Actual:", "required fields"))
    report = feedback.markdown(result)
    assert f"Case: {l02_checker.CASES[0]['name']}" in report
    assert all(group in report for group in (
        "Submission format", "Examples and edge cases", "Your predictions",
        "Your own tests", "Explanation completeness",
    ))
    assert "same PR branch" in report and "Submission deadline:" in report


def test_starter_keeps_five_helpers_unsolved_and_fails_feedback(l02_checker):
    source = (l02_checker.ROOT / "starter.py").read_bytes()
    module = ast.parse(source)
    supplied = {"bucket", "rank", "solve"}
    helpers = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name not in supplied]
    assert len(helpers) == 5
    for helper in helpers:
        assert len(helper.body) == 2 and isinstance(helper.body[1], ast.Raise)
        assert isinstance(helper.body[1].exc, ast.Name) and helper.body[1].exc.id == "NotImplementedError"
    assert b"BEGIN SOLUTION" not in source
    assert not (l02_checker.ROOT / "reference-answers.json").exists()
    assert not list(l02_checker.ROOT.rglob("hidden_tests"))
    result = feedback.run_checks(l02_checker.ROOT, "alice", source)
    assert not result.passed
    assert "NotImplementedError" in "\n".join(check.detail for check in result.checks)


def test_mismatch_identifies_the_nested_field(l02_checker):
    with pytest.raises(AssertionError, match=r"output\['items'\]\[0\]"):
        l02_checker.assert_output({"items": ["wrong"]}, {"items": ["expected"]})
    with pytest.raises(AssertionError, match="required fields"):
        l02_checker.assert_output({"items": [], "extra": []}, {"items": []})
    with pytest.raises(AssertionError, match="return a list"):
        l02_checker.assert_output({"items": ()}, {"items": []})


def test_numeric_checks_use_documented_tolerances(l02_checker):
    l02_checker.assert_output([1.0 + 5e-10, 5e-11], [1.0, 0.0])
    for bad in (float("nan"), float("inf"), True, "1.0", 1.01):
        with pytest.raises(AssertionError):
            l02_checker.assert_output([bad], [1.0])


def test_personal_cases_must_be_distinct_and_new(l02_checker, tmp_path):
    candidate = tmp_path / "alice.py"
    candidate.write_text("MY_CASES = [({'new': 1}, []), ({'new': 1}, [])]\ndef solve(case): pass\n")
    with pytest.raises(AssertionError, match="distinct inputs"):
        l02_checker.test_own_cases_are_new_and_pass(candidate)
    candidate.write_text(
        f"MY_CASES = [({l02_checker.CASES[0]['case']!r}, []), ({{'new': 1}}, [])]\n"
        "def solve(case): pass\n"
    )
    with pytest.raises(AssertionError, match="beyond the supplied cases"):
        l02_checker.test_own_cases_are_new_and_pass(candidate)


def test_personal_cases_accept_two_new_inputs(l02_checker, tmp_path):
    candidate = tmp_path / "alice.py"
    # This dummy function tests the case validator, not either task's algorithm.
    candidate.write_text(
        "MY_CASES = [({'new': 1}, []), ({'new': 2}, [])]\n"
        f"def solve(case): return {{{l02_checker.PROJECTION!r}: []}}\n"
    )
    l02_checker.test_own_cases_are_new_and_pass(candidate)


def test_missing_reflection_explains_the_requirement(l02_checker, tmp_path):
    candidate = tmp_path / "alice.py"
    candidate.write_text("NOTES = 'REPLACE'\ndef solve(case): pass\n")
    with pytest.raises(AssertionError, match="at least 200 characters"):
        l02_checker.test_notes_are_present(candidate)
