"""Exercise student feedback end to end, including failed and unsafe PR shapes."""

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


@pytest.fixture
def pull_request(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Feedback test")
    git(repo, "config", "user.email", "feedback-test@example.invalid")
    git(repo, "config", "commit.gpgsign", "false")
    folder = repo / "tasks/l01-tokenization/demo"
    shutil.copytree(EXAMPLE, folder)
    manifest = folder / "task.toml"
    text = manifest.read_text().replace('id = "example/word-count"', 'id = "l01-tokenization/demo"')
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
    event = {"pull_request": {"title": "l01-tokenization/demo: alice", "body": "Related to #99",
                              "user": {"login": "Alice"}, "head": {"sha": head}, "base": {"sha": base}}}
    return repo, event


def test_pr_runs_only_its_own_file_against_base_tests(pull_request):
    repo, event = pull_request
    result = feedback.pr_feedback(repo, event, "student")
    assert result.passed, feedback.markdown(result)
    assert not (repo / "tasks/l01-tokenization/demo/submissions/alice.py").exists()


@pytest.mark.parametrize("title,body,message", [
    ("wrong title", "Related to #99", "Use PR title"),
    ("l01-tokenization/demo: alice", "", "Include Related to"),
    ("l01-tokenization/demo: alice", "Related to #99\nFixes #99", "not Fixes/Closes/Resolves"),
])
def test_pr_metadata_gives_specific_fixes(pull_request, title, body, message):
    repo, event = pull_request
    event["pull_request"].update(title=title, body=body)
    result = feedback.pr_feedback(repo, event, "student")
    assert not result.passed and any(message in p for p in result.problems)


@pytest.mark.parametrize("change", ["checker", "wrong-user", "symlink", "deleted"])
def test_pr_rejects_invalid_scope_and_nonregular_files(pull_request, change):
    repo, event = pull_request
    git(repo, "switch", "student")
    folder = repo / "tasks/l01-tokenization/demo"
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
