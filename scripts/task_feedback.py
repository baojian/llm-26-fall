"""Give students actionable feedback from a task's existing public pytest checks.

Local: uv run python scripts/task_feedback.py tasks/l01-tokenization/digit-grouping octocat
CI:    python scripts/task_feedback.py --event "$GITHUB_EVENT_PATH"
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_SOURCE_BYTES = 128 * 1024
GROUPS = {
    "test_filename": "Submission format",
    "test_cases": "Examples and edge cases",
    "test_predictions": "Your predictions",
    "test_own_cases": "Your own tests",
    "test_notes": "Explanation completeness",
    "test_raw_text": "Supplied data",
}
HINTS = {
    "Submission format": "Use your lowercase GitHub username as the filename.",
    "Examples and edge cases": "Try the first failing input locally. Compare the expected and actual outputs, including whitespace and Unicode characters.",
    "Your predictions": "Include exactly the three requested inputs, then investigate each disagreement between your prediction and solve.",
    "Your own tests": "Use distinct new inputs, not copies of the supplied cases, and check their expected outputs by hand.",
    "Explanation completeness": "Write your explanation in NOTES using the handout's length and content requirements. A human will review the reasoning.",
    "Supplied data": "This check concerns the supplied task data. Report it to the teaching team; keep the supplied tests and data unchanged.",
    "Other checks": "Read the error below, fix your submission, and run the same checker again.",
}


@dataclass
class Check:
    name: str
    group: str
    status: str
    detail: str = ""


@dataclass
class Feedback:
    task: str
    username: str
    deadline: str = "See the task instructions"
    checks: list[Check] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    skipped: bool = False

    @property
    def passed(self) -> bool:
        return bool(self.checks) and not self.problems and all(c.status == "passed" for c in self.checks)


def check_group(name: str) -> str:
    return next((label for prefix, label in GROUPS.items() if name.startswith(prefix)), "Other checks")


def read_results(path: Path) -> list[Check]:
    """Read pytest's standard report, including collection errors and skips."""
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("The test report was too large. Remove excessive output and try again.")
    checks = []
    for case in ET.parse(path).getroot().iter("testcase"):
        name = case.get("name", "unnamed check")
        problem = next((child for child in case if child.tag in {"failure", "error", "skipped"}), None)
        status = "passed" if problem is None else ("skipped" if problem.tag == "skipped" else "failed")
        message = "" if problem is None else problem.get("message", "")
        detail = "" if problem is None else (message if message.startswith("AssertionError") else problem.text or message)
        checks.append(Check(name, check_group(name), status, detail[:6000]))
    return checks


def run_checks(folder: Path, username: str, source: bytes, timeout: float = 30) -> Feedback:
    """Run only this student's file against a fresh copy of the public checker."""
    info = tomllib.loads((folder / "task.toml").read_text())["task"]
    result = Feedback(info["id"], username, str(info["deadline"]))
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,37}[a-z0-9])?", username):
        result.problems.append("Use your lowercase GitHub username, without .py.")
        return result
    if len(source) > MAX_SOURCE_BYTES:
        result.problems.append("The submission exceeds 128 KiB. Submit only your small Python solution file.")
        return result
    try:
        ast.parse(source, filename=f"{username}.py")
    except SyntaxError as error:
        result.checks.append(Check("Python syntax", "Submission format", "failed",
                                   f"Line {error.lineno}: {error.msg}\n{error.text or ''}"))
        return result
    with tempfile.TemporaryDirectory(prefix="task-feedback-") as tmp:
        work = Path(tmp)
        task = work / "task"
        shutil.copytree(folder, task, ignore=shutil.ignore_patterns("submissions", "__pycache__", ".pytest_cache"))
        (task / "submissions").mkdir()
        (task / "submissions" / f"{username}.py").write_bytes(source)
        config = work / "pytest.ini"
        config.write_text("[pytest]\n")
        report = work / "report.xml"
        log = work / "pytest.log"
        command = [sys.executable, "-I", "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider",
                   "-c", str(config), "--confcutdir", str(work), f"--junitxml={report}", str(task / "tests")]
        # The child does not need GitHub credentials or workflow-command paths.
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(work), "LANG": "C.UTF-8",
               "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
        with log.open("w") as output:
            try:
                completed = subprocess.run(command, cwd=work, env=env, stdout=output,
                                           stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired:
                result.problems.append(f"Checks stopped after {timeout:g} seconds. Look for an infinite loop or very slow code, including code executed on import.")
                return result
        if report.exists():
            try:
                result.checks = read_results(report)
            except (ET.ParseError, ValueError) as error:
                result.problems.append(f"Could not read the test report: {error}")
        if not result.checks:
            result.problems.append("No checks completed. This is not a passing result. Check syntax, imports, and the required solve function.")
        elif not any(f"user.{username}]" in c.name for c in result.checks):
            result.problems.append("No checks ran against your submission. Ask the teaching team to check the task's test collection.")
        if completed.returncode and not any(c.status == "failed" for c in result.checks):
            result.problems.append(f"The checker exited with status {completed.returncode}.")
        if result.problems:
            with log.open() as output:
                result.checks.append(Check("Checker output", "Other checks", "failed", output.read(6000)))
    return result


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def pr_feedback(root: Path, event: dict, head_ref: str = "refs/remotes/origin/task-feedback") -> Feedback:
    """Use base-branch tests; copy only the reviewed PR's one allowed blob."""
    pr = event["pull_request"]
    username = pr["user"]["login"].lower()
    head = git(root, "rev-parse", head_ref).decode().strip()
    if head != pr["head"]["sha"]:
        return Feedback("Task submission", username, problems=["The PR changed while this run started. Use the feedback run for the latest commit."])
    changed = git(root, "diff", "--name-only", "--no-renames", "-z", f"{pr['base']['sha']}...{head}")
    paths = [p.decode() for p in changed.split(b"\0") if p]
    candidates = [p for p in paths if re.match(r"tasks/[^/]+/[^/]+/submissions/", p)
                  and not p.endswith("/README.md")]
    title_task = re.match(r"(l\d{2}-[a-z0-9-]+/[a-z0-9-]+):", pr["title"])
    if not candidates and not title_task:
        return Feedback("No exercise solution in this PR", username, skipped=True)
    task_id = title_task[1] if title_task else "/".join(candidates[0].split("/")[1:3])
    result = Feedback(task_id, username)
    folder = root / "tasks" / task_id
    if not (folder / "task.toml").is_file():
        result.problems.append("This task is not published on the base branch. Check the task name and submission folder.")
        return result
    info = tomllib.loads((folder / "task.toml").read_text())["task"]
    result.deadline = str(info["deadline"])
    expected = f"tasks/{task_id}/submissions/{username}.py"
    if paths != [expected]:
        result.problems.append(f"Change only {expected}. Keep other students' files, supplied tests, data, and course files unchanged.")
    if pr["title"] != f"{task_id}: {username}":
        result.problems.append(f"Use PR title: {task_id}: {username}")
    issue = info.get("issue")
    if issue:
        body = pr.get("body") or ""
        reference = rf"(?:#{issue}\b|https://github\.com/baojian/llm-26-fall/issues/{issue}\b)"
        if not re.search(rf"Related to\s+{reference}", body, re.I):
            result.problems.append(f"Include Related to #{issue} in the PR description.")
        if re.search(rf"(?:fix(?:es|ed)?|close[sd]?|resolve[sd]?)\s+{reference}", body, re.I):
            result.problems.append(f"Use Related to #{issue}, not Fixes/Closes/Resolves, so the shared task issue stays open.")
    if result.problems:
        return result
    entry = git(root, "ls-tree", head, "--", expected).decode()
    if not entry.startswith(("100644 blob ", "100755 blob ")):
        result.problems.append("Submit a regular Python file, not a deleted file, symlink, or submodule.")
        return result
    source = git(root, "show", f"{head}:{expected}")
    return run_checks(folder, username, source)


def markdown(result: Feedback) -> str:
    if result.skipped:
        return "## Task feedback\n\nNo student exercise solution to check in this PR.\n"
    lines = [f"## Task feedback: {html.escape(result.task)}", "",
             "**All public checks passed — ready for human review.**" if result.passed else
             "**Another attempt is needed. Start with the feedback below.**", ""]
    for problem in result.problems:
        lines += [f"- {html.escape(problem)}"]
    if result.problems:
        lines.append("")
    groups = defaultdict(list)
    for group in [*GROUPS.values(), "Other checks"]:
        for check in result.checks:
            if check.group == group:
                groups[group].append(check)
    if groups:
        lines += ["| Milestone | Passed | Status |", "| :--- | ---: | :--- |"]
        for group, checks in groups.items():
            count = sum(c.status == "passed" for c in checks)
            lines.append(f"| {group} | {count}/{len(checks)} | {'✅ Complete' if count == len(checks) else '🔧 Try again'} |")
        lines.append("")
    for group, checks in groups.items():
        failures = [c for c in checks if c.status != "passed"]
        if not failures:
            continue
        lines += [f"### {group}", "", HINTS[group], ""]
        for check in failures:
            case = re.fullmatch(r"test_cases\[(.*)-expected\d+-user\.[^]]+\]", check.name)
            label = f"Input: {case[1]}" if case else check.name.split("[", 1)[0].removeprefix("test_").replace("_", " ")
            lines += [f"<details><summary>{html.escape(label[:250])}</summary>", "",
                      f"<pre>{html.escape(check.detail)}</pre>", "", "</details>", ""]
    lines += ["### Your next step", ""]
    if result.passed:
        lines += ["Your code passed the published checks. The teaching team still reviews your reasoning and additional cases."]
    else:
        lines += ["Fix your own submission, run the checker locally, and push to the **same PR branch** for another attempt. Keep supplied tests and data unchanged."]
    lines += ["", "```sh", f"uv run python scripts/task_feedback.py tasks/{result.task} {result.username}", "```", "",
              "These milestones are practice feedback, not a grade. The explanation check verifies completeness, not correctness of your reasoning.", "",
              f"**Submission deadline:** {html.escape(result.deadline)}. **Merge hold:** solutions merge only after the deadline, in one batch per task. Passing checks never triggers a merge.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path, nargs="?")
    parser.add_argument("username", nargs="?")
    parser.add_argument("--event", type=Path, help="GitHub pull_request event JSON; the base branch must be checked out")
    parser.add_argument("--summary", type=Path, help="write a Markdown report, e.g. workspace/feedback.md")
    args = parser.parse_args()
    try:
        if args.event:
            result = pr_feedback(ROOT, json.loads(args.event.read_text()))
        else:
            if not args.task or not args.username:
                parser.error("provide a task folder and lowercase username, or --event")
            result = run_checks(args.task.resolve(), args.username,
                                (args.task / "submissions" / f"{args.username}.py").read_bytes())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        result = Feedback("Task submission", args.username or "student", problems=[f"Could not start the checker: {error}"])
    report = markdown(result)
    summary = args.summary or (Path(os.environ["GITHUB_STEP_SUMMARY"]) if os.environ.get("GITHUB_STEP_SUMMARY") else None)
    if summary:
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(report, encoding="utf-8")
        print("Task feedback written to the report. " + ("Passed." if result.passed else "Not applicable." if result.skipped else "Needs another attempt."))
    else:
        print(report)
    sys.exit(0 if result.passed or result.skipped else 1)


if __name__ == "__main__":
    main()
