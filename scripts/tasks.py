"""Create, check, and tally course tasks under tasks/.

    uv run python scripts/tasks.py new l02-ngram split-sentences "Split text into sentences"
    uv run python scripts/tasks.py check tasks/l02-ngram/split-sentences [username]
    uv run python scripts/tasks.py list
    uv run python scripts/tasks.py progress
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
SKIP = {"TEMPLATE"}


def create_task(lecture: str, slug: str, title: str, tasks_root: Path = TASKS) -> Path:
    """Copy TEMPLATE/ to tasks/<lecture>/<slug>/ and fill in the placeholders."""
    for name, value in (("lecture", lecture), ("slug", slug)):
        if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", value):
            raise ValueError(f"{name} must be lowercase words joined by hyphens, got {value!r}")
    destination = tasks_root / lecture / slug
    if destination.exists():
        raise FileExistsError(f"{destination} already exists")
    shutil.copytree(tasks_root / "TEMPLATE", destination)
    for path in destination.rglob("*"):
        if path.is_file() and path.suffix in {".toml", ".md", ".py"}:
            text = path.read_text(encoding="utf-8")
            text = text.replace("{{LECTURE}}", lecture).replace("{{SLUG}}", slug).replace("{{TITLE}}", title)
            path.write_text(text, encoding="utf-8")
    return destination


def task_folders(tasks_root: Path = TASKS) -> list[Path]:
    return sorted(p.parent for p in tasks_root.glob("*/*/task.toml") if p.parent.parent.name not in SKIP)


def manifest(folder: Path) -> dict:
    return tomllib.loads((folder / "task.toml").read_text(encoding="utf-8"))


def submissions(folder: Path) -> list[str]:
    return sorted(p.stem for p in (folder / "submissions").glob("*.py") if not p.name.startswith((".", "_")))


def check(folder: Path, username: str | None = None) -> int:
    """Run the task's checker with pytest; returns the exit code."""
    command = [sys.executable, "-m", "pytest", "-q", str(folder / "tests")]
    if username:
        # Test ids end with user.<username>], so alice does not select alice2.
        command += ["-k", f"user.{username}]"]
    return subprocess.call(command, cwd=ROOT)


def list_tasks(tasks_root: Path = TASKS) -> str:
    rows = ["| Task | Title | Difficulty | Deadline | Submissions |", "| :--- | :--- | :--- | :--- | ---: |"]
    for folder in task_folders(tasks_root):
        info = manifest(folder)["task"]
        rows.append(f"| {info['id']} | {info['title']} | {info['difficulty']} | {str(info['deadline'])[:10]} | {len(submissions(folder))} |")
    return "\n".join(rows)


def progress(tasks_root: Path = TASKS) -> str:
    counts: Counter = Counter()
    for folder in task_folders(tasks_root):
        counts.update(submissions(folder))
    rows = ["| Student | Tasks submitted |", "| :--- | ---: |"]
    rows += [f"| {user} | {count} |" for user, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    new = sub.add_parser("new", help="Create a task folder from TEMPLATE/")
    new.add_argument("lecture", help="lecture label, e.g. l02-ngram")
    new.add_argument("slug", help="short task name, e.g. split-sentences")
    new.add_argument("title")
    run = sub.add_parser("check", help="Run a task's checker on every submission, or on one username")
    run.add_argument("task", type=Path)
    run.add_argument("username", nargs="?")
    sub.add_parser("list", help="Table of tasks with deadlines and submission counts")
    sub.add_parser("progress", help="Table of submissions per student")
    args = parser.parse_args()

    if args.command == "new":
        print(create_task(args.lecture, args.slug, args.title))
    elif args.command == "check":
        sys.exit(check(args.task.resolve(), args.username))
    elif args.command == "list":
        print(list_tasks())
    elif args.command == "progress":
        print(progress())


if __name__ == "__main__":
    main()
