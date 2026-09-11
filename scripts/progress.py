"""Build the student progress board from merged task submissions.

Runs every task's checker, reads the survey responses, optionally counts
merged pull requests through ``gh``, and writes ``tasks/PROGRESS.md`` and
``tasks/progress.svg``. Usernames listed in ``tasks/.progress-optout`` are left
off the board.

    uv run python scripts/progress.py             # with merged-PR counts from GitHub
    uv run python scripts/progress.py --no-github  # offline
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import tasks as task_tools  # noqa: E402

TASKS = ROOT / "tasks"
SURVEY_RESPONSES = ROOT / "surveys/lecture-01/responses"
REPO = "baojian/llm-26-fall"
BADGES = [
    ("🥇", "first merged PR", lambda s: (s.merged_prs or 0) >= 1 or s.submitted >= 1 or s.survey),
    ("🔥", "5 tasks passed", lambda s: s.passed >= 5),
    ("🏆", "10 tasks passed", lambda s: s.passed >= 10),
    ("🎯", "all correct (3+ tasks)", lambda s: s.submitted >= 3 and s.passed == s.submitted),
    ("💡", "proposal accepted", lambda s: s.proposals >= 1),
]


@dataclass
class Student:
    username: str
    submitted: int = 0
    passed: int = 0
    lectures: set = field(default_factory=set)
    survey: bool = False
    proposals: int = 0
    merged_prs: int | None = None

    @property
    def correct_rate(self) -> float | None:
        return self.passed / self.submitted if self.submitted else None

    @property
    def participation(self) -> int:
        return len(self.lectures) + (1 if self.survey else 0)

    def badges(self) -> str:
        return " ".join(icon for icon, _, earned in BADGES if earned(self))


def checker_results(folder: Path) -> dict[str, bool]:
    """Run one task's checker; return {username: all tests passed}."""
    usernames = task_tools.submissions(folder)
    if not usernames:
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.xml"
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(folder / "tests"),
                        f"--junitxml={report}"], cwd=ROOT, capture_output=True, text=True)
        outcomes: dict[str, bool] = {user: True for user in usernames}
        if not report.exists():
            return {user: False for user in usernames}
        for case in ET.parse(report).getroot().iter("testcase"):
            name = case.get("name", "")
            if "[" not in name or "@" not in name:
                continue
            user = name[name.rindex("@") + 1:].rstrip("]")
            if user in outcomes and any(child.tag in {"failure", "error"} for child in case):
                outcomes[user] = False
    return outcomes


def merged_pr_counts(repo: str = REPO) -> dict[str, int] | None:
    """Merged PRs per author (lowercase login) via gh; None when gh is unavailable."""
    try:
        output = subprocess.run(["gh", "pr", "list", "--repo", repo, "--state", "merged", "--limit", "1000",
                                 "--json", "author"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    counts: dict[str, int] = defaultdict(int)
    for pr in json.loads(output):
        counts[pr["author"]["login"].lower()] += 1
    return dict(counts)


def collect(tasks_root: Path = TASKS, survey_dir: Path = SURVEY_RESPONSES,
            prs: dict[str, int] | None = None) -> list[Student]:
    students: dict[str, Student] = {}

    def get(user: str) -> Student:
        return students.setdefault(user, Student(user))

    for folder in task_tools.task_folders(tasks_root):
        info = task_tools.manifest(folder)["task"]
        for user, ok in checker_results(folder).items():
            student = get(user)
            student.submitted += 1
            student.passed += ok
            student.lectures.add(info["lecture"])
        for user in info.get("proposed_by", "").split(","):
            if user.strip():
                get(user.strip().lower()).proposals += 1
    if survey_dir.exists():
        for path in survey_dir.glob("*.md"):
            get(path.stem.lower()).survey = True
    if prs is not None:
        for user, count in prs.items():
            if user in students:
                students[user].merged_prs = count
        for student in students.values():
            student.merged_prs = student.merged_prs or 0
    optout = tasks_root / ".progress-optout"
    hidden = {word for line in optout.read_text().splitlines() if not line.startswith("#") for word in line.split()} if optout.exists() else set()
    board = [s for s in students.values() if s.username not in hidden]
    return sorted(board, key=lambda s: (-s.passed, -(s.correct_rate or 0), -s.participation, s.username))


def board_markdown(students: list[Student], today: dt.date, github: bool) -> str:
    rows = ["| # | Student | Tasks passed | Correct rate | Participation | Merged PRs | Badges |",
            "| ---: | :--- | ---: | ---: | ---: | ---: | :--- |"]
    for rank, s in enumerate(students, 1):
        rate = "—" if s.correct_rate is None else f"{s.correct_rate:.0%}"
        prs = "—" if s.merged_prs is None else str(s.merged_prs)
        link = f"[{s.username}](https://github.com/{REPO}/pulls?q=is%3Apr+is%3Amerged+author%3A{s.username})"
        rows.append(f"| {rank} | {link} | {s.passed} | {rate} | {s.participation} | {prs} | {s.badges()} |")
    legend = " · ".join(f"{icon} {text}" for icon, text, _ in BADGES)
    pr_note = "" if github else " Merged-PR counts were not fetched in this run."
    return f"""# Progress board

Generated on {today:%B %-d, %Y} from the merged files in this repository.{pr_note}
Click a username to see that student's merged pull requests.

![Bar chart of tasks passed per student](progress.svg)

{chr(10).join(rows)}

**Columns.** *Tasks passed*: submissions whose checker passes. *Correct rate*:
passed ÷ submitted. *Participation*: lectures with at least one submission,
plus one for the Lecture 01 survey. *Merged PRs*: all pull requests merged
into this repository, including fixes and proposals.

**Badges.** {legend}.

To be left off this board, add your username to `tasks/.progress-optout`.
Regenerate with `uv run python scripts/progress.py`.
"""


def board_svg(students: list[Student], today: dt.date, limit: int = 25) -> str:
    shown = [s for s in students if s.passed or s.submitted][:limit] or students[:limit]
    left, top, width, row, bar = 220, 84, 960, 34, 22
    plot_width = width - left - 80
    height = top + row * max(len(shown), 1) + 60
    largest = max((s.passed for s in shown), default=0) or 1
    ink, muted, accent, grid, paper = "#202b38", "#566471", "#20578c", "#dce2e7", "#fbfbf9"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             f'role="img" aria-labelledby="t" font-family="Arial, Helvetica Neue, PingFang SC, sans-serif">',
             '<title id="t">Tasks passed per student</title>',
             f'<rect width="{width}" height="{height}" fill="{paper}"/>',
             f'<text x="32" y="40" font-size="24" font-weight="700" fill="{ink}">Tasks passed per student</text>',
             f'<text x="32" y="64" font-size="15" fill="{muted}">{escape(f"{len(students)} students on the board · generated {today:%B %-d, %Y}")}</text>']
    for tick in range(0, largest + 1, max(1, largest // 8)):
        x = left + plot_width * tick / largest
        parts.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{top + row * len(shown)}" stroke="{grid}"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + row * len(shown) + 20}" font-size="13" fill="{muted}" text-anchor="middle">{tick}</text>')
    for index, s in enumerate(shown):
        y = top + row * index
        w = plot_width * s.passed / largest
        parts.append(f'<text x="{left - 12}" y="{y + bar / 2 + 5}" font-size="15" fill="{ink}" text-anchor="end">{escape(s.username)}</text>')
        if s.passed:
            parts.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar}" rx="4" fill="{accent}"/>')
        parts.append(f'<text x="{left + w + 8:.1f}" y="{y + bar / 2 + 5}" font-size="14" fill="{ink}">{s.passed}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-github", action="store_true", help="skip the merged-PR count")
    parser.add_argument("--date", type=dt.date.fromisoformat, default=dt.date.today())
    args = parser.parse_args()
    prs = None if args.no_github else merged_pr_counts()
    students = collect(prs=prs)
    (TASKS / "PROGRESS.md").write_text(board_markdown(students, args.date, prs is not None), encoding="utf-8")
    (TASKS / "progress.svg").write_text(board_svg(students, args.date), encoding="utf-8")
    for s in students[:10]:
        print(f"{s.passed:3d} passed / {s.submitted} submitted  {s.username}  {s.badges()}")
    print(f"{len(students)} students -> {TASKS / 'PROGRESS.md'}")


if __name__ == "__main__":
    main()
