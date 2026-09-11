"""Check the progress board builder on a temporary tasks tree and on the committed board."""

import datetime as dt
import shutil
from pathlib import Path

from scripts import progress, tasks

ROOT = Path(__file__).resolve().parents[1]


def make_tree(tmp_path):
    tasks_root = tmp_path / "tasks"
    shutil.copytree(ROOT / "tasks/TEMPLATE", tasks_root / "TEMPLATE")
    shutil.copytree(ROOT / "tasks/example", tasks_root / "example")
    demo = tasks.create_task("l02-ngram", "demo", "Demo", tasks_root)
    (demo / "task.toml").write_text((demo / "task.toml").read_text().replace('proposed_by = ""', 'proposed_by = "alice"'))
    (demo / "submissions/alice.py").write_text("def solve(text):\n    return ['REPLACE'] if text else []\n")
    (demo / "submissions/bob-x.py").write_text("def solve(text):\n    return []\n")
    (demo / "submissions/carol.py").write_text("def solve(text):\n    return ['REPLACE'] if text else []\n")
    survey = tmp_path / "responses"
    survey.mkdir()
    (survey / "Alice.md").write_text("GitHub username: Alice\n")
    (survey / "dave.md").write_text("GitHub username: dave\n")
    return tasks_root, survey


def test_collect_counts_passes_participation_and_badges(tmp_path):
    tasks_root, survey = make_tree(tmp_path)
    (tasks_root / ".progress-optout").write_text("# hidden\ncarol\n")
    board = progress.collect(tasks_root, survey, prs={"alice": 3, "bob-x": 1, "octocat": 0})
    by_name = {s.username: s for s in board}
    assert [s.username for s in board] == ["alice", "octocat", "bob-x", "dave"]
    assert "carol" not in by_name
    alice = by_name["alice"]
    assert (alice.submitted, alice.passed, alice.correct_rate, alice.participation) == (1, 1, 1.0, 2)
    assert alice.proposals == 1 and alice.merged_prs == 3 and "💡" in alice.badges()
    bob = by_name["bob-x"]
    assert (bob.submitted, bob.passed, bob.correct_rate) == (1, 0, 0.0)
    assert by_name["octocat"].passed == 1 and by_name["octocat"].lectures == {"example"}
    assert by_name["dave"].submitted == 0 and by_name["dave"].survey and by_name["dave"].merged_prs == 0


def test_markdown_and_svg_render(tmp_path):
    tasks_root, survey = make_tree(tmp_path)
    board = progress.collect(tasks_root, survey, prs=None)
    text = progress.board_markdown(board, dt.date(2026, 9, 16), github=False)
    assert "| 1 | [alice](https://github.com/baojian/llm-26-fall/pulls?q=is%3Apr+is%3Amerged+author%3Aalice) | 1 | 100% | 2 | — |" in text
    assert "Merged-PR counts were not fetched" in text
    svg = progress.board_svg(board, dt.date(2026, 9, 16))
    assert svg.startswith("<svg") and svg.count("<rect") == 1 + sum(1 for s in board if s.passed)


def test_committed_board_lists_every_survey_and_task_participant():
    text = (ROOT / "tasks/PROGRESS.md").read_text(encoding="utf-8")
    usernames = {p.stem.lower() for p in (ROOT / "surveys/lecture-01/responses").glob("*.md")}
    for folder in tasks.task_folders():
        usernames.update(tasks.submissions(folder))
    optout = ROOT / "tasks/.progress-optout"
    usernames -= {w for line in optout.read_text().splitlines() if not line.startswith("#") for w in line.split()}
    for user in usernames:
        assert f"[{user}](" in text, f"{user} is missing from tasks/PROGRESS.md; rerun scripts/progress.py"
    assert (ROOT / "tasks/progress.svg").read_text().startswith("<svg")
