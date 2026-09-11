"""Check the task template, the example task, and scripts/tasks.py."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import tasks

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["task.toml", "instruction.md", "tests/test_task.py", "submissions/README.md"]


@pytest.fixture
def tasks_root(tmp_path):
    shutil.copytree(ROOT / "tasks/TEMPLATE", tmp_path / "TEMPLATE")
    return tmp_path


def test_every_task_folder_has_the_required_files():
    folders = tasks.task_folders()
    assert folders, "no tasks found"
    for folder in folders + [ROOT / "tasks/TEMPLATE"]:
        for name in REQUIRED:
            assert (folder / name).exists(), f"{folder.name} is missing {name}"
        info = tasks.manifest(folder)
        assert set(info) == {"task", "submission"}
        assert info["task"]["difficulty"] in {"easy", "medium", "hard"}


def test_example_submission_passes_its_checker():
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "tasks/example/word-count/tests", "-k", "octocat"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert " passed" in result.stdout and "failed" not in result.stdout


def test_new_task_fills_placeholders_and_checks_a_submission(tasks_root):
    folder = tasks.create_task("l02-ngram", "split-sentences", "Split text into sentences", tasks_root)
    assert folder == tasks_root / "l02-ngram/split-sentences"
    text = "".join(p.read_text() for p in folder.rglob("*") if p.is_file())
    assert "{{" not in text
    info = tasks.manifest(folder)["task"]
    assert info == {**info, "id": "l02-ngram/split-sentences", "lecture": "l02-ngram", "title": "Split text into sentences"}
    assert "l02-ngram/split-sentences: <username>" in (folder / "instruction.md").read_text()
    with pytest.raises(FileExistsError):
        tasks.create_task("l02-ngram", "split-sentences", "again", tasks_root)
    with pytest.raises(ValueError):
        tasks.create_task("L02 ngram", "x", "bad label", tasks_root)


def test_template_checker_rejects_uppercase_filenames_and_wrong_answers(tasks_root):
    folder = tasks.create_task("l02-ngram", "demo", "Demo", tasks_root)
    (folder / "submissions/Octocat.py").write_text("def solve(text):\n    return ['REPLACE'] if text else []\n")
    (folder / "submissions/wrong.py").write_text("def solve(text):\n    return []\n")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", str(folder / "tests")], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "test_filename_is_a_lowercase_username[@Octocat]" in result.stdout
    assert "test_cases[REPLACE-expected0-@wrong]" in result.stdout
    assert "2 failed, 4 passed" in result.stdout


def test_list_and_progress_tables(tasks_root):
    folder = tasks.create_task("l02-ngram", "demo", "Demo", tasks_root)
    (folder / "submissions/alice.py").write_text("def solve(text):\n    return []\n")
    (folder / "submissions/bob.py").write_text("def solve(text):\n    return []\n")
    assert "| l02-ngram/demo | Demo | easy | 2026-09-22 | 2 |" in tasks.list_tasks(tasks_root)
    assert tasks.progress(tasks_root).splitlines()[2:] == ["| alice | 1 |", "| bob | 1 |"]
