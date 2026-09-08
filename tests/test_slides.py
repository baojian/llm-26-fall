"""Check lecture creation and the executable examples used in the sample deck."""

import json
import shutil
from pathlib import Path

import pytest

from scripts.slides import create_lecture


SLIDES = Path(__file__).resolve().parents[1] / "slides"


@pytest.fixture
def slides_root(tmp_path):
    shutil.copytree(SLIDES / "template", tmp_path / "template")
    return tmp_path


def test_new_lecture_has_consistent_metadata_and_notebook(slides_root):
    destination = create_lecture(1, "Tokenization & Unicode", slides_root)
    assert destination.name == "01-tokenization-unicode"
    metadata = json.loads((destination / "lecture.json").read_text())
    notebook = json.loads((destination / "practice.ipynb").read_text())
    assert metadata["title"] == "Tokenization & Unicode"
    assert metadata["number"] == "01"
    assert "Lecture 01: Tokenization & Unicode" in "".join(notebook["cells"][0]["source"])
    assert "{{TITLE}}" not in (destination / "slides.md").read_text()
    assert (destination / "index.html").read_bytes() == (SLIDES / "template/index.html").read_bytes()


def test_existing_work_is_never_overwritten(slides_root):
    destination = create_lecture(1, "Tokenization", slides_root)
    (destination / "slides.md").write_text("Instructor's revised lecture")
    with pytest.raises(FileExistsError):
        create_lecture(1, "Tokenization", slides_root)
    assert (destination / "slides.md").read_text() == "Instructor's revised lecture"


@pytest.mark.parametrize("number,title", [(0, "Title"), (100, "Title"), (1, " "), (1, "A\nB"), (1, "../..")])
def test_invalid_input_creates_no_lecture(slides_root, number, title):
    with pytest.raises(ValueError):
        create_lecture(number, title, slides_root)
    assert [entry.name for entry in slides_root.iterdir()] == ["template"]


def test_sample_notebook_runs_without_external_data(capsys):
    notebook = json.loads((SLIDES / "example/practice.ipynb").read_text())
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "practice.ipynb", "exec"), namespace)
    output = capsys.readouterr().out
    assert "hello 5 5" in output
    assert "你好 2 6" in output
    assert "🙂 1 4" in output
    figure = json.loads((SLIDES / "example/assets/sequence-length.json").read_text())
    assert figure["data"][0]["y"] == namespace["totals"]
