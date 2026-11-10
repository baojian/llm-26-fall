"""Run the taught post-training calculations offline, in slide exercise order."""
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("number,expected", [("08", ["E01", "E02", "E03", "E04"]),
                                            ("09", ["E01", "E02", "E03", "E04", "P01", "P02"])])
def test_offline_calculations_and_exercise_alignment(number, expected, tmp_path, monkeypatch):
    deck = ROOT / "slides" / f"lecture-{number}"
    nb = json.loads((deck / "practice.ipynb").read_text())
    exercises = [match[1] for cell in nb["cells"] if cell["cell_type"] == "markdown"
                 if (match := re.search(r"^## ([EP]\d+)\b", "".join(cell["source"])))]
    assert exercises == expected
    slide_ids = re.findall(r"(?:Exercise|Practice) ([EP]\d+) ·", (deck / "slides.md").read_text())
    assert slide_ids == expected
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"lecture-{number}-cell-{index}", "exec"), namespace)
    if number == "09":
        # Independent directional derivative checks for the PPO teaching function.
        objective = namespace["clipped_objective"]
        h = 1e-5
        for ratio, advantage, slope in [(1.4, 2, 0), (.6, -2, 0), (.6, 2, 2), (1.4, -2, -2)]:
            actual = (objective(ratio+h, advantage) - objective(ratio-h, advantage)) / (2*h)
            assert actual == pytest.approx(slope, abs=1e-8)
