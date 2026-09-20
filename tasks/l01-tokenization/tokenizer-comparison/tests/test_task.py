"""Re-encode submitted measurements; qualitative reasoning is peer-reviewed."""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts import tokenizer_comparison as comparison

FOLDER = Path(__file__).resolve().parents[1]
SUBMISSIONS = sorted(path for path in (FOLDER / "submissions").glob("*.json")
                     if not path.name.startswith((".", "_")))


@pytest.mark.parametrize("path", SUBMISSIONS, ids=lambda path: f"user.{path.stem}")
def test_contribution_matches_pinned_tokenizers(path):
    assert re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,37}[a-z0-9])?", path.stem), "Use a lowercase GitHub username"
    record = comparison.read_json(path)
    assert record["contributors"][0] == path.stem, "The filename must match the first contributor"
    comparison.check_contribution(record, comparison.manifest(), comparison.DEFAULT_CACHE)
