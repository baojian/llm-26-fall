"""Check the survey tally and keep the committed results in sync with the responses."""

import re
from pathlib import Path

from scripts import survey_results as survey


def write(folder, name, checked):
    lines = ["# Lecture 01 LLM app survey", "", f"GitHub username: {name}", ""]
    lines += [f"- [{'x' if app in checked else ' '}] {app}" for app in survey.LISTED_APPS]
    other = checked.get("Other") if isinstance(checked, dict) else None
    lines.append(f"- [{'x' if other else ' '}] Other: {other or 'APP_NAME'}")
    (folder / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_tally_counts_checked_boxes_and_resolves_other(tmp_path):
    write(tmp_path, "a", {"ChatGPT", "Kimi"})
    write(tmp_path, "b", {"ChatGPT"})
    write(tmp_path, "c", {"Other": "Minimax"})
    write(tmp_path, "d", set())
    (tmp_path / "e.md").write_text("GitHub username: e\n- [X] Gemini\n", encoding="utf-8")
    counts, responses = survey.tally(tmp_path)
    assert responses == 5
    assert counts["ChatGPT"] == 2 and counts["Kimi"] == 1 and counts["Gemini"] == 1
    assert counts["Minimax (Other)"] == 1 and counts["Claude"] == 0
    assert sum(counts.values()) == 5


def test_responses_with_more_than_two_selections_are_counted(tmp_path):
    write(tmp_path, "a", {"ChatGPT", "Kimi", "Claude"})
    write(tmp_path, "b", {"DeepSeek"})
    write(tmp_path, "c", set(survey.LISTED_APPS))
    counts, responses = survey.tally(tmp_path)
    assert responses == 3
    assert counts["DeepSeek"] == counts["ChatGPT"] == counts["Kimi"] == counts["Claude"] == 2
    assert sum(counts.values()) == 4 + len(survey.LISTED_APPS)


def test_bars_are_sorted_by_count_then_listing_order():
    bars = survey.ordered(survey.tally(survey.SURVEY / "responses")[0])
    counts = [count for _, count in bars]
    assert counts == sorted(counts, reverse=True)
    assert set(survey.LISTED_APPS) <= {app for app, _ in bars}


def test_committed_results_match_the_responses():
    """Rerun scripts/survey_results.py after merging survey PRs."""
    counts, responses = survey.tally(survey.SURVEY / "responses")
    readme = (survey.SURVEY / "README.md").read_text(encoding="utf-8")
    block = readme[readme.index(survey.START):readme.index(survey.END)]
    assert responses == len(list((survey.SURVEY / "responses").glob("*.md")))
    assert f"**{responses} responses**" in block
    for app, count in survey.ordered(counts):
        assert f"| {app} | {count} |" in block, app
    svg = (survey.SURVEY / "results.svg").read_text(encoding="utf-8")
    assert svg.startswith("<svg") and svg.count("<rect") == 1 + sum(1 for c in counts.values() if c)
    assert re.search(r"<title[^>]*>Which LLM apps do you use most\?</title>", svg)


def test_update_readme_keeps_surrounding_text(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"before\n\n{survey.START}\nold\n{survey.END}\n\nafter\n", encoding="utf-8")
    survey.update_readme(readme, f"{survey.START}\nnew\n{survey.END}")
    assert readme.read_text(encoding="utf-8") == f"before\n\n{survey.START}\nnew\n{survey.END}\n\nafter\n"
