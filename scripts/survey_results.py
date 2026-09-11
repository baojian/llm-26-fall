"""Tally the lecture-01 LLM-app survey and draw the first course figure.

Reads every ``surveys/lecture-01/responses/*.md``, counts the checked apps, and
writes ``results.svg`` (a bar chart, one bar per app) next to the responses and
refreshes the results block of ``surveys/lecture-01/README.md``, which GitHub
shows on the folder page. Standard library only.

    uv run python scripts/survey_results.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
SURVEY = ROOT / "surveys/lecture-01"
LISTED_APPS = ["ChatGPT", "Claude", "Gemini", "DeepSeek", "Doubao (豆包)", "Qwen (千问)",
               "Kimi", "Tencent Yuanbao (腾讯元宝)", "Zhipu Qingyan (智谱清言)"]
MAX_CHOICES = 2
CHECKED = re.compile(r"^- \[[xX]\] (.+?)\s*$", re.MULTILINE)


def selections(text: str) -> list[str]:
    """Return the checked app names in one response, with Other resolved to its app name."""
    chosen = []
    for label in CHECKED.findall(text):
        if label.startswith("Other:"):
            name = label[len("Other:"):].strip()
            chosen.append(f"{name} (Other)" if name and name != "APP_NAME" else "Other")
        else:
            chosen.append(label)
    return chosen


def tally(responses_dir: Path) -> tuple[Counter, int, int]:
    """Count apps over valid responses. Returns (counts, valid responses, over-selected responses)."""
    counts: Counter = Counter({app: 0 for app in LISTED_APPS})
    valid = skipped = 0
    for path in sorted(responses_dir.glob("*.md")):
        chosen = selections(path.read_text(encoding="utf-8"))
        if len(chosen) > MAX_CHOICES:
            skipped += 1
            continue
        valid += 1
        counts.update(chosen)
    return counts, valid, skipped


def ordered(counts: Counter) -> list[tuple[str, int]]:
    """Bars from most to least used; ties keep the survey's listing order."""
    order = {app: index for index, app in enumerate(LISTED_APPS)}
    return sorted(counts.items(), key=lambda item: (-item[1], order.get(item[0], len(order)), item[0]))


def bar_chart_svg(bars: list[tuple[str, int]], title: str, subtitle: str) -> str:
    """Horizontal bar chart: one series, direct labels, recessive grid, no legend."""
    left, top, width, row = 250, 96, 960, 44
    bar_height = 28
    plot_width = width - left - 90
    height = top + row * len(bars) + 64
    largest = max((count for _, count in bars), default=0) or 1
    ink, muted, accent, grid, paper = "#202b38", "#566471", "#20578c", "#dce2e7", "#fbfbf9"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-labelledby="chart-title chart-desc" font-family="Arial, Helvetica Neue, PingFang SC, Microsoft YaHei, sans-serif">',
        f'<title id="chart-title">{escape(title)}</title>',
        f'<desc id="chart-desc">{escape("; ".join(f"{app}: {count}" for app, count in bars))}</desc>',
        f'<rect width="{width}" height="{height}" fill="{paper}"/>',
        f'<text x="32" y="44" font-size="26" font-weight="700" fill="{ink}">{escape(title)}</text>',
        f'<text x="32" y="72" font-size="17" fill="{muted}">{escape(subtitle)}</text>',
    ]
    step = 1 if largest <= 10 else 2 if largest <= 20 else 5
    for tick in range(0, largest + 1, step):
        x = left + plot_width * tick / largest
        parts.append(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{top + row * len(bars)}" stroke="{grid}" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + row * len(bars) + 24}" font-size="15" fill="{muted}" text-anchor="middle">{tick}</text>')
    for index, (app, count) in enumerate(bars):
        y = top + row * index
        bar_width = plot_width * count / largest
        parts.append(f'<text x="{left - 14}" y="{y + bar_height / 2 + 6}" font-size="17" fill="{ink}" text-anchor="end">{escape(app)}</text>')
        if count:
            parts.append(f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" rx="4" fill="{accent}"/>')
        parts.append(f'<text x="{left + bar_width + 10:.1f}" y="{y + bar_height / 2 + 6}" font-size="16" fill="{ink}">{count}</text>')
    parts.append(f'<text x="{left + plot_width / 2:.1f}" y="{height - 12}" font-size="15" fill="{muted}" text-anchor="middle">Students who selected the app (at most two selections each)</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


START, END = "<!-- survey-results:start -->", "<!-- survey-results:end -->"


def results_block(bars: list[tuple[str, int]], valid: int, skipped: int, today: dt.date) -> str:
    """The Markdown shown on the folder page, between the README's markers."""
    rows = "\n".join(f"| {app} | {count} |" for app, count in bars)
    skipped_note = (
        f" {skipped} response{'s' if skipped != 1 else ''} selected more than {MAX_CHOICES} apps and "
        f"{'are' if skipped != 1 else 'is'} not counted."
        if skipped else ""
    )
    return f"""{START}
Counted from the merged files in [responses/](responses/) on {today:%B %-d, %Y}:
**{valid} responses**, at most two selections each.{skipped_note}

![Bar chart of the number of students who selected each LLM app](results.svg)

| App | Students |
| :--- | ---: |
{rows}

Apps written under *Other* get their own bar with the suffix "(Other)".
{END}"""


def update_readme(readme: Path, block: str) -> None:
    """Replace the marked block in the README, keeping everything around it."""
    text = readme.read_text(encoding="utf-8")
    start, end = text.index(START), text.index(END) + len(END)
    readme.write_text(text[:start] + block + text[end:], encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--responses", type=Path, default=SURVEY / "responses")
    parser.add_argument("--out", type=Path, default=SURVEY, help="folder holding README.md and receiving results.svg")
    parser.add_argument("--date", type=dt.date.fromisoformat, default=dt.date.today())
    args = parser.parse_args()

    counts, valid, skipped = tally(args.responses)
    bars = ordered(counts)
    subtitle = f"Lecture 01 survey · {valid} responses · counted {args.date:%B %-d, %Y}"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.svg").write_text(bar_chart_svg(bars, "Which LLM apps do you use most?", subtitle), encoding="utf-8")
    update_readme(args.out / "README.md", results_block(bars, valid, skipped, args.date))
    for app, count in bars:
        print(f"{count:3d}  {app}")
    print(f"{valid} responses counted, {skipped} skipped -> {args.out / 'results.svg'}")


if __name__ == "__main__":
    main()
