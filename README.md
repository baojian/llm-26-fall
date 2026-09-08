# NLP and LLMs — Fall 2026

Course materials for **NLP and LLMs** at Fudan University.

Course website: <https://baojian.github.io/llm-26-fall/>.

## Course Information

- **Course code:** CS40008.01
- **Semester:** Fall 2026 (2026–2027 academic year, first semester)
- **Schedule:** Wednesdays, periods 6–8 (13:30–16:10), weeks 1–16
- **First class:** September 9, 2026
- **Location:** Handan Campus, HGX103
- **Teaching language:** Chinese / English

## Schedule

See [docs/schedule.md](docs/schedule.md) for Fudan's daily period timetable,
what "periods 6–8" means in clock time (13:30–16:10), and the calendar weeks
with holidays and exam weeks. First class: September 9, 2026 (week 1). Last class: December 23, 2026
(week 16). Week 5, October 7, is **skip** for National Day. The revised lecture
sequence fits into the other 15 Wednesday meetings.

## First-lecture survey

Select at most two LLM apps you use most in everyday life through the
[Lecture 01 survey issue](https://github.com/baojian/llm-26-fall/issues/6).
Each student submits one response file in a PR using the GitHub website.
See the [survey guide](surveys/lecture-01/README.md) for instructions.

## Lecture slides

The [Reveal.js framework](slides/README.md) provides a shared theme, companion
notebooks, references, and browser-based demonstrations. It supports Plotly
charts, exported Excalidraw diagrams, and prepared Manim videos. The
[tokenization sample](slides/example/index.html) demonstrates the format; it is
not a complete first lecture. Published decks can be presented from a classroom
browser without an instructor laptop.

For local practice, run `uv sync` and `uv run python scripts/slides.py serve`,
then open the lecture and click **Notebook**. The launcher opens a working copy
under `workspace/` in JupyterLab, reusing a compatible server or starting one
with the course's uv environment.

## Your Workspace

Put your own notes, experiments, and exercise solutions in
[`workspace/`](workspace/README.md). Everything there except its README is
ignored by git, so pulling new course material never conflicts with your files
and your work stays out of any pull request you open. To modify a course file,
copy it into `workspace/` and edit the copy.

## Repository Status

The revised course schedule and content plan are available as a review draft.
Lecture materials will be prepared through pull requests so that changes remain
easy to review and discuss.

This repository will contain course-facing materials such as lecture notes,
examples, exercises, and contribution guidance. Students will be welcome to
suggest improvements and contribute suitable material through issues and pull
requests after the participation guidelines are published.
