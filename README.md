# NLP and LLMs — Fall 2026

Course materials for **NLP and LLMs** (CS40008.01) at Fudan University. Everything students need is in this
repo or on the course website: <https://baojian.github.io/llm-26-fall/>.

## Getting started

1. Install [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone the course: `git clone https://github.com/baojian/llm-26-fall.git`.
3. Run `cd llm-26-fall`, then start the local course server: `uv run python scripts/slides.py serve` and open <http://127.0.0.1:8000>.
4. For example, Lecture 01 is at <http://localhost:8000/slides/lecture-01/>.
5. Slides are updated before each lecture, so run `git pull` before class and restart the server.

You can also hand these steps to a coding agent such as Claude Code or Codex in one prompt.

Quizzes, assignments, and the individual project are submitted on [eLearning](https://elearning.fudan.edu.cn/); weekly tasks and surveys are submitted as pull requests from the GitHub website. See the [participation workflow](docs/participation-workflow.md).

The [Lecture 01 survey](surveys/lecture-01/README.md) closes on **September 20,
2026, at 23:59 (Asia/Shanghai)**. Submit your response PR by then, or check that
your existing response is merged into `main`. New responses will not be
accepted after the deadline. The teaching team will archive the survey under
`tasks/` afterward; accepted responses do not need to be resubmitted.

Find assignment deadlines and updates in the
[assignment announcement issues](https://github.com/baojian/llm-26-fall/issues?q=is%3Aissue%20label%3Aassignment).
Subscribe to an assignment's issue to follow its announcements and clarifications.

## Course information

- **Course code:** CS40008.01
- **Semester:** Fall 2026 (2026–2027 academic year, fall semester)
- **Schedule:** Wednesdays, periods 6–8 (13:30–16:10), weeks 1–16
- **First / last class:** September 9 / December 23, 2026
- **Make-up class:** Saturday, October 10 (National Day falls on October 7)
- **Location:** Handan Campus, HGX103
- **Teaching language:** Chinese lectures, English materials
- **Assessment:** Quizzes 10%, assignments 45%, individual project 45%

Dates, periods, and holidays: [docs/schedule.md](docs/schedule.md). Assessment details: [course website](https://baojian.github.io/llm-26-fall/).

## What is in this repository

| Folder | Contents |
| --- | --- |
| [`slides/`](slides/README.md) | Reveal.js lecture decks with companion notebooks. [Lecture 01](slides/lecture-01/index.html) (tokenization) and its [exercise notebook](slides/lecture-01/lecture-01-exercise.ipynb) are published; a [tokenization sample deck](slides/example/index.html) shows the format. |
| [`docs/`](docs/README.md) | Reading pages and reference notes: the bilingual [Lecture 01 preprocessing reader](docs/lecture-01-pre-tokenization.html), the [pretraining corpora catalog](docs/lecture-01-pretraining-datasets.md), the [tokenizer reading list](docs/tokenizer-reading-list.md), the [pretraining plan](docs/pretraining-plan.md), the [Lecture 02 note on LM metrics](docs/lecture-02-lm-metrics.md), and the [schedule](docs/schedule.md). |
| [`papers/`](papers/README.md) | PDFs and citations for the course readings. |
| [`surveys/`](surveys/lecture-01/README.md) | Weekly surveys: one response file per student, tallied into a chart. |
| [`tasks/`](tasks/README.md) | Small self-checking exercises: one submission file per student, checked automatically. The [progress board](tasks/PROGRESS.md) shows everyone's merged work. |
| [`assignments/`](assignments/README.md) | Graded assignments: [A1 Tokenization and language models](assignments/a1-tokenization-and-language-models/README.md). Submit solution ZIPs privately through eLearning. |
| [`workspace/`](workspace/README.md) | Your own notes, experiments, and exercise solutions. Everything there except its README is ignored by git, so `git pull` never conflicts with your files and your work stays out of any pull request you open. To modify a course file, copy it into `workspace/` and edit the copy: `uv run python workspace/<file>.py`. |
| `pipeline/` | The course data pipeline, built up lecture by lecture: the n-gram estimator, the bits-per-byte evaluation (`pipeline/eval.py`), and the reference-model document filter. Standard library only. |
| `scripts/` | Course tooling: the local server, notebook launcher, survey tally, progress board, the pretraining-dataset downloader, and the Lecture 02 experiment script. |
