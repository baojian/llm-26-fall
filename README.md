# NLP and LLMs — Fall 2026

Course materials for **NLP and LLMs** (CS40008.01) at Fudan University.
Visit the [course website](https://baojian.github.io/llm-26-fall/) for lectures,
readings, and assessment information.

## Getting started

1. Install [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone the course: `git clone https://github.com/baojian/llm-26-fall.git`.
3. Run `cd llm-26-fall`, then start the local course server: `uv run python scripts/slides.py serve` and open <http://127.0.0.1:8000>.
4. Run `git pull` before class and restart the server to get the latest materials.

Submit quizzes, assignments, and individual projects through
[eLearning](https://elearning.fudan.edu.cn/). For optional participation
activities, see the [participation workflow](docs/participation-workflow.md).

Find assignment deadlines and updates in the
[assignment announcement issues](https://github.com/baojian/llm-26-fall/issues?q=is%3Aissue%20label%3Aassignment).
Subscribe to an assignment's issue to follow its announcements and clarifications.

## Course information

- **Course code:** CS40008.01
- **Semester:** Fall 2026 (2026–2027 academic year, fall semester)
- **Schedule:** Wednesdays, periods 6–8 (13:30–16:10), weeks 1–16
- **First / last class:** September 9 / December 23, 2026
- **Make-up class:** Saturday, October 10 (replaces October 7)
- **Location:** Handan Campus, HGX103
- **Teaching language:** Chinese lectures, English materials
- **Assessment:** Quizzes 10%, assignments 45%, individual project 45%

Dates, periods, and holidays: [docs/schedule.md](docs/schedule.md). Assessment details: [course website](https://baojian.github.io/llm-26-fall/).

The [individual project catalog](docs/project-candidates.html) starts with a
student-proposed project option and contains 48 suggested directions for
Lecture 05, with recent sources, baselines, evaluation plans, compute estimates,
and possible upstream contributions. Search by topic or browse by category.
The one-page proposal is due **October 21, 2026,
23:59 (Asia/Shanghai)** through eLearning.

## What is in this repository

| Folder | Contents |
| --- | --- |
| [slides/](slides/README.md) | Lecture decks, companion notebooks, and teaching notes. |
| [docs/](docs/README.md) | Course schedule, reading guides, and reference notes. |
| [papers/](papers/README.md) | PDFs and citations for the course readings. |
| [tasks/](tasks/README.md) | Optional participation activities, instructions, and the [progress board](tasks/PROGRESS.md). |
| [assignments/](assignments/README.md) | Assignment handouts, starter code, supplied data, and public tests. |
| [workspace/](workspace/README.md) | Personal notes, experiments, and exercise solutions; contents other than its README are ignored by git. |
| [pipeline/](pipeline/) | Data preparation, language modeling, and evaluation utilities. |
| [scripts/](scripts/) | Course server, notebook launcher, and maintenance tools. |
