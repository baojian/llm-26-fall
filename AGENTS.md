# Repository Guidelines

## Project Structure & Module Organization

This repository contains course materials for Fudan University’s Fall 2026 NLP and LLM course. `README.md` is the course entry point, while `docs/` holds shared documentation such as `docs/schedule.md`. Python dependencies and version requirements live in `pyproject.toml`; `uv.lock` records the reproducible environment. Put personal notes, experiments, and exercise solutions under `workspace/`, whose contents are ignored except for `workspace/README.md`.

There is currently no package or test directory. Add course-facing examples and exercises outside `workspace/` using descriptive directories as the repository grows; keep related tests in `tests/` when executable code is introduced.

## Build, Test, and Development Commands

- `uv sync`: create or refresh `.venv` from the tracked lockfile.
- `uv run python path/to/script.py`: run Python with project dependencies.
- `uv add <package>`: add a runtime dependency and update both dependency files.
- `uv add --dev <package>`: add development tooling.
- `uv run pytest`: run tests once pytest and a test suite are added.
- `uv run ruff check .`: run lint checks once Ruff configuration is added.

Prefer `uv run` to manually activating the virtual environment. Commit `uv.lock` whenever `pyproject.toml` dependency changes alter it.

## Coding Style & Naming Conventions

Write all repository content in English. Use four-space indentation for Python, `snake_case` for modules and functions, and `PascalCase` for classes. Choose descriptive, lowercase Markdown filenames (for example, `docs/tokenization.md`). Keep Markdown headings hierarchical, prose concise, and links relative within the repository. No formatter or linter is configured yet; follow surrounding style and keep future tooling changes separate and documented.

## Testing Guidelines

No test framework or coverage threshold is currently configured. For new executable material, add focused pytest tests named `tests/test_<topic>.py`, with test functions named `test_<behavior>`. Include normal cases and failure or edge cases. Verify documentation links and commands manually when changing course notes.

## Course Assessment Policy

Instructor decision, September 15, 2026: undergraduate and master's students are treated equally throughout this course. Use the same required work, learning objectives, grading criteria, point allocations, and maximum scores for both groups.

There are no starred parts, graduate-only extensions, extension bonus points, or extra-credit grading tracks. Do not vary an assignment, project rubric, or grade calculation by degree level. Older plans describing these distinctions are superseded; correct conflicting handouts, rubrics, submission instructions, and grading code when working on them.

Graded assignments and project deliverables are submitted privately through eLearning. Public assignment releases contain instructions, starter code, supplied data, and public tests. Reference solutions, hidden tests, and grading administration belong in the private instructor repository; student submissions and grades must not be committed to either repository.

## Survey Policy

The Lecture 01 LLM-app survey allows any number of selections, including zero
(instructor decision, September 16, 2026). Include responses with more than two
selected apps in the results. The former two-app limit is superseded.

Instructor decision, September 20, 2026: survey submissions close at
**23:59 on September 20, 2026 (Asia/Shanghai, UTC+08:00)**. Students should
verify that their response is merged into `main`; students who have not
submitted must open their response PR by the deadline. Review and resolve
responses submitted on time even if the review or merge happens afterward.
Do not accept new survey responses after the cutoff, including at the future
archive location.

After the deadline, close collection issue #6 and resolve outstanding survey
PRs, then implement issue #139 through a separate PR: move the survey to
`tasks/l01-tokenization/llm-app-survey/`, preserve accepted responses and
participation, and leave a forwarding README at the old location. Begin the
migration no earlier than September 21, 2026, 00:00 (Asia/Shanghai). The move
does not reopen submissions or require students to resubmit.

## Commit & Pull Request Guidelines

Recent commits use short, lowercase, imperative summaries such as `configure uv Python environment`; follow that pattern and keep each commit focused. Make changes through pull requests rather than committing directly to `main`. PRs should explain the purpose, list validation performed, link relevant issues, and include screenshots only for visual output. Never commit secrets, model weights, checkpoints, logs, or experiment outputs; keep secrets in ignored `.env` files and provide `.env.example` when configuration must be documented.
