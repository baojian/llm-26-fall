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

## Commit & Pull Request Guidelines

Recent commits use short, lowercase, imperative summaries such as `configure uv Python environment`; follow that pattern and keep each commit focused. Make changes through pull requests rather than committing directly to `main`. PRs should explain the purpose, list validation performed, link relevant issues, and include screenshots only for visual output. Never commit secrets, model weights, checkpoints, logs, or experiment outputs; keep secrets in ignored `.env` files and provide `.env.example` when configuration must be documented.
