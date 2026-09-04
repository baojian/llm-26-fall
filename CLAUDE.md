# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Course materials for **Natural Language Processing and Large Language Models** (CS40008.01) at Fudan University, Fall 2026. All course content in this repository (lecture notes, examples, exercises, docs) is written in English, even though the course is taught in Chinese.

The repo is currently scaffolding only: `README.md`, a student-facing `docs/schedule.md` (class period times and our 2026–2027 meeting dates), a uv project definition with the core dependencies, and a `.gitignore`. There is no source code, no tests, and no lint/format configuration yet. Do not assume a package layout exists; check the tree before referencing paths.

## Python environment (uv)

The project is managed with [uv](https://docs.astral.sh/uv/). Python is pinned to 3.11 via `.python-version`, and `pyproject.toml` requires `>=3.11`. `uv.lock` is **intentionally tracked** so every student gets a reproducible environment; never add it to `.gitignore` and always commit lockfile changes alongside `pyproject.toml` changes.

```bash
uv sync                      # create/refresh .venv from uv.lock
uv add <package>             # add a dependency (updates pyproject.toml and uv.lock)
uv add --dev <package>       # add a dev-only dependency
uv run python <script.py>    # run inside the project environment
uv run jupyter lab           # once jupyter is added as a dependency
```

### Toolchain mirrors Stanford CS336

The Python version and dependency set intentionally match Stanford's CS336 (Language Modeling from Scratch, Spring 2026) lecture repo, checked out locally at `/Users/baojian/git/stanford-cs336-lectures`. Both use Python 3.11 and the same five direct dependencies with the same lower bounds: `edtrace` (executable-lecture framework), `einops`, `mmh3`, `modal` (remote GPU execution), and `tiktoken`. PyTorch, NumPy, Triton, and SymPy are not listed directly; they arrive transitively through `edtrace`, exactly as in CS336.

When adding a dependency, check CS336's `pyproject.toml` first and reuse the same package and version bound if it is already there. Our lockfile resolved later than CS336's (September vs. May 2026), so patch/minor versions are newer (e.g. torch 2.14 vs 2.11); keep the bounds identical rather than pinning exact versions unless a lecture breaks.

Prefer `uv run <cmd>` over activating `.venv` manually. When test or lint tools are introduced, the expected invocations are `uv run pytest <path>::<test_name>` and `uv run ruff check .`; `.gitignore` already anticipates pytest, mypy, ruff, and Jupyter caches.

## Conventions

- **Changes go through pull requests.** The README states lecture materials are prepared via PRs so they stay reviewable. Do not commit directly to `main`.
- **Never commit model weights or experiment output.** `.gitignore` excludes `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `checkpoints/`, `outputs/`, and `logs/`. Keep large artifacts out of git; reference download locations instead.
- **Secrets stay in `.env`** (ignored). If an example config is needed, name it `.env.example`, which is explicitly allowed by `.gitignore`.
- Student contributions via issues and PRs are planned but participation guidelines have not been published yet.
