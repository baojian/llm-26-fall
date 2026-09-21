# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Course materials for **Natural Language Processing and Large Language Models** (CS40008.01) at Fudan University, Fall 2026. The public course website is <https://baojian.github.io/llm-26-fall/>. All content (lecture notes, slides, notebooks, docs) is written in English, even though the course is taught in Chinese. The first class was September 9, 2026; the course runs on Wednesdays through December 23, with October 7 skipped for National Day (see `docs/schedule.md`).

## Repository layout

```text
index.html                 Course website (GitHub Pages, served as-is from main)
assets/                    translations.js (zh strings) and local-materials.js for the website
docs/schedule.md           Class periods and the dated week table (source of truth for dates)
docs/course-revision.md    Instructor's syllabus and assessment review draft (Sep 7, 2026)
papers/                    Course reading PDFs with a provenance README
scripts/slides.py          `new`, `serve`, `vendor` commands for the slide framework
scripts/notebooks.py       JupyterLab launcher used by `serve` (opens copies under workspace/)
slides/                    Reveal.js lecture framework: shared/, template/, example/, lecture-01/, vendor/
slides/tools/              Playwright-based checkers (check.mjs, check-notebook.mjs), run via npm
tasks/l01-tokenization/llm-app-survey/  Closed survey archive and results
surveys/lecture-01/        Forwarding README to the survey archive
tests/                     pytest suite for scripts, notebooks, and the lecture-01 material
workspace/                 Student scratch area, git-ignored except its README
```

Only `lecture-01` exists so far. New lectures are created with `uv run python scripts/slides.py new <N> "<Title>"`, which copies `slides/template/` into `slides/NN-<slug>/`. Read `slides/AGENTS.md` and `slides/README.md` before creating or editing any lecture; they define the layout rules, size limits, asset conventions (Plotly JSON, Excalidraw SVG plus source, Manim MP4 plus poster), and the required checks.

## Course website (`index.html`)

A single hand-written page with inline CSS and vanilla JS. No build step; `.nojekyll` makes GitHub Pages serve the `main` root as-is. Edit the HTML directly.

- **Bilingual by dictionary, not by paired spans.** Visible text is written once in English. `assets/translations.js` maps each English string to Chinese; the page script walks text nodes and swaps them when the EN/中文 button sets `zh` (remembered in `localStorage` as `course-language`). When you change any English text, update the matching key in `assets/translations.js`, or the Chinese view silently falls back to English. There is no `?lang=` URL parameter.
- The schedule table highlights the current week from each row's `data-start` Sunday date. Keep those dates, the Coursework cards (quiz and assignment weeks), and `docs/schedule.md` consistent.
- Material links in the table carry `data-local-path`; `assets/local-materials.js` rewrites them to the local preview server when the page is opened from `localhost`. Slide and notebook links point at `http://127.0.0.1:8000`, so students must run `scripts/slides.py serve` first.
- The live assessment split is quizzes 10%, assignments 45%, individual project 45% (`index.html`, legend near the Assessment section). `docs/course-revision.md` still shows an older 5/15/40/40 proposal; the website is authoritative for students.

## Python environment (uv)

Managed with [uv](https://docs.astral.sh/uv/). Python is pinned to 3.11 via `.python-version`; `pyproject.toml` requires `>=3.11`. `uv.lock` is **intentionally tracked**; never add it to `.gitignore` and always commit lockfile changes alongside `pyproject.toml` changes.

```bash
uv sync                                    # core environment (slides, JupyterLab, tests)
uv sync --extra tokenization               # adds torch, transformers, tokenizers, ollama, spacy, ...
uv sync --extra tokenization --extra multimodal   # adds diffusers and accelerate
uv run python scripts/slides.py serve      # course preview at http://127.0.0.1:8000
uv run python -m pytest tests/             # run the test suite (see note below)
```

Run tests as `uv run python -m pytest`, not `uv run pytest`. The tests import `scripts.slides` and `scripts.notebooks` from the repo root, and only the `python -m` form puts the root on `sys.path`. Some tests execute every code cell of the lecture-01 notebooks offline; the full suite takes about 15 seconds.

### Toolchain mirrors Stanford CS336

The core dependency set matches Stanford's CS336 lecture repo, checked out locally at `/Users/baojian/git/stanford-cs336-lectures`: Python 3.11 and the same five packages with the same lower bounds (`edtrace`, `einops`, `mmh3`, `modal`, `tiktoken`). PyTorch, NumPy, Triton, and SymPy arrive transitively through `edtrace`. On top of CS336 this repo adds `jupyterlab`, `ipykernel`, and `psutil` for the notebook launcher, optional extras `tokenization` and `multimodal` for the lecture-01 notebooks, and `pytest` in the `dev` dependency group.

When adding a dependency, check CS336's `pyproject.toml` first and reuse the same package and bound if it is already there. Keep lower bounds rather than exact pins unless a lecture breaks. Heavy packages belong in an extra, not in the core list, so `uv sync` stays fast for students.

## Slide framework checks (Node)

Authoring checks use Playwright and are separate from pytest:

```bash
npm ci --prefix slides && npm --prefix slides run browser:install   # once
npm --prefix slides run check -- lecture-01      # screenshots, overflow, assets, exercise IDs
npm --prefix slides run pdf -- lecture-01        # PDF export into slides/.checks/
npm --prefix slides run check:notebook -- http://127.0.0.1:8000 lecture-01   # needs serve running
```

Output goes to the ignored `slides/.checks/`. Vendored Reveal.js, KaTeX, and Plotly live in `slides/vendor/` and are tracked on purpose so GitHub Pages needs no build; refresh them only with `uv run python scripts/slides.py vendor` after `npm ci`.

## Task Solution Merge Policy — Mandatory

Instructor decision, September 20, 2026: merge student exercise solutions
**only after the task's published deadline, in one batch per task**.

- Apply this to every PR containing a student task solution or a correction
  to one, regardless of its title, author, or file location. Review and run
  checks before the deadline, but do not merge or enable auto-merge.
- Before any merge, read the task's `task.toml`, `instruction.md`, and release
  issue; verify that their deadlines agree and that the current time is
  strictly after the cutoff in its stated time zone. A missing, ambiguous,
  or conflicting deadline blocks merging. Passing tests does not waive this
  rule. The three released Lecture 01 exercises cannot merge before
  **September 28, 2026, 00:00 (Asia/Shanghai, UTC+08:00)**.
- After the deadline, review all on-time PRs for that task before starting
  its single batch. Resolve outstanding reviews or obtain the instructor's
  decision on them first. Keep individual student PRs for attribution; merge
  the accepted PRs together in that batch, then refresh the progress board.
- Record the batch's PR numbers, reviewed head commits, deadline, and status
  in the task's `merge-batch.md` through a PR before merging solutions;
  record completion afterward. Check this record and the task's merge
  history before starting. An interrupted batch may resume; do not start a
  second batch or add submissions to a completed batch.
- Late submissions, later solution fixes, and any exception to the deadline
  or single-batch rule require an explicit instructor decision naming the
  task and exception. A general request to "merge student PRs" does not
  override this policy. Do not change a deadline to bypass it.

Survey responses and course-material fixes containing no student exercise
solutions follow their own review rules. Public PRs and forks remain visible
before merge; this policy delays publication on `main`, not access to the
PRs. Work requiring private submissions must use eLearning or another
instructor-approved private channel.

## Archived Lecture 01 survey

Collection issue #6 is closed. The deadline was **September 20, 2026, 23:59
(Asia/Shanghai, UTC+08:00)**; on-time PRs were reviewed even when merged later.
After those reviews, #139 moved the accepted survey on September 21 to
`tasks/l01-tokenization/llm-app-survey/`. The former `surveys/lecture-01/`
location contains only a forwarding README.

Keep accepted response contents, filenames, totals, and participation intact.
Do not accept new responses at either location or ask accepted students to
resubmit. Any number of app selections, including zero or more than two, is
valid under the instructor's September 16 policy. The archive has a separate
Markdown-response tally, no Python task interface, and one survey participation
entry per student; it does not affect coding-task correctness. Regenerate the
archive results with `uv run python scripts/survey_results.py` and participation
with `uv run python scripts/progress.py`.

## Conventions

- **Changes go through pull requests.** Do not commit directly to `main`. Instructor work so far has used `codex/<topic>` branches; use a similar descriptive branch name.
- **Never commit model weights or experiment output.** `.gitignore` excludes `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `checkpoints/`, `outputs/`, `/output/`, and `logs/`. Reference download locations instead.
- **Instructor material lives elsewhere.** Solutions, hidden tests, rubrics, answer keys, and grading scripts belong in the private companion repo `baojian/llm-26-fall-instructors`, expected at `../llm-26-fall-instructors` (it exists on GitHub but is not checked out on this machine as of September 10, 2026). Never add such material here. Assignment handouts arrive only through that repo's `grading/strip_solutions.py`, so do not hand-edit released handout files; fix the canonical copy there and re-release.
- **Spring 2026 sources.** The previous course is `baojian/llm-26`, checked out at `/Users/baojian/git/llm-26`. `docs/course-revision.md` maps which spring notebooks and CS336 lectures each fall week reuses; consult it before authoring a new lecture.
- **Student work lives in `workspace/`.** Everything there except `workspace/README.md` is git-ignored. The notebook launcher copies lecture notebooks and assets into `workspace/slides/<lecture>/` and never overwrites existing student files. Never place course material there and never commit anything from it. When a student asks to modify a course file, copy it into `workspace/` and edit the copy.
- **Secrets stay in `.env`** (ignored). An example config must be named `.env.example`.
- `AGENTS.md` at the root is the Codex-facing twin of this file; keep the two consistent when repository structure changes.
