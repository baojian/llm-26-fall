# Lecture checks and review previews

The **Lecture quality** workflow checks lecture-related pull requests. It also
supports **Actions → Lecture quality → Run workflow** for a rehearsal of the
selected branch before class. A student submission by itself does not trigger
the lecture workflow.

## What runs automatically

| Check | Evidence for the reviewer |
| :--- | :--- |
| Lecture examples and notebooks | Existing Python tests execute the offline teaching examples and verify numerical results, tensor shapes, notebook structure, and the notebook launcher; Node tests check the n-gram demonstration |
| Slides and PDF | Every deck is checked at three screen sizes for layout, font size, math rendering, local links and assets, and supported interactions; PDF export checks the print layout and page count |
| Reading pages and navigation | Browser checks exercise the bilingual reader, mobile layouts, and shared navigation |

Lecture changes recheck the example and all lecture decks because they share
the renderer, navigation, and assets. Newly added folders named `lecture-NN` or
`NN-topic` are discovered through their `lecture.json`. The unfinished
`slides/template/` is excluded. Add focused numerical tests in
`tests/test_lecture_NN.py` when introducing a lecture; discovering a deck does
not create tests for its teaching claims.

Python uses the course's `uv.lock` and `.python-version`; browser dependencies
come from `slides/package-lock.json`. The checks use small CPU examples, with
no model downloads or API credentials. Tests that require optional tokenization
packages are skipped in this core environment. Optional live model
demonstrations still need a manual rehearsal on the lecture machine.

## Review a lecture PR

1. Open the PR's **Checks** tab and the **Lecture quality** run.
2. Read each slide job's summary and download its `lecture-preview-*` artifact.
   It contains slide screenshots, a PDF, and `report.json` for a completed run.
   The summary records both the PR source commit and the tested checkout
   (GitHub's merge preview for a PR).
3. Inspect the screenshots and PDF for legibility, figure labels, and answer
   reveals. Use `lecture-reading-previews` to inspect reading pages and
   `lecture-example-results` for the Python test report.
4. Resolve failures on the same PR. Each push starts a new run and cancels the
   superseded run. A failed browser check may leave partial screenshots but
   no complete PDF; its log identifies the first failure. Layout failures
   also save `failure-WIDTH.png` at the failing viewport.

Artifacts are retained for 14 days. Download them before they expire, or rerun
the workflow. Reports and previews are review artifacts; the workflow does not
publish them to the course site or merge the PR. Passing checks supports human
review of clarity, pacing, learning objectives, and source accuracy.

## Reproduce a check locally

```sh
uv sync --locked
npm --prefix slides ci
npm --prefix slides run browser:install
uv run python -m pytest tests/test_lecture_03.py
node --test tests/test_ngram_review.mjs
npm --prefix slides run pdf -- lecture-03
npm --prefix slides run check:notes
```

Replace the lecture number as needed. Outputs stay in the ignored
`slides/.checks/` directory. See the [slide authoring guide](../slides/README.md)
for the full local checks and notebook launch rehearsal. Student exercise PRs
receive their own [task feedback](task-feedback.md).
