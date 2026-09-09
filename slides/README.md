# Reveal.js lecture framework

Use one shared visual system for the course, with a Markdown file and companion
notebook for each lecture. The [sample deck](example/index.html) demonstrates
the layouts using a short tokenization lesson. It is a format sample, not the
complete first lecture.

[Lecture 01](lecture-01/index.html) combines course introduction, local-model
experiments, Unicode, and byte BPE for three 45-minute periods. Its
[teaching plan](lecture-01/teaching-plan.md) maps the timing and sources; its
[notebook](lecture-01/lecture-01-exercise.ipynb) contains the complete
Spring example texts, Ollama demonstrations, matching E01–E06 exercises, and
optional multimodal experiments. Its 71 slides include ten historical content
pages and three September 2026 updates on models, Terminal-Bench Science, and
the Navier–Stokes research announcement, plus two Turing test illustrations,
restored regex practice, and an interactive
BPE trace. Short practices P01–P02 accompany the six E exercises in the same
notebook order. The deck bundles the Big Data image
and Johannesburg video clip; see [media provenance](lecture-01/assets/README.md).
Course paper PDFs are kept in [papers/](../papers/README.md).

## Teaching from a classroom browser

Once the slide files reach the branch published by GitHub Pages, open the deck
at `/llm-26-fall/slides/FOLDER/` on the classroom computer. The HTML, styles,
equations, and demonstration assets are hosted with the course. No local
installation or instructor laptop is required for viewing.

- **Space / arrows:** advance through slides and staged answers.
- **Esc:** overview of the deck.
- **F:** fullscreen.
- **S:** speaker view with notes. The browser may ask to allow a popup.
- **Notebook:** open your working copy in JupyterLab in a new browser tab when
  using the local course server.
- **References:** open the reading slide.
- **Print:** open the PDF print layout in a new tab.

The print layout has **−**, **+**, **100%**, and **Fit page** controls for
previewing the slides at different sizes. These controls and their zoom level
do not change the exported PDF's page dimensions.

The sample's text demonstration runs in the browser. Python notebook work uses
the local workflow below. On the public course website, the Notebook link shows
these setup instructions. Check website access on the classroom network before
the lecture.

## Students: open slides and notebooks locally

From your copy of this repository, run:

```sh
uv sync
uv run python scripts/slides.py serve
```

Open `http://127.0.0.1:8000/slides/example/` (or the lecture's folder) and click
**Notebook**. A new tab opens JupyterLab with a personal copy at
`workspace/slides/FOLDER/NOTEBOOK.ipynb`. Lecture 01 uses
`workspace/slides/lecture-01/lecture-01-exercise.ipynb`; the sample uses
`workspace/slides/example/practice.ipynb`. Later clicks reopen that copy and
preserve saved answers. The lecture's `assets/` folder is copied the first time
it is needed; existing personal files are not replaced by new course releases.

To receive an updated handout, rename your existing personal notebook in
JupyterLab, then click **Notebook** again. Keep the renamed file for your earlier
answers. Lecture 01's text-model demonstrations require a separate Ollama
installation and model; preparation is explained inside the notebook. Its core
tokenization exercises run offline without Ollama.

Lecture 01's optional vision experiment uses the bundled Big Data image and an already
installed vision-capable model. Its recorded clip plays with manual controls
in the slides or notebook. These demonstrations and the article-authorship
poll add no timed exercises beyond E01–E06. Sentiment is E01 and translation
is E02; Unicode and tokenization exercises continue as E03–E06. These IDs are
separate from the gallery's five task groups, which combine image and video
generation in Task 5.

For optional Stable Diffusion generation, retain the optional packages when
starting the preview:

```sh
uv sync --extra tokenization --extra multimodal
uv run --extra tokenization --extra multimodal python scripts/slides.py serve
```

Prepare the complete model pipeline separately and set `SD_MODEL` in the
notebook. The generation cell loads only local or cached weights; installing
the Python packages does not download a model. Other users can keep the basic
setup above.

The launcher checks registered local Jupyter servers. It reuses one when its
root directory includes this notebook, JupyterLab is available, and its
`python3` kernel uses the same uv environment. Otherwise it starts JupyterLab
with the course environment on an available local port. Different browser tabs
use separate JupyterLab workspaces while sharing the server.

JupyterLab and ipykernel are included in `pyproject.toml` and `uv.lock`. Each
student runs their own server on their own computer. A public website or a
generic static server cannot start Jupyter; use `scripts/slides.py serve` for
this integration. GPU exercises still need the course compute environment.

JupyterLab keeps running when the slide preview stops, so ongoing notebook work
continues. Shut it down from JupyterLab when finished. Its local settings,
runtime files, and logs are stored in ignored `workspace/.jupyter/`.

The course page's Lecture 01 slide and exercise links open the local server at
`http://127.0.0.1:8000`, including when clicked from the public course page.
Start the preview before clicking either link. If using another port, open the
local `index.html` on that port; its material links keep the same server address.
Paper links point to PDFs hosted with the course and also work locally.

## Files you edit

```text
slides/
  shared/                 One theme and initialization script
  vendor/                 Pinned Reveal.js, KaTeX, and Plotly browser assets
  template/               Starter files copied by the creation command
  example/                Checked sample with a browser demonstration
  lecture-01/             Lecture 01 deck and lecture-01-exercise.ipynb
  01-tokenization/        A lecture created when its content is ready
    index.html            Shared viewer shell
    lecture.json          Title, language, and optional demo module
    slides.md             Explanations, exercises, and references
    practice.ipynb        Runnable examples in the same order
    assets/               Chart data, drawings, videos, and their sources
```

The `01-tokenization/` folder above illustrates the creation command's output.
Each lecture can set `notebook` in `lecture.json` to a filename such as
`lecture-01-exercise.ipynb`. The default is `practice.ipynb`; the file must be
inside the lecture folder.

Lecture 01 also provides `lecture-01-exercise-tokenization.ipynb` for extended
practice from the Spring course, with minimum edit distance removed. Install
its optional dependencies with `uv sync --extra tokenization` and launch the
preview with `uv run --extra tokenization python scripts/slides.py serve`.
Open it through the course page's **Extended practice** link or the deck's
**Extended tokenization notebook** slide. The default **Notebook** link keeps
the timed classroom exercises E01–E06.

Additional notebooks are listed in `additional_notebooks` in `lecture.json`.
Use `../shared/notebook.html?lecture=FOLDER&notebook=FILENAME.ipynb` to open one.
The launcher accepts only listed notebooks and creates separate personal
copies. It adds newly supplied assets without replacing existing student files.

## Create and preview a lecture

From the repository root:

```sh
uv run python scripts/slides.py new 1 "Tokenization"
uv run python scripts/slides.py serve
```

Open `http://127.0.0.1:8000/slides/01-tokenization/`. Edit `slides.md` and reload
the browser. The creation command refuses to overwrite an existing folder.
Replace all `REPLACE:` prompts before treating the deck as ready.

The creation command is for preparing materials. Students use the same local
preview command to work through notebooks; the lecturer can also present the
published slides directly from a classroom browser.

## Reusable slide layouts

Separate slides with a line containing `---`. Use a clear slide ID and a short
title. Detailed explanations follow `Note:` and appear in speaker view.

```markdown
<!-- .slide: id="unicode" -->

## Code points and UTF-8 bytes

A code point identifies a character value. UTF-8 encodes it as bytes.

Note:
Ask students to predict the byte count for a Chinese word.
```

The shared styles provide:

| Layout | How to use it |
| --- | --- |
| Title | `class="title-slide"` with title, subtitle, and byline |
| Section introduction | `class="section-slide"` with one central idea |
| Topic outline | `class="outline-slide"`, a list with `class="outline-topics"`, and `aria-current="step"` on the current topic |
| Explanation | A heading and a few short paragraphs or list items |
| Comparison | `<div class="columns">` containing two `<div>` elements; add `columns-wide-left` for a 60% / 40% split |
| Code | A fenced Python block, usually 6–12 lines |
| Exercise | `class="exercise"`, an exercise ID, time, and a task |
| Answer | `<div class="answer fragment">` after the exercise |
| Results | A native Markdown or HTML table with labeled columns |
| References | `class="references" id="references"` and precise readings |

Use `$$ ... $$` for display equations and `$ ... $` for inline math. The sample
also demonstrates a local `demo.js` with editable input and a reset button.
Keep custom interaction code out of the Markdown and the shared initializer.

Repeat the same topic outline at section transitions. The current topic appears
in bold black, while the others appear in gray. A caption can identify the
current teaching period. See Lecture 01 for the four-topic example.

## Standard visual toolkit

Reveal.js is the presentation shell. When preparing a lecture, use Plotly.js
for interactive charts, Excalidraw for diagrams, and Manim for animated
mechanisms where they help the explanation. The shared viewer handles their
prepared assets; individual lectures do not need their own library setup.

### Plotly.js

Place a Plotly figure specification with `data`, `layout`, and optional `config`
in the lecture's `assets/` folder, then embed it:

```html
<div class="plot" data-plotly="assets/loss.json" role="img"
     aria-label="Describe the plotted result and its units."></div>
```

The viewer loads its local Plotly bundle only when a chart is present. Loss
curves, distributions, and heatmaps use the same convention. Shared defaults
provide readable text and consistent colors. The sample's
[figure specification](example/assets/sequence-length.json) plots checked toy
corpus counts. Label simulated data explicitly and keep measurement provenance
with real results. Use SVG-based traces for reliable PDF export.

### Excalidraw

Keep the editable drawing and an exported SVG together. Embed the export:

```html
<img class="diagram" src="assets/tokenizer.svg"
     data-excalidraw-source="assets/tokenizer.excalidraw"
     alt="Text passes through the tokenizer to produce token IDs.">
```

Prepare and export the drawing before publication. Use generous label sizes
and ensure exported fonts and images work without remote requests. The browser
shows the finished drawing; live Excalidraw editing is not part of this viewer.

### Manim

Keep the scene's Python source, the rendered MP4, and a summary poster together:

```html
<video class="animation" src="assets/bpe.mp4"
       data-manim-source="assets/bpe.py" poster="assets/bpe-summary.svg"
       controls playsinline preload="metadata"
       aria-label="Two BPE merges combine l, o, and w into the token low."></video>
```

Render the scene using Manim during preparation, then copy the finished media
into the lecture folder. The classroom browser plays the video with normal
controls; changing slides pauses it. The viewer uses the poster in the PDF.
Use a short silent clip with visible explanations, or provide captions for
narration. Manim and Excalidraw are authoring tools, so they are not installed
or run by the classroom viewer. Their exports must exist before publishing.

Recorded video examples use the same `video.animation` element, controls,
descriptive label, and poster, but omit `data-manim-source` when the clip was
not produced by a Manim scene. Keep its source attribution in the lecture's
asset notes. Lecture 01's recorded clip demonstrates this convention;
its extracted poster appears in print, and playback is manual.

Official guides: [Plotly.js](https://plotly.com/javascript/getting-started/),
[Excalidraw exports](https://docs.excalidraw.com/docs/@excalidraw/excalidraw/api/utils/export),
and [Manim rendering](https://docs.manim.community/en/stable/tutorials/quickstart.html).

## Checks before class

Install authoring dependencies once:

```sh
npm ci --prefix slides
npm --prefix slides run browser:install
```

Check a deck and generate review images:

```sh
npm --prefix slides run check -- example
npm --prefix slides run pdf -- example
uv run python -m pytest tests/
```

The checker tests every slide at 1440×900, 1280×720, and 1024×768. It rejects
overflowing content, small text, missing local assets, unmatched exercise IDs,
unfinished placeholders, and slide-specific style overrides. It also checks
sample interactions and runs with external runtime requests blocked.

Review the images and PDF in `slides/.checks/example/`. Long text should become
another slide. A successful check does not establish that the teaching content
is accurate or that the visual composition is effective.

With the local course preview running, check the JupyterLab integration with
`npm --prefix slides run check:notebook`. This opens the sample notebook twice
to verify server reuse and the course kernel, without editing or executing
cells. JupyterLab remains available afterward.

To check Lecture 01 instead, run
`npm --prefix slides run check:notebook -- http://127.0.0.1:8000 lecture-01`.
Append `--course-page` to test its exercise link directly from `index.html`.
Append `--notebook=lecture-01-exercise-tokenization.ipynb` to check the extended
notebook instead. This also works with `--course-page`.

For PDF export directly in Chrome or Chromium, open **Print**, choose **Save as
PDF**, landscape orientation, no margins, and background graphics. The PDF shows
staged answers in their final state. Speaker notes remain outside the PDF.

## Working with an AI assistant

Use the [authoring prompt](authoring-prompt.md) and the local [authoring rules](AGENTS.md).
Ask for a teaching outline and a small pilot before expanding to a full lecture.
Keep shared CSS stable while revising content, and inspect screenshots after
each substantive edit.

## Dependencies and publication

Reveal.js, KaTeX, and Plotly versions are pinned in `package.json` and `package-lock.json`.
The vendored assets are tracked so the existing GitHub Pages deployment can
serve the slides without a separate build pipeline. Update them deliberately:

```sh
npm ci --prefix slides
uv run python scripts/slides.py vendor
```

Retain the third-party licenses in `vendor/`. Commit a new lecture through a PR,
merge it through the repository's usual workflow, and verify the published URL.
An unmerged local draft is not yet available on the public course website.

Official documentation:
[Markdown](https://revealjs.com/markdown/),
[math](https://revealjs.com/math/),
[speaker view](https://revealjs.com/speaker-view/),
[PDF export](https://revealjs.com/pdf-export/).
