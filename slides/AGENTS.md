# Lecture authoring

Read this file and `slides/README.md` before creating or changing a lecture.

## Content and teaching

- Write the lecture around its central question and concrete learning objectives.
- Put a short exercise immediately after the concept it checks. Give it an ID,
  a time estimate, an expected student response, and checked solution notes.
- Keep exercise IDs and order consistent with `practice.ipynb`.
- Verify all calculations and runnable examples. Identify toy data explicitly.
- Cite factual claims, figures, and borrowed material beside the relevant
  content or in speaker notes. Use verified URLs and precise reading sections.
- Keep ungraded practice distinct from the course's published assessment rules.
- Use speaker notes for detailed explanations, common mistakes, and timing.

## Layout

- Reuse `shared/theme.css`, `shared/course.js`, and the layouts demonstrated in
  `example/`. Lecture changes normally belong in `slides.md` and its notebook.
- Use one idea per slide. Aim for at most 70 words of visible prose and 12 code
  lines. Split a crowded slide before reducing content or changing the design.
- Keep body text at least 30px, code at least 26px, and sources at least 24px on
  the 1280×720 canvas. Keep code lines short enough to show without scrolling.
- Use a stable slide ID. Keep the final reading slide's ID `references`.
- Use native tables for comparisons and measurements. Label units and sources.
- Do not add inline styles, automatic text shrinking, or new fonts per lecture.
- Use animation only when the timing explains a mechanism or stages an answer.

## Browser delivery and verification

- Use Reveal.js as the presentation shell and apply the standard toolkit
  automatically when preparing relevant content: Plotly.js for interactive
  charts, Excalidraw for diagrams, and Manim for animations of mechanisms.
- Choose tools according to the explanation; every lecture need not use all
  three. Keep concise text, equations, code, and tables when they suffice.
- Follow the embedding examples in `README.md`. Plotly specifications load
  automatically. Export Excalidraw drawings to SVG and render Manim to MP4
  before publication. Keep editable `.excalidraw` and Python source beside
  each exported asset. Do not require either editor during the lecture.
- Give charts meaningful axis labels and descriptions. Give diagrams alt text
  and animations playback controls, a descriptive label, and a poster that
  summarizes the mechanism for the PDF. Avoid autoplay.

- The published slides must work on a classroom computer in its browser. Host
  browser assets with the course. Students run `uv run python scripts/slides.py
  serve` locally to open companion notebooks in JupyterLab.
- Keep the Notebook link connected to the shared launcher. It opens a working
  copy under `workspace/`, reuses compatible JupyterLab servers, and starts the
  project environment when needed. Preserve existing student work.
- Keep lecture-specific interactions in that lecture's `demo.js`, declared in
  `lecture.json`. Provide an initial visible example and a reset action.
- Use browser demonstrations for small mechanisms. Heavy computation needs
  prepared results or an explicitly documented remote compute environment.
- Run `npm --prefix slides run check -- FOLDER` after editing a lecture.
- Inspect every generated slide image for balance, overlap, and readability.
  Automatic checks are necessary but do not replace looking at the slides.
- Run `npm --prefix slides run pdf -- FOLDER` before distributing a PDF, and
  check its page count and legibility.
- Keep screenshots, PDFs under review, and test reports in ignored `.checks/`.
- Preserve the shared theme when fixing lecture content. If the task changes
  the framework, recheck the example deck and affected lectures.
