# Lecture 11 — Diffusion models and diffusion language models

The lecture introduces forward noising and learned denoising for image
generation, builds the autoencoder/VAE and DDPM/D3PM foundations, then covers
diffusion language models through Mercury, LLaDA and Dream. Training, sampling,
generation order, and quality–speed trade-offs connect the two parts.

The instructor's PowerPoint is converted into editable Reveal.js content.
All 69 slides retain their original order, body text, equations, illustrations,
and citations. The third pass gives the pages specific topic headings, removes
the 69 original page-number placeholders, and uses the shared course cover
with CS40008.01 and November 18, 2026. Reveal's navigation counter remains.

## Preview and edit

Run the course server and open the lecture:

```sh
uv run python scripts/slides.py serve
```

<http://127.0.0.1:8000/slides/lecture-11/>

- Edit headings, paragraphs, references, and equations directly in
  [slides.md](slides.md). The 113 text boxes are real HTML, not page images.
- The 31 equations use editable TeX inside `$...$` delimiters and the course's
  local KaTeX renderer. Original mathematical notation and wording are retained.
- The 69 illustration instances use 66 original picture files, including five
  GIFs. Text inside paper screenshots and diagrams remains part of those images.
- [native-layout.css](native-layout.css) records the original positions and run
  typography. Each text box has a stable `data-source-text-id`; CSS class names
  connect it to its geometry and text styles. [native.css](native.css) contains
  common layout rules, the slide 49 spacing adjustment for KaTeX, and the
  footer-link width adjustment for Linux font fallback.
- For a later approved content revision, update the manifest's expected
  `paragraphs` alongside the HTML/TeX; retain `original_paragraphs` and source
  hashes as the historical baseline.
- [adaptations.json](adaptations.json) records the revised page titles and
  course cover. The builder reapplies these changes and excludes source slide
  numbers on every rebuild. The cover uses the same shared typography and
  spacing as Lecture 10; content pages retain their source layout.
- [demo.js](demo.js) manages GIF playback. Slide 6's second picture is a Reveal
  fragment. Print displays its final state and first-frame GIF posters.

The deck keeps the course toolbar, keyboard navigation, overview, speaker view,
References, and Print. No notebook or new exercise was added. The PowerPoint
has no authored speaker notes; the content is now visible HTML rather than a
transcript hidden in notes.

## Preservation and limitations

The source design uses Tahoma with occasional font overrides. CSS preserves
those font choices with system fallbacks. Browser font metrics and KaTeX cause
small line-wrap and paragraph-spacing differences from PowerPoint. Source
font sizes on content pages are retained; text is not automatically shrunk. In slide 49, three
paragraph gaps are reduced by 6px to accommodate inline math without overlap.
On slides 31–32, the LLaDA link boxes are widened from 160.25px to 200px
to keep the original 24px label on one line when Linux falls back to Arial.
Their position, wording, and font size are unchanged. Original small citations
and illustration labels are retained.

[The manifest](assets/source-manifest.json) records the source PPTX hash,
original DrawingML text, editable TeX/text, text-box IDs, source font sizes,
illustration geometry/crops, and image hashes. It also keeps all 180 original
text boxes, including the removed numbers and replaced cover, separately from
the 113 current text boxes. `mc:Choice` is used for modern
math, avoiding duplicated legacy fallback pictures. Main text is searchable
and selectable in the browser and PDF. Illustration labels remain raster text.

The first round's full-slide SVGs have been removed from the current deck.
The earlier vector baseline remains in Git history. Original image files are
loaded through Reveal's lazy image support; all runtime resources are local.

## Rebuild and verify

Routine body edits should change the HTML/TeX and CSS directly. Update titles
and cover text in `adaptations.json`, then rebuild. Rebuilding replaces manual
HTML edits with a new conversion of the recorded source PPTX and reapplies the
title/cover adaptations. The original PowerPoint is kept outside the repository.

```sh
uv run --with pillow==12.1.0 python slides/lecture-11/build_native.py SOURCE.pptx
npm --prefix slides run check -- lecture-11
npm --prefix slides run pdf -- lecture-11
node slides/lecture-11/check-demo.mjs http://127.0.0.1:8000
uv run python -m pytest tests/test_lecture_11_math.py -q
```

The builder verifies the source hash. Pillow is only an authoring dependency
for lossless first-frame GIF posters. Viewing requires neither Python nor a
network connection to an external service.

Checks cover body text/order, revised native headings, removed page numbers,
course cover, source font sizes,
clipping, image hashes/crops, math rendering, text selection/editing, GIF
playback, links, click reveals, and PDF output. Review artifacts are kept in
ignored `slides/.checks/lecture-11/`.
