# Source assets

All illustrations come from the instructor-supplied
`lecture-11-diffusion-language-models.pptx`. Attribution printed in that deck
remains on the corresponding page. Body text and references remain unchanged.

- `media/`: 66 original PNG/JPEG/GIF files used by 69 picture instances. Their
  bytes match the source PPTX; the original crops on slides 6 and 58 are CSS.
- `media/*-poster.png`: lossless first-frame GIF posters for printing and
  inactive slides, derived with Pillow 12.1.0.
- `source-manifest.json`: source SHA-256, source text and TeX, picture SHA-256
  values, page order, original geometry/crops and source font sizes. All original
  text boxes remain in the provenance inventory alongside the current expected
  text; `../adaptations.json` records the authorized title and cover revisions.

Headings, body text, lists, equations and references are native HTML/KaTeX in
`../slides.md`. No full-slide SVG/PNG is loaded by the deck. Text embedded in
illustrations remains part of the original picture. Four red annotation
rectangles are native CSS outlines.
