# Lecture 11 — Diffusion Language Models

This is the unchanged-content baseline of the instructor's
`lecture-11-diffusion-language-models.pptx`, prepared on November 18, 2026
(Asia/Shanghai). It preserves all 69 pages in their original order, including
the original course label, date, wording, equations, figures, citations, and
page numbers. Content improvements are deferred to a later revision.

Open [the Reveal.js deck](index.html) through the course server:

```sh
uv run python scripts/slides.py serve
```

Then visit <http://127.0.0.1:8000/slides/lecture-11/>. The course toolbar,
keyboard navigation, overview, speaker view, References, and Print use the
same shared framework as earlier lectures. No notebook or new exercise was
added because neither exists in the supplied deck.

## Preservation format

Microsoft PowerPoint exported the original deck to PDF. Each PDF page was
converted into a self-contained SVG with glyph outlines and embedded images,
preserving typography without requiring the original fonts on classroom
computers. These are vector page imports, not individually editable HTML
text boxes. Edit or rebuild individual pages in a later content revision.

[slides.md](slides.md) contains the original page sequence, accessible titles,
and verbatim extracted text in speaker notes. The source has no authored
speaker notes. [The manifest](assets/source-manifest.json) records source and
asset hashes, source XML page names, text, and link rectangles. Mathematical
layout is preserved in the SVG; the extracted text is an editing aid, not a
replacement for the rendered equations. Text inside source pictures remains
in those pictures.

The five original GIFs run on slides 31, 33, and 34. Slide 6 retains its
original click-to-appear illustration. Print shows the final illustration
and the original PowerPoint poster frames for GIFs. Original hyperlinks
remain clickable, including the picture link on slide 44 omitted by the
PowerPoint PDF exporter.

## Rebuild and check

Export all 69 pages with Microsoft PowerPoint for Mac, then run:

```sh
uv run --with pymupdf==1.26.7 python slides/lecture-11/import_source.py SOURCE.pptx POWERPOINT_EXPORT.pdf
npm --prefix slides run check -- lecture-11
npm --prefix slides run pdf -- lecture-11
```

The source PPTX and intermediate PDF are not committed. The importer uses
PyMuPDF only during authoring; viewing needs no Python or external network
requests. Source-preservation metadata enables complete-frame checks without
changing the layout requirements for ordinary course slides. Original small
figure labels are deliberately retained in this baseline.

Review artifacts remain under `slides/.checks/lecture-11/`. The lecture's
`check-demo.mjs` checks media hashes, hyperlink preservation, and the click
sequence against a running course server.
