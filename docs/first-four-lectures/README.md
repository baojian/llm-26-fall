# First four lectures: LaTeX notes plan

This draft proposes the structure of a compact, illustrated review before the
full Transformer paper. It includes the single-head attention, causal masking,
and verification covered in Lecture 04.

The planned finished notes have about 20–25 core pages plus short supporting
appendices. This deliverable is an outline and figure plan, with two editable
vector figure prototypes.

## Proposed sections

1. Language modeling as a common task.
2. From text to tokens.
3. Count-based language models.
4. Evaluating and sampling language models.
5. Learning token representations.
6. From token IDs to a trainable neural LM.
7. Representing longer context.
8. Attention, causality, and verification.

The [LaTeX outline](outline.tex) gives the learning questions, topic order,
equations, worked examples, scope boundaries, source map, and 32 figure or panel
specifications. Related panels can share a figure in the finished notes.
The [course map](figures/course-map.tex) and
[attention example](figures/attention-example.tex) are implemented in
TikZ/PGFPlots; the remaining figures are planned.

## Build

Install a TeX distribution with latexmk, TikZ/PGFPlots, and the standard
LaTeX packages used in the preamble. From the repository root:

    make -C docs/first-four-lectures outline

The PDF and intermediate TeX files are written to the ignored output/pdf
directory. To inspect the layout with Poppler:

    mkdir -p output/pdf/first-four-lectures-preview
    pdftoppm -r 120 -png output/pdf/outline.pdf output/pdf/first-four-lectures-preview/page

## Source and figure policy

- Base coverage on the published Fall Lecture 01–04 slides and classroom
  notebooks. The outline records source commit 1adbaa5.
- Keep the classical-embedding and recurrent-model bridges proportional to
  the material actually taught; use the longer reading for appendices.
- Use TikZ for conceptual diagrams and generated vector PDFs for numerical
  plots. Reuse the checked lecture fixtures and record source, units, split,
  seed, and experiment settings in captions or figure-generation metadata.
- The prototype attention plot uses the hand-chosen values in
  [Lecture 04's fixture](../../slides/lecture-04/assets/attention-values.json).
- Stop before multi-head composition, positional-encoding formulas,
  residual/normalization/feedforward blocks, and the complete Transformer
  architecture. The notes serve the same course requirements for all students.
