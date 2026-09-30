# First four lectures: LaTeX notes plan

This draft proposes a compact, illustrated review ending with recurrent
language models, LSTMs, and their limitations. Following the instructor's
September 30 revision, attention ideas begin in Lecture 05.

The planned finished notes have about 20–25 core pages plus short supporting
appendices. This deliverable is an outline and figure plan, with three editable
vector figure prototypes.

## Proposed sections

1. Language modeling as a common task.
2. From text to tokens.
3. Count-based language models.
4. Evaluating and sampling language models.
5. Learning token representations.
6. From token IDs to a trainable neural LM.
7. Recurrent language models.
8. LSTMs and the limits of recurrent memory.

The final two sections follow one story: fixed windows, recurrent state,
training through time, difficult gradient paths, gated memory, and the
limitations that remain. The closing question motivates Lecture 05:
how can a prediction access earlier information more directly?

The [LaTeX outline](outline.tex) gives the learning questions, topic order,
equations, worked examples, scope boundaries, source map, and 32 figure or panel
specifications. Related panels can share a figure in the finished notes.
The [course map](figures/course-map.tex),
[gradient-path illustration](figures/recurrent-gradients.tex), and
[LSTM cell](figures/lstm-cell.tex) are implemented in TikZ/PGFPlots;
the remaining figures are planned.

## Build

Install a TeX distribution with latexmk, TikZ/PGFPlots, and the standard
LaTeX packages used in the preamble. From the repository root:

    make -C docs/first-four-lectures outline

The PDF and intermediate TeX files are written to the ignored output/pdf
directory. To inspect the layout with Poppler:

    mkdir -p output/pdf/first-four-lectures-preview
    pdftoppm -r 120 -png output/pdf/outline.pdf output/pdf/first-four-lectures-preview/page

## Source and figure policy

- Use the Fall Lecture 01–04 slides and classroom notebooks at commit
  1adbaa5 as the source baseline. The revised outline expands the RNN/LSTM
  discussion and reserves attention for Lecture 05. It does not claim that
  the older Lecture 04 deck already follows this revised sequence.
- Keep the classical-embedding bridge short. Use the existing recurrence
  reading and primary papers to develop the final two sections, with longer
  BPTT derivations in supporting appendices.
- Use TikZ for conceptual diagrams and generated vector PDFs for numerical
  plots. Reuse the checked lecture fixtures and record source, units, split,
  seed, and experiment settings in captions or figure-generation metadata.
- Use readable labels at final size, consistent notation, and captions that
  state each figure's explanatory question and assumptions. Distinguish
  curves and paths with labels or line styles as well as color.
- The gradient plot shows scalar powers, not measured RNN gradients. The
  LSTM prototype uses the common formulation with a forget gate and no
  peephole connections; its numerical check uses hand-chosen gate values.
- Stop before attention mechanisms, Q/K/V, attention masks, and the full
  Transformer architecture. These belong to the sequence starting in
  Lecture 05. The notes serve the same course requirements for all students.
