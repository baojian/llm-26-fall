# Scaling laws: a 90-minute board lesson

This standalone lesson develops **Kaplan → Chinchilla → Porian** through
definitions, a compute estimate, constrained optimization, and experimental
checks. It is written for teaching at a board, with an explicit writing order,
spoken explanations, questions, answers, and erasing cues.

The [LaTeX source](board-notes.tex) contains the entire lesson and its original
vector figures. It uses the same synthetic loss surface as the
[Lecture 07 notebook](../../slides/lecture-07/practice.ipynb). The timed route
totals 90 minutes, including discussion and short arithmetic exercises.
Appendices are preparation material outside those 90 minutes.

This is an alternative full lesson. The existing Lecture 07 timetable,
guest session, slide deck, and notebook exercise IDs are unchanged. The board
questions are ungraded discussion prompts, with answers included for teaching.

## Build

Install a TeX distribution with `latexmk`, TikZ, PGFPlots, and `tcolorbox`.
From the repository root:

```sh
make -C docs/lecture-07-scaling-laws
```

The PDF and TeX build intermediates go to ignored `output/pdf/`; the PDF is
`output/pdf/lecture-07-scaling-laws-board-notes.pdf`.
No Python packages, network downloads, or GPU runs are needed for the build.

## Content and sources

- **Kaplan:** empirical loss curves; distinguish the adjusted compute exponent
  0.73 from the fixed-batch analysis with an exponent near 0.88.
- **Chinchilla:** derive the optimum of an assumed two-variable loss surface;
  distinguish the approximate half/half allocation from a universal token ratio.
- **Porian:** output-head compute, warmup, and optimizer choices affect the
  fitted allocation. The cited results use arXiv v4, January 2025, of the
  paper first released in 2024.
- **Validation:** freeze a synthetic prediction before revealing a deliberately
  shifted larger-scale result. A good fit alone does not establish extrapolation.

Every historical result has a primary-source reference. All diagrams and
classroom calculations are newly authored; simulated values are labeled.
The calculus is conditional on the assumed loss model and compute constraint.
No training experiment or empirical coefficient estimation is reported here.
