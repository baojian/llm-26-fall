# First four lectures: illustrated LaTeX notes

This first draft provides a compact, illustrated review ending with recurrent
language models, LSTMs, and their limitations. Following the instructor's
September 30 revision, attention ideas begin in Lecture 05.

The [lecture notes](main.tex) contain **24 pages of main material**, including
the opening route and closing review, plus a **four-page guide to 15 papers**.
All **32 figures** are implemented in editable TikZ/PGFPlots. The text includes
worked calculations, short PyTorch excerpts, and checks for understanding.
This is a first draft for instructor review.

## Sections

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

The nine files in [sections/](sections/) hold the eight teaching sections and
the closing review. [figures/](figures/) contains one source per figure;
their F01–F32 numbering matches the reading guide and the original inventory.
The [earlier outline](outline.tex) remains available as the planning document.

Worked examples include the weighted BPE totals (25, 18, 11), fixed-vocabulary
encoding, bigram MLE and smoothing, token/byte metric conversions, PPMI,
negative-sampling gradients, shifted targets, stable cross-entropy, the
two-token neural LM, weight tying, a memory ledger, scalar recurrence,
gradient clipping, and an LSTM update with a stated gradient-path assumption.

## Reading guide

The [annotated guide](reading-guide.tex), included in the compiled PDF,
organizes the 15 papers into four groups:

1. Statistical language modeling and tokenization (R01–R03).
2. Neural language models and embeddings (R04–R07).
3. Recurrence, gradients, and memory (R08–R11).
4. Encoder–decoder models and the Lecture 05 attention handoff (R12–R15).

Each entry gives the paper's role, a reading focus, and connections to the
notes and figures. Six are recommended close readings: R03
(subword BPE), R04 (neural probabilistic LM), R06 (negative sampling), R10
(LSTM), R13 (sequence to sequence), and R14 (learned alignment). The reading
priorities guide study depth and do not add graded requirements. R14 and R15
belong to Lecture 05; the first-four recap ends with recurrent limitations.

The [BibTeX bibliography](references.bib) contains exactly these 15 readings.
The guide uses `bibentry` to render their metadata directly from that file.
The gradient-clipping paper, forget-gate extension, and subsequent Transformer
paper remain supporting links outside the selected 15. The guide preserves
the difference between the original 1997 LSTM and the later cell used in
the notes, and between the 2014 Bahdanau preprint and its 2015 conference paper.

## Build

Install a TeX distribution with latexmk, BibTeX, TikZ/PGFPlots, natbib,
bibentry, and the standard LaTeX packages used in the preamble. From the
repository root:

    make -C docs/first-four-lectures notes

This writes `output/pdf/lectures-01-04-notes.pdf`. Latexmk runs BibTeX and
the required LaTeX passes automatically. To build the earlier planning document,
use `make -C docs/first-four-lectures outline`, which writes `output/pdf/outline.pdf`.
All PDF and intermediate build files remain under ignored `output/pdf/`.
Inspect the notes with Poppler:

    mkdir -p output/pdf/first-four-lectures-preview
    pdftoppm -r 120 -png output/pdf/lectures-01-04-notes.pdf output/pdf/first-four-lectures-preview/page

## Numerical sources

Figures redraw existing public course fixtures at commit
`1adbaa5442a567cbd2ab65035b8e485778dd46a9`:

| Figure / example | Source | Interpretation |
| --- | --- | --- |
| F05, BPE totals | L01 notebook and `assets/sequence-length.json` | Exact weighted merge arithmetic |
| F07, Chinese token counts | L01 `assets/heldout-chinese.json` | Same two held-out sentences; 0, 8, 32 merges; constructed training corpora |
| F15, context association | L03 `assets/counts-ppmi.json` | Invented counts totaling 35; PPMI in bits |
| F23, left panel | L03 `assets/tiny-lm-loss.json` | Neural bigram, 8 pairs, K=10, d=4, seed 0 |
| F23, right panel | L04 `assets/tiny-lm-loss.json` | Two-token NPLM, 6 windows, K=7, d=4, H=8, seed 7 |
| L02 sample/retrain discussion | L02 `assets/self-training-loop.json` and `assets/lecture02-results.json` | Reuses the recorded run; changing token budgets prevent an isolated causal comparison |

Both F23 runs use 200 full-batch SGD updates with learning rate 0.5 on CPU.
The notes report the saved course results, with their original models and
data conventions. No new training results replace them. L02's original upstream
dataset revisions were not recorded; its asset README documents that limitation
and the available fingerprints. Other figures use exact arithmetic or clearly
identified hand-chosen values. Build-time rendering needs no dataset, model
download, or GPU.

## Source and figure policy

- Use the Fall Lecture 01–04 slides and classroom notebooks at commit
  1adbaa5 as the source baseline. The revised notes expand the RNN/LSTM
  discussion and reserves attention for Lecture 05. It does not claim that
  the older Lecture 04 deck already follows this revised sequence.
- Keep the classical-embedding bridge short. Use the existing recurrence
  reading and primary papers to develop the final two sections, with longer
  BPTT derivations in supporting reading.
- Use TikZ for conceptual diagrams and PGFPlots for numerical plots, compiled
  into vector PDF. Reuse the checked lecture fixtures and record source, units, split,
  seed, and experiment settings in captions or figure-generation metadata.
- Use readable labels at final size, consistent notation, and captions that
  state each figure's explanatory question and assumptions. Distinguish
  curves and paths with labels or line styles as well as color.
- The gradient plot shows scalar powers, not measured RNN gradients. The
  LSTM figure uses the common formulation with a forget gate and no
  peephole connections; its numerical check uses hand-chosen gate values.
- Stop before attention mechanisms, Q/K/V, attention masks, and the full
  Transformer architecture. These belong to the sequence starting in
  Lecture 05. The notes serve the same course requirements for all students.
