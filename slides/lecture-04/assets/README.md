# Lecture 04 assets

The figures include original course diagrams and two illustrations reused from
the instructor's source slides. The text corpus and numerical vectors are
invented toy data. There are no model weights, student submissions, or benchmark
results in this folder.

| Files | Content and provenance |
| --- | --- |
| `context-windows.svg` and `.excalidraw` | Three sentence-local windows in the notebook's `BOS red key opens EOS` example; compact 180×64 token boxes with centered labels |
| `feedforward-lm.svg` and `.excalidraw` | Embedding lookup, ordered concatenation, a tanh hidden layer, and vocabulary logits; follows the NPLM mechanism in the instructor's Spring Lecture 04, pages 17–20, and Bengio et al. (2003), §2 |
| `recurrent-state.svg` and `.excalidraw` | Shared recurrent transition across three inputs; follows the Spring lecture's recurrence explanation, pages 38–44 |
| `encoder-bottleneck.png` | The encoder–decoder diagram from slide 7 of the instructor's `lecture-05-slides-transformers.pptx`; extracted unchanged from `ppt/media/image10.png` (1419×230 pixels) |
| `rnn-attention.png` | The recurrent-attention diagram from page 2, “Context vector c_i,” of the 57-slide Desktop `lecture-05-slides-transformers.pptx` snapshot used on October 10; extracted unchanged from `ppt/media/image1.png` (1256×577 pixels) |
| `tiny-lm-loss.json` | Editable Plotly figure sampled from notebook E02 at updates 0, 1, 5, 10, 20, 40, 60, 100, 150, and 200; includes the uniform-logit baseline `log(7)` |

The original editable diagrams were authored with `@excalidraw/excalidraw`
0.18.1. The compact context-window layout keeps its rectangles, centered text,
and arrows synchronized between the editable scene and native SVG. Their
dimensions, fonts, and colors follow the existing course diagrams. Open a scene
in Excalidraw to revise it, then export its SVG alongside it. The bottleneck
illustration retains the original raster artwork from the PowerPoint. The
classroom browser uses the SVG and PNG assets without authoring dependencies.

The context-vector page was inspected in the open PowerPoint and the newly saved
57-slide Desktop file on October 10, 2026 (Asia/Shanghai). Its attention PNG is
identical to `ppt/media/image11.png` on page 8 of the earlier 83-slide copy.
The bottleneck provenance above refers to that earlier copy. The context-vector
page's text and formula screenshots are transcribed as editable text and KaTeX.
The figure's score is a dot product; a visible caption distinguishes its scoring
and decoder indexing from the lecture's additive model. Speaker notes map its
notation to the lecture and explain that its encoder and decoder are schematic.

## Numerical examples

The loss curve uses CPU float32, seed 7, a seven-token vocabulary, context size
2, embedding width 4, hidden width 8, SGD learning rate 0.5, and 200 updates on
six consistent windows. Index 0 is the loss before training; later indices are
measured after that many completed updates. On the reference environment the
loss moves from about 2.172859 to 0.006996 nats. This is a fit/debugging check,
with no held-out evaluation. The automated test recomputes the curve and allows
small platform-dependent floating-point differences.

The earlier causal-attention fixture, computation module, interactive controls,
and related figures remain as legacy files for existing computation tests.
They are not loaded by the revised Lecture 04 deck. Their migration belongs to
the separate Lecture 05 revision. Lecture 04 uses the source encoder–decoder
illustration to motivate alignment.
The notebook now supplies deterministic recurrence, LSTM, and additive-score
calculations in addition to the existing NPLM fitting example.

## Sources

- [Spring slides](https://github.com/baojian/llm-26/blob/main/slides/lecture-04-slides/lecture-04-slides.pdf)
  and [Spring notebook](https://github.com/baojian/llm-26/blob/main/lecture-04-neural-lms/lecture-04-neural-lms.ipynb)
- [Bengio et al. (2003)](https://www.jmlr.org/papers/v3/bengio03a.html)
- [Bahdanau et al. (ICLR 2015)](https://arxiv.org/abs/1409.0473)
- [CS336 Lecture 2](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)
  and [Assignment 1](https://github.com/stanford-cs336/assignment1-basics)

Precise source sections and qualifications appear in the slide notes and
notebook. The source PowerPoint remains with the instructor's original materials.
