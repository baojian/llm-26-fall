# Lecture 04 assets

All figures are original course teaching assets. The text corpus and numerical
vectors are invented toy data. There are no model weights, student submissions,
or benchmark results in this folder.

| Files | Content and provenance |
| --- | --- |
| `context-windows.svg` and `.excalidraw` | Three sentence-local windows in the notebook's `BOS red key opens EOS` example |
| `feedforward-lm.svg` and `.excalidraw` | Embedding lookup, ordered concatenation, a tanh hidden layer, and vocabulary logits; follows the NPLM mechanism in the instructor's Spring Lecture 04, pages 17–20, and Bengio et al. (2003), §2 |
| `recurrent-state.svg` and `.excalidraw` | Shared recurrent transition across three inputs; follows the Spring lecture's recurrence explanation, pages 38–44 |
| `context-selection.svg` and `.excalidraw` | A query selects among stored encoder states; an original schematic of the motivation in Bahdanau et al. (ICLR 2015), §§2–3 |
| `qkv-path.svg` and `.excalidraw` | Learned Q/K/V projections, score normalization, and value aggregation; see Vaswani et al. (2017), §3.2 |
| `attention-lm.svg` and `.excalidraw` | The checked causal head placed between embeddings and a vocabulary readout; a schematic continuation, not a trained Transformer |
| `tiny-lm-loss.json` | Editable Plotly figure sampled from notebook E02 at updates 0, 1, 5, 10, 20, 40, 60, 100, 150, and 200; includes the uniform-logit baseline `log(7)` |
| `attention-values.json` | Hand-chosen Q, K, V and position labels, matching notebook E03–E04 exactly |
| `attention-demo.json` | Initial Plotly view from `figureFor(fixture, initialState)` in `attention-demo.js`; query 2, causal mask on, original values |

The diagrams were authored as editable Excalidraw scenes and exported with
`@excalidraw/excalidraw` 0.18.1. Their dimensions, fonts, and colors follow the
existing course diagrams. Open a scene in Excalidraw to revise it, then export
its SVG alongside it. The classroom browser uses only the SVG files.

## Numerical examples

The loss curve uses CPU float32, seed 7, a seven-token vocabulary, context size
2, embedding width 4, hidden width 8, SGD learning rate 0.5, and 200 updates on
six consistent windows. Index 0 is the loss before training; later indices are
measured after that many completed updates. On the reference environment the
loss moves from about 2.172859 to 0.006996 nats. This is a fit/debugging check,
with no held-out evaluation. The automated test recomputes the curve and allows
small platform-dependent floating-point differences.

The attention fixture uses key width 2 and value width 2. Query 2 is
`(sqrt(2), 0)`, keys are `(1,0)`, `(0,1)`, `(1,1)`, and values are `(1,0)`,
`(0,2)`, `(3,1)`. Its unmasked scores are `(1,0,1)`. The masked query-2 output
is approximately `(0.731059, 0.537883)`. `Change V3` adds `(10,10)` to only
the last value. The row outline and adjacent numbers follow the selected query.

[demo.js](../demo.js) connects native buttons to the Plotly view.
[attention-demo.js](../attention-demo.js) computes scores, weights, outputs,
and the figure specification. Reset restores query, mask, and values. The
initial view is also the PDF example. All browser runtime assets are local.

## Sources

- [Spring slides](https://github.com/baojian/llm-26/blob/main/slides/lecture-04-slides/lecture-04-slides.pdf)
  and [Spring notebook](https://github.com/baojian/llm-26/blob/main/lecture-04-neural-lms/lecture-04-neural-lms.ipynb)
- [Bengio et al. (2003)](https://www.jmlr.org/papers/v3/bengio03a.html)
- [Bahdanau et al. (ICLR 2015)](https://arxiv.org/abs/1409.0473)
- [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)
- [CS336 Lecture 2](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)
  and [Assignment 1](https://github.com/stanford-cs336/assignment1-basics)

Precise source sections and qualifications appear in the slide notes and
notebook. No screenshots of third-party figures were copied into this deck.
