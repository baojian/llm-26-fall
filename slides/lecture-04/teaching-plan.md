# Lecture 04: Neural LMs and Attention

**Date:** Wednesday, September 30, 2026

**Central question:** How can a model select useful context instead of
compressing everything into one state?

This first version contains **56 slides**, five ungraded classroom exercises,
and a companion notebook that runs offline on CPU. It follows the accepted
Lecture 01–03 Reveal.js shell, typography, section outlines, answer fragments,
speaker notes, and notebook launcher. Preparation is tracked in
[issue #234](https://github.com/baojian/llm-26-fall/issues/234).

## Learning objectives

Students should be able to:

- Extend Lecture 03's one-token `TinyLM` with a fixed-window context network.
- Trace the tensor shapes, run a training loop, and fit a consistent tiny batch.
- Explain the motivation for recurrence, gated memory, and attention.
- Compute and implement scaled dot-product attention with a causal mask.
- Verify forward outputs, gradients, and future-input independence.
- Estimate the storage of a materialized attention score matrix.

Full Transformer architecture follows in Lecture 05 on October 10. This
notebook trains an NPLM and verifies a single attention head; it does not
train a Transformer.
All students use the same practices and expectations. These practices introduce
no graded submissions, extra credit, or changes to the assessment policy.

## Three-period sequence

Times include activities. Breaks are outside the 135 teaching minutes.

| Period | Minutes | Slides | Teaching and activity |
| --- | --- | --- | --- |
| 1 | 0–5 | 1–5 | Objectives, bigram recap, and two prefixes sharing the last token |
| 1 | 5–15 | 6–10 | Vocabulary, sentence-local windows, feedforward path, nonlinearity, and shapes |
| 1 | 15–19 | 11 | **E01, 4 min:** contexts and tensor shapes |
| 1 | 19–29 | 12–15 | PyTorch forward path, raw-logit loss, numerical stability, and one update |
| 1 | 29–35 | 16 | **E02, 6 min:** repair a detached loss and fit the tiny batch |
| 1 | 35–45 | 17–20 | Loss curve, predictions, diagnosis, and fitting versus generalization |
| 2 | 0–8 | 21–26 | Fixed windows, recurrence, gradient paths, LSTM purpose, and the encoder bottleneck |
| 2 | 8–20 | 27–30 | Weighted context, Q/K/V projections, shapes, and score scaling |
| 2 | 20–25 | 31–32 | Stable row-wise softmax and the hand-chosen numerical example |
| 2 | 25–30 | 33 | **E03, 5 min:** one attention output |
| 2 | 30–42 | 34–36 | Value contributions, the matrix expression, and batched tensor operations |
| 2 | 42–45 | 37 | Interpretation limits and catch-up |
| 3 | 0–6 | 38–41 | Shifted targets, the causal mask, and mask-before-softmax |
| 3 | 6–10 | 42 | **E04, 4 min:** allowed connections and future-value changes |
| 3 | 10–14 | 43 | Interactive query, mask, and value controls |
| 3 | 14–22 | 44–47 | Learned projections, explicit reference, and output/gradient comparisons |
| 3 | 22–28 | 48 | **E05, 6 min:** implementation and correctness checks |
| 3 | 28–35 | 49–51 | Causal perturbations, failure cases, and score-matrix storage |
| 3 | 35–43 | 52–54 | Position in the LM path, order information, and exit questions |
| 3 | 43–45 | 55–56 | Published A1 reminder and selected reading |

If students need more implementation time, shorten the recurrence discussion
to its diagrams and keep the LSTM derivation in optional reading. Use the
prepared training curve if live execution is delayed. Preserve the five
exercise prompts and their checked explanations.

## Notebook correspondence

The [notebook](lecture-04-exercise.ipynb) uses E01–E05 in the slide order.
Its explanations are executable and contain no stored outputs. Predict first,
then reveal or run. Optional P01 recomputes storage/FLOPs; P02 verifies a paired
key/value permutation. Neither adds a required submission.

| Exercise | Slide | Notebook cells | Expected result |
| --- | ---: | --- | --- |
| E01 | 11 | `window-data`, `context-model` | Six sentence-local windows; lookup/flat/hidden/logit shapes `(2,2,4)`, `(2,8)`, `(2,8)`, `(2,7)` for the two-row probe |
| E02 | 16 | `broken-loop`, `fit-tiny-batch` | Detach error is explained; all six targets fit after 200 updates, with mean loss below 0.03 nats |
| E03 | 33 | `attention-numbers` | Query-2 scores `(1,0,1)`, weights about `(0.4223,0.1554,0.4223)`, output `(1.6893,0.7330)` |
| E04 | 42 | `causal-numbers` | Lower-triangular allowed mask; query-2 output `(0.7311,0.5379)` is independent of V3 |
| E05 | 48 | `attention-implementation` through `causality-check` | Forward/gradient comparisons pass, finite differences agree, and a prefix-only loss has no gradient to future inputs |

Use separate documents before extracting overlapping windows when extending
the corpus into a train/dev experiment. The provided six-window fit has **no
held-out evaluation** and must not be presented as a quality benchmark.

## Continuity with the Spring lecture

The original [84-page Lecture 04 PDF](https://github.com/baojian/llm-26/blob/main/slides/lecture-04-slides/lecture-04-slides.pdf)
and [notebook](https://github.com/baojian/llm-26/blob/main/lecture-04-neural-lms/lecture-04-neural-lms.ipynb)
were inspected for this version. The source deck is titled *Neural networks
and sequence learning*. The map below documents the teaching choices.

| Spring material | Treatment here |
| --- | --- |
| Slides 2–14: units, nonlinearities, XOR, and hidden representations | Retain the hidden-layer/nonlinearity explanation; longer logic-gate and XOR work is optional |
| Slides 15–16: feedforward text classifiers | Optional comparison; the main task stays next-token prediction |
| Slides 17–24 and notebook §3: NPLM | Retain the context window and learned embedding/hidden/readout path; simplify to plain PyTorch and a local vocabulary |
| Slides 25–35 and notebook §§1–2: training, graphs, micrograd | Reuse Lecture 03 prerequisites; add a short numerical-stability example and a broken-loop debugging exercise; link full micrograd work as optional |
| Slides 38–46, 60–73, and notebook §4: recurrence and LSTM | Retain an eight-minute motivation; link full BPTT, gates, and training as optional |
| Slides 47–59 and 74–83: historical comparisons, variants, and applications | Keep accessible in the original source; do not treat historical benchmark numbers as current model comparisons |

The new attention and verification portion adopts CS336's explicit tensor
reasoning and component-testing approach. It adds original small examples,
rather than importing a complete CS336 assignment. Relevant sources are
[Lecture 2](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)
(`tensor_einops`, gradient examples, and resource accounting) and
[Assignment 1's component contracts](https://github.com/stanford-cs336/assignment1-basics/blob/main/tests/adapters.py).
The full Transformer, GPU kernels, and optimizer/schedule experiments remain
in their later course topics.

## Browser demonstration

Slide 43 starts at query 2, with the mask on and original values. `Change V3`
adds `(10,10)` to the third value vector. This leaves query 2 unchanged until
the mask is turned off. Query 3 may use its own value. `Reset` restores query,
mask, and values. Native buttons support keyboard activation and expose their
pressed state; the chart has an updated accessible numerical description.

The matrix shows all three rows. The green outline identifies the selected
query. The right-hand values show its unmasked scores, final weights, and
output. The PDF retains the original masked example. All assets and plotting
libraries are local. See [asset provenance](assets/README.md).

## Preparation and validation

```sh
uv run python -m pytest tests/test_lecture_04.py
node --test tests/test_attention_demo.mjs
npm --prefix slides run check -- lecture-04
npm --prefix slides run pdf -- lecture-04
uv run python -m pytest tests/
```

Inspect every generated slide image and PDF page. The browser check covers
all three framework viewport sizes, query/mask/value/reset controls, and
exercise answer fragments. The tests execute the notebook with network calls
disabled and compare its numerical examples with the chart data. Review
artifacts belong under ignored `slides/.checks/lecture-04/`.

The shared course server opens the companion notebook through its normal
launcher and preserves student working copies under `workspace/`.
