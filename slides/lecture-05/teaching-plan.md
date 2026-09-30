# Lecture 05: The Transformer as a working model

**Date:** Saturday, October 10, 2026 (Asia/Shanghai), make-up class.
Time and room follow the university notice.

**Central question:** Which components turn attention into a trainable language model?

This package contains **51 slides**, five ungraded classroom exercises, an
offline CPU notebook, four editable diagrams, and an interactive position
example. It follows the shared Lecture 01–04 template: title and byline,
typography, repeating outlines, answer fragments, notes, and notebook launcher.
Preparation is tracked in [issue #259](https://github.com/baojian/llm-26-fall/issues/259).

## Learning objectives and prerequisites

Students should be able to trace multiple heads and a complete decoder block;
explain positions, residual paths, LayerNorm, and the FFN; count trainable
parameters; verify full-model causality and gradients; and interpret a
controlled component comparison with stated limits.

Lecture 04 supplies one checked causal attention head and the toy corpus.
Lecture 03 supplies embedding lookup, raw-logit cross-entropy, autograd, and
weight tying. Lecture 01 supplies BPE. Review these briefly, preserving time
for the new block and complete-model checks.

All students have the same practices and expectations. E01–E05 are ungraded.
The public package contains no current Quiz 2 questions, reference solutions
for graded assignments, hidden tests, or student data. Quiz 2 uses the published
15-minute slot and is administered separately through the instructor workflow.

## Three-period sequence

Times include exercises. Breaks are outside the 135 minutes. These are planning
allocations; a classroom pacing rehearsal remains part of instructor review.

| Period | Minutes | Slides | Teaching and activity |
| --- | --- | --- | --- |
| 1 | 0–8 | 1–6 | Objectives, attention recap, and three architecture families |
| 1 | 8–22 | 7–12 | Multiple learned views, equations, split/merge axes, and code |
| 1 | 22–27 | 13 | **E01, 5 min:** shapes, values, and gradients against a head loop |
| 1 | 27–39 | 14–18 | Permutations; sinusoidal, learned, and rotary positions |
| 1 | 39–40 | 19 | Browser prediction before toggling the controls |
| 1 | 40–45 | 20 | **E02, 5 min:** swap two tokens with and without fixed positions |
| 2 | 0–13 | 21–24 | Outline, residual paths, pre/post-LN, and the feature axis |
| 2 | 13–17 | 25 | **E03, 4 min:** LayerNorm and a residual sum |
| 2 | 17–27 | 26–29 | Position-wise FFN, block code, and parameter ledger |
| 2 | 27–32 | 30 | **E04, 5 min:** count the complete tied model |
| 2 | 32–41 | 31–33 | Assemble the decoder, shifted targets, and allowed diagonal |
| 2 | 41–45 | 34–35 | Cross-attention and the BERT masked-prediction contrast |
| 3 | 0–5 | 36–39 | Configuration, raw-logit loss, and verification plan |
| 3 | 5–11 | 40 | **E05, 6 min:** full-model perturbations and input-state gradients |
| 3 | 11–19 | 41–44 | Evidence, irreducible toy loss, fitting loop, and measured curves |
| 3 | 19–25 | 45–48 | Comparison limits, attention storage, history, and model choices |
| 3 | 25–30 | 49–51 | Recap, course dates, readings, and questions |
| 3 | 30–45 | Separate private material | **Quiz 2, 15 min** |

If discussion runs long, assign rotary positions and the historical translation
table as reading (slides 18 and 47). Use the prepared loss curve if live
execution is delayed. Preserve the five exercise prompts, their checked
explanations, and the quiz slot. The full notebook code is supplied: the short
activities ask for predictions and checks, not typing the model from scratch.
The extra FFN isolation check in E03 can be revisited after slide 27. E05's
training cells support the discussion after its timed causality activity.

## Notebook correspondence

The [notebook](lecture-05-exercise.ipynb) has no saved outputs. Students predict
before executing the next cell or revealing the slide answer.

| Exercise | Slide / ID | Notebook code cells | Expected evidence |
| --- | --- | --- | --- |
| E01 | 13 / `exercise-01` | `multi-head-attention`, `e01-trace`, `e01-wrong-reshape` | Split Q and scores `(2,4,4,4)`, joined output `(2,4,16)`; float64 forward and gradient errors below 1e-10; wrong reshape fails values despite equal shapes |
| E02 | 20 / `exercise-02` | `e02-permutation` | Unmasked restored outputs agree without positions; fixed sinusoidal slots yield max difference about 1.36505 |
| E03 | 25 / `exercise-03` | `e03-block` | Mean 2, variance 1, normalized values approximately ±0.999995; residual `(3,2)`; normalize the feature axis |
| E04 | 30 / `exercise-04` | `decoder-model`, `e04-count` | 2,112 parameters per block; 4,432 for the tied model, 4,544 untied; tying reuses the same Parameter |
| E05 | 40 / `exercise-05` | `e05-causality`, `e05-fit` | Logits `(2,4,7)`; future perturbation leaves prefix logits unchanged; zero future-state gradient; an unmasked control leaks; mean loss below 0.20 after 200 updates |

`ln-comparison` changes only block norm order from the same initial state.
Both variants keep the final LayerNorm. Thus the post-LN comparison is not a
full reproduction of the original 2017 architecture.

## Exact teaching model

| Setting | Value |
| --- | --- |
| Vocabulary | `BOS, red, blue, key, opens, closes, EOS`, IDs 0–6 |
| Documents | `[0,1,3,4,6]` and `[0,2,3,5,6]` |
| Input / target | Each document's first / last four tokens; no padding or cross-document windows |
| Shape | Batch 2, length 4, width 16, 4 heads of width 4 |
| Blocks | 2; pre-LN; ReLU FFN with inner width 32 |
| Positions | Learned absolute table, 4 × 16; no embedding scaling |
| Norms | Last feature axis; affine scale and bias; epsilon 1e-5; final LayerNorm |
| Linear layers | No biases; output reuses the token table |
| Regularization | No dropout, label smoothing, weight decay, or gradient clipping |
| Initialization | Seed 7; embedding/Linear weights normal with standard deviation 0.02; norm scale 1 and bias 0 |
| Training | CPU float32, full batch, 200 AdamW updates; learning rate 0.02, betas (0.9, 0.999), epsilon 1e-8 |
| References | Float64 head comparisons with 1e-10 tolerance; full-model causality also checked independently in float64 |

One block has `4*d*d + 2*d*d_ff + 4*d = 2112` parameters. Two blocks,
token/position tables, and the final norm give `4224 + 112 + 64 + 32 = 4432`.
The shared readout adds no parameters. The count stays constant when only the
number of equal-width heads changes at fixed model width.

The identical `BOS` prefix has two equally frequent next tokens. The best mean
loss is bounded below by `log(2)/4 ≈ 0.1733` nats per token. Require correct
predictions on the six deterministic targets and near-balanced probabilities
after BOS; do not require zero loss or perfect top-1 accuracy on all eight.
There is no held-out set, so this run supports fitting and debugging claims
only. The [asset notes](assets/README.md) record the measured results and
reproduction procedure, including the loss spikes.

## Map from all 85 source slides

The instructor supplied `lecture-05-Transformers-I/lecture-05-slides.pptx`,
titled *Lecture 05 – Attention and Transformers*, dated October 15, 2025.
All 85 slides were extracted and inspected in a native PowerPoint PDF export.
Ranges below are inclusive and cover each source slide once. Destination
numbers refer to this 51-slide version. Optional material is retained in
[further reading](optional-reading.md).

| 2025 source slide(s) | Treatment and destination |
| --- | --- |
| 1–2 | Replace course/date/title and outline; current 1–3, 21, 36 |
| 3–8 | Condense encoder–decoder, recurrent bottleneck, and context-vector review; 6, 34 and reading “From a context vector to attention” |
| 9 | Replace repeated outline with the shared active-topic outline |
| 10–16 | Move noisy-signal weighted-average analogy to reading; brief reminder in slide 4 notes |
| 17 | Replace repeated outline with the shared active-topic outline |
| 18–20 | Condense proximity-versus-context examples; 4 and contextualization reading |
| 21 | Treat embedding lookup as Lecture 03 prerequisite; omit invented semantic-coordinate labels |
| 22–28 | Retain `bank of the river` and Q/K/V mechanism; 4–5, 14, 19–20; use explicitly invented browser vectors |
| 29 | Correct dictionary lookup and dimension statement; 5 notes |
| 30–31 | Reuse checked single-head implementation from Lecture 04; 5 and E01 bridge |
| 32–35 | Retain multi-head motivation and projections; 7–13; remove guaranteed role/parameter-growth implication |
| 36 | Condense recap into 5 and 8 |
| 37 | Replace repeated outline with the shared active-topic outline |
| 38–41 | Retain original architecture context; 6, 8–9, 31, 34, 48; draw the implemented decoder explicitly |
| 42 | Correct requirement for pretrained embeddings; 17, notebook, tokenization reading |
| 43–50 | Refer BPE and merge application to Lecture 01; tokenization reading |
| 51 | Consolidate repeated architecture diagram; 31 |
| 52–54 | Retain and correct sinusoidal formula; 14–16, E02 |
| 55 | Learned/sinusoidal/RoPE comparison in 14–20; NoPos caveat in reading |
| 56–57 | Consolidate architecture; retain jointly trained token/position tables; 17, 31 |
| 58 | Retain teacher forcing and shifted targets using the shared toy corpus; 32–33 |
| 59–61 | Recap scale, mask-before-softmax, and diagonal; 5, 12, 33, E05 |
| 62–63 | Move original attention visualizations to reading with interpretation limits |
| 64 | Retain head-pruning study as optional reading, scoped to its experiments |
| 65–67 | Retain residuals and LayerNorm; 22–25, 28; distinguish pre/post-LN and avoid stability guarantees |
| 68–69 | Retain position-wise FFN; 26–28 and notebook isolation check |
| 70–71 | Complete readout and loss path; 31, 38, 43 |
| 72–74 | Keep translation context; 47 and historical reading; avoid a long benchmark survey |
| 75 | Correct speed/FLOPs and ensemble comparison; 47 and historical reading |
| 76 | State exact notebook configuration and differences from 2017; 29, 37, 48 |
| 77 | Replace old quiz with new ungraded E04; explicit bias/norm/tying conventions; current quiz stays private |
| 78 | Keep original architecture-variation results in §6.2 reading |
| 79–80 | Keep constituency parsing in §6.3 reading |
| 81 | Retain quadratic-cost motivation in 46; Reformer as optional research reading |
| 82 | Linformer approximation in optional research reading |
| 83 | Retain transfer-of-modifications lesson in 45 and optional reading |
| 84 | Correct gradient guarantees and generation parallelism; 22, 45–46 notes and reading |
| 85 | Refresh verified primary readings; 51, notebook, and optional code-reading link |

New material includes explicit tensor axes, independent forward/gradient
references, a complete executable decoder, shared-parameter counting,
full-model causality controls, the irreducible toy loss, and a measured
pre/post-LN comparison. BERT is a short encoder/objective contrast in slide 35;
removing a causal mask while retaining next-token targets would leak targets.

## Corrections carried into the presentation

- Source 29: `{'a': 1, 'b': 2, 'c': 3}['b']` is 2. Q/K widths must agree;
  the value width can differ.
- Source 32: equal-width heads partition the projections at fixed model width;
  they do not multiply the `4*d*d` parameter count or guarantee grammatical roles.
- Sources 42/57: token embeddings can be initialized randomly and trained
  jointly; separately pretrained vectors are optional.
- Source 54: each frequency has one sine and one cosine coordinate, with
  consistent zero-based indices.
- Sources 60/61: give the scale's variance assumptions, mask before softmax,
  and allow the diagonal when inputs and targets are shifted.
- Sources 66/67/84: normalization order changes the function; normalize over
  features; residuals and norms do not guarantee stable gradients.
- Source 75: estimated training FLOPs are not measured wall-clock speed.
- Source 84: known training positions can be parallelized; autoregressive
  generation still proceeds through dependent next-token steps.

## Preparation and verification

```sh
uv sync --locked
uv run python -m pytest tests/test_lecture_05.py
node --test tests/test_position_demo.mjs
npm --prefix slides run check -- lecture-05
npm --prefix slides run pdf -- lecture-05
uv run python -m pytest tests/
```

Inspect every slide image and PDF page. Browser checks cover three viewport
sizes, all four position/swap states, reset, keyboard activation, and answer
fragments. Numerical tests execute the notebook offline, compare head outputs
and gradients against PyTorch, and verify the model's count and causality.
The measured curve preserves every update rather than hiding transient spikes.

Use `uv run python scripts/slides.py serve` for the shared Notebook launcher.
It creates a personal working copy under `workspace/` and preserves later
edits. Review PDFs, screenshots, execution records, and reports belong under
ignored `slides/.checks/lecture-05/`. Publication and instructor pacing review
follow the preparation PR.
