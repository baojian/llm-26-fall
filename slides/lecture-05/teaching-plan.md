# Lecture 05: Attention and the Transformer

**Date:** Saturday, October 10, 2026 (Asia/Shanghai), make-up class.
Time and room follow the university notice.

**Central question:** How does attention select useful context, and how does it lead to the Transformer?

This package contains **55 slides**, five ungraded classroom exercises, an
offline CPU notebook, five editable diagrams, and an interactive position
example. It follows the shared Lecture 01–04 template: title and byline,
typography, repeating outlines, answer fragments, notes, and notebook launcher.
Preparation is tracked in [issue #259](https://github.com/baojian/llm-26-fall/issues/259).

## Learning objectives and prerequisites

Students should be able to connect the recurrent bottleneck to learned alignment,
compute attention weights and a weighted context, and trace a complete decoder block;
explain positions, residual paths, LayerNorm, and the FFN; count trainable
parameters; verify full-model causality and gradients; and interpret a
controlled component comparison with stated limits.

The first-four-lecture recap ends with RNNs, LSTMs, and their limitations.
Attention begins here: no earlier attention implementation is required.
Lecture 03 supplies embedding lookup, raw-logit cross-entropy, autograd, and
weight tying; Lecture 01 supplies BPE. We reuse the two-sentence toy corpus
from Lecture 04 and restate it in full. The published Lecture 04 deck's
attention extension is available as an additional reference.

All students have the same practices and expectations. E01–E05 are ungraded.
The public package contains no current Quiz 2 questions, reference solutions
for graded assignments, hidden tests, or student data. Quiz 2 uses the published
15-minute slot and is administered separately through the instructor workflow.

## Three-period sequence

Times include exercises. Breaks are outside the 135 minutes. These are planning
allocations; a classroom pacing rehearsal remains part of instructor review.

| Period | Minutes | Slides | Teaching and activity |
| --- | --- | --- | --- |
| 1 | 0–4 | 1–3 | Objectives and outline |
| 1 | 4–15 | 4–7 | RNN/LSTM limits, encoder bottleneck, learned alignment, and a weighted context |
| 1 | 15–28 | 8–11 | Q/K/V roles, full head equation, scaled numerical example, and masking before softmax |
| 1 | 28–40 | 12–18 | Architecture families; multiple projections; split/merge axes and code |
| 1 | 40–45 | 19 | **E01, 5 min:** compare shapes, values, and gradients against a head loop |
| 2 | 0–8 | 20–25 | Outline; permutations; sinusoidal and learned positions; browser demonstration |
| 2 | 8–13 | 26 | **E02, 5 min:** swap two tokens with and without fixed positions |
| 2 | 13–22 | 27–29 | Residual paths, pre/post-LN, and the feature axis |
| 2 | 22–26 | 30 | **E03, 4 min:** normalize a vector and calculate a residual sum |
| 2 | 26–36 | 31–34 | Position-wise FFN, complete block, and parameter ledger |
| 2 | 36–40 | 35 | Assemble the complete decoder before counting it |
| 2 | 40–45 | 36 | **E04, 5 min:** count the tied model |
| 3 | 0–8 | 37–44 | Outline; shifted targets and visibility; attention uses; toy configuration and checks |
| 3 | 8–14 | 45 | **E05, 6 min:** full-model perturbations and input-state gradients |
| 3 | 14–18 | 46–48 | Verification evidence, irreducible toy loss, and fitting loop |
| 3 | 18–22 | 49–50 | Prepared normalization comparison and its limits |
| 3 | 22–25 | 51–52 | Attention storage and the baseline's differences from the 2017 model |
| 3 | 25–30 | 53–55 | Recap, published course dates, readings, and questions |
| 3 | 30–45 | Separate private material | **Quiz 2, 15 min** |

Speaker-note allocations sum to **45 + 45 + 30 minutes of teaching**.
The five exercises are included in those totals. The quiz adds 15 minutes,
for the published 135-minute class. This is a timed plan, not an observed
classroom rehearsal.

Preserve the attention walkthrough and exercise prompts if discussion runs
long. The BERT contrast (slide 41) and normalization comparison (slides 49–50)
can move to the supplied reading and notebook, freeing five minutes.
RoPE and historical translation benchmarks are already in optional reading.
Use the prepared curve if live execution is delayed. The full notebook code
is supplied: activities ask for predictions and checks, rather than typing
the model from scratch. The FFN isolation check in E03 can be revisited after
slide 32. E05's training cells support the discussion after its timed
causality activity.

## Notebook correspondence

The [notebook](lecture-05-exercise.ipynb) has no saved outputs. Students predict
before executing the next cell or revealing the slide answer.

| Exercise | Slide / ID | Notebook code cells | Expected evidence |
| --- | --- | --- | --- |
| E01 | 19 / `exercise-01` | `multi-head-attention`, `e01-trace`, `e01-wrong-reshape` | Split Q and scores `(2,4,4,4)`, joined output `(2,4,16)`; float64 forward and gradient errors below 1e-10; wrong reshape fails values despite equal shapes |
| E02 | 26 / `exercise-02` | `e02-permutation` | Unmasked restored outputs agree without positions; fixed sinusoidal slots yield max difference about 1.36505 |
| E03 | 30 / `exercise-03` | `e03-block` | Mean 2, variance 1, normalized values approximately ±0.999995; residual `(3,2)`; normalize the feature axis |
| E04 | 36 / `exercise-04` | `decoder-model`, `e04-count` | 2,112 parameters per block; 4,432 for the tied model, 4,544 untied; tying reuses the same Parameter |
| E05 | 45 / `exercise-05` | `e05-causality`, `e05-fit` | Logits `(2,4,7)`; future perturbation leaves prefix logits unchanged; zero future-state gradient; an unmasked control leaks; reference-run mean loss below 0.20 after 200 updates |

`alignment-example` reproduces slide 7 before E01: scores `(0,1)`, weights
about `(0.2689,0.7311)`, and context `(0.7311,0.2689)`.

`single-head-example` and `single-head-mask` reproduce slides 10–11:
scaled scores `(1,0,1)` yield weights `(0.4223,0.1554,0.4223)` and output
`(1.2670,0.7330)`. Masking future slot 2 gives weights `(0.7311,0.2689,0)`
and output `(0.7311,0.5379)`. Perturbing the future value has no effect;
future key/value gradients are zero. The unmasked control changes, and
zeroing a weight after softmax leaves a row sum below one.

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
The reference CPU run meets this criterion after 200 updates; the exact plateau
exit can vary across platforms. Report the observed result when reproducing it.
There is no held-out set, so this run supports fitting and debugging claims
only. The [asset notes](assets/README.md) record the measured results and
reproduction procedure, including the loss spikes.

## Map from all 85 source slides

The instructor supplied `lecture-05-Transformers-I/lecture-05-slides.pptx`,
titled *Lecture 05 – Attention and Transformers*, dated October 15, 2025.
All 85 slides were extracted and inspected in a native PowerPoint PDF export.
Ranges below are inclusive and cover each source slide once. Destination
numbers refer to this 55-slide version. Optional material is retained in
[further reading](optional-reading.md).

| 2025 source slide(s) | Treatment and destination |
| --- | --- |
| 1–2 | Replace course/date/title and outlines; current 1–3, 20, 37 |
| 3–8 | Retain recurrent bottleneck and learned alignment; 4–7, 12, 40 and reading “From a context vector to attention” |
| 9 | Replace repeated outline with the shared active-topic outline |
| 10–16 | Move noisy-signal weighted-average analogy to optional reading |
| 17 | Replace repeated outline with the shared active-topic outline |
| 18–20 | Condense proximity-versus-context examples into slide 8 notes and contextualization reading |
| 21 | Treat embedding lookup as a Lecture 03 prerequisite; omit invented semantic-coordinate labels |
| 22–28 | Introduce Q/K/V and supply transparent numerical examples; 7–11; keep the invented bank/river permutation example in 25–26 |
| 29 | Correct dictionary lookup and compatible Q/K dimensions; 9 notes |
| 30–31 | Teach one head from the beginning; 8–11 and the new notebook walkthrough |
| 32–35 | Retain multi-head motivation and projections; 13–19; remove guaranteed roles and parameter-growth implication |
| 36 | Condense recap into 9 and 14 |
| 37 | Replace repeated outline with the shared active-topic outline |
| 38–41 | Retain original architecture context; 12, 14–15, 35, 40, 52; draw the implemented decoder explicitly |
| 42 | Correct requirement for pretrained embeddings; 24, notebook, tokenization reading |
| 43–50 | Refer BPE and merge application to Lecture 01; tokenization reading |
| 51 | Consolidate repeated architecture diagram; 35 |
| 52–54 | Retain and correct sinusoidal formula; 21–23, E02 |
| 55 | Learned and sinusoidal positions in 21–26; RoPE derivation and NoPos caveat in optional reading |
| 56–57 | Consolidate architecture; retain jointly trained token/position tables; 24, 35 |
| 58 | Retain teacher forcing and shifted targets using the shared toy corpus; 38–39 |
| 59–61 | Teach scaling and mask-before-softmax explicitly; 9–11, 18, 39, E05 |
| 62–63 | Move original attention visualizations to reading with interpretation limits |
| 64 | Retain head-pruning study as optional reading, scoped to its experiments |
| 65–67 | Retain residuals and LayerNorm; 27–30, 33; distinguish pre/post-LN and avoid stability guarantees |
| 68–69 | Retain position-wise FFN; 31–33 and notebook isolation check |
| 70–71 | Complete readout and loss path; 35, 43, 48 |
| 72–74 | Preserve translation context in optional historical reading |
| 75 | Correct speed/FLOPs and ensemble comparison in optional historical reading |
| 76 | State exact notebook configuration and differences from 2017; 34, 42, 52 |
| 77 | Replace old quiz with ungraded E04; explicit bias/norm/tying conventions; current quiz stays private |
| 78 | Keep original architecture-variation results in §6.2 reading |
| 79–80 | Keep constituency parsing in §6.3 reading |
| 81 | Retain quadratic-cost motivation in 51; Reformer as optional research reading |
| 82 | Linformer approximation in optional research reading |
| 83 | Retain transfer-of-modifications lesson in 50 and optional reading |
| 84 | Correct gradient guarantees and generation parallelism; 4, 27, 50–51 notes and reading |
| 85 | Refresh verified primary readings; 55, notebook, and optional code-reading link |

New material includes explicit tensor axes, independent forward/gradient
references, a complete executable decoder, shared-parameter counting,
full-model causality controls, the irreducible toy loss, and a measured
pre/post-LN comparison. BERT is a short encoder/objective contrast in slide 41;
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
