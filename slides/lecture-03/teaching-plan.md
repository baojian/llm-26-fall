# Lecture 03 teaching plan

**Date:** Wednesday, September 23, 2026

**Central question:** How do discrete tokens become trainable representations?

The core deck has **50 slides**. Students follow one computation from token
IDs to next-token loss, train a small model, and reason about its parameters
and memory. The distributional hypothesis and one skip-gram example explain
how an objective learns vectors. Longer classification and static-embedding
material is in [optional reading](classical-reading.md), with links to the
complete Spring lecture.

## Learning objectives

By the end, students can:

- Trace lookup, context states, output logits, shifted targets, and loss shapes.
- Run a PyTorch training step and inspect gradient pathways.
- Distinguish token embeddings, contextual states, and retrieval vectors.
- Explain weight tying and count unique table parameters and storage.
- Inspect configuration and tokenizer metadata without downloading weights.

The running `TinyLM` is explicitly a **bigram model with a vector bottleneck**.
It does not use the full prefix. A feedforward context network is introduced
next week; Transformer architecture comes later in the course.

## Three-period sequence

Times include the exercises. Breaks are outside the 135 teaching minutes.

| Period | Minutes | Slides | Teaching and activity |
| --- | --- | --- | --- |
| 1 | 0–10 | 1–6 | Recall n-gram prediction; introduce the token-to-loss shapes and the three representation types |
| 1 | 10–22 | 7–9 | Trainable lookup, integer IDs, one-hot equivalence; **E01, 4 min** |
| 1 | 22–38 | 10–15 | Distributional intuition, one skip-gram example, negative sampling; **E02, 4 min**, followed immediately by autograd |
| 1 | 38–45 | 16–17 | Similarity limitations and context-dependent states |
| 2 | 0–12 | 18–21 | Shift sequences into next-token pairs; **E03, 4 min** |
| 2 | 12–26 | 22–29 | Explicit bigram limitation, output projection, raw-logit cross-entropy, uniform baseline, numerical stability |
| 2 | 26–40 | 30–33 | Inspect gradients, run SGD, and read the toy loss curve; **E04, 6 min** |
| 2 | 40–45 | 34 | Debugging checks and the distinction between fitting and generalization |
| 3 | 0–10 | 35–38 | Weight tying and its gradient pathways; **E05, 4 min** |
| 3 | 10–23 | 39–45 | Unique parameters, pinned Qwen configurations, tokenizer IDs versus rows, memory and projection cost; **E06, 4 min** |
| 3 | 23–30 | 46–50 | Retrieval preview, optional student contribution, exit questions, readings |
| 3 | 30–45 | — | **Quiz 1**, 15 minutes, as published on the course website |

Quiz content and grading remain in the private instructor repository. E01–E06
and P01–P03 are **ungraded practices** and use no Assignment A1 data or code.
The same activities and expectations apply to all students.

## Notebook correspondence

The notebook runs on a CPU with the core `uv sync` environment and makes no
network requests. Predict the result before running each cell. There are
26 minutes of timed pair practice within the schedule above.

| ID | Slide ID | Expected response or computation |
| --- | --- | --- |
| E01 | `exercise-01` | Table `(10,4)`, IDs `(2,3)`, lookup `(2,3,4)`, one-hot `(2,3,10)`; product equals lookup |
| E02 | `exercise-02`, `autograd` | Positive-context gradient `(-0.2689,-0.1345)`; full loss 1.2873; autograd agrees with the hand calculation |
| E03 | `exercise-03` | Last targets 5 and 9; embeddings `(2,4,4)`, logits `(2,4,10)`; eight predictions; zero-logit loss `log(10)` |
| E04 | `exercise-04`, `training-curve` | Run 200 SGD updates on eight compatible pairs; final loss below 0.1 and all predictions correct |
| E05 | `exercise-05`, `tied-gradient` | Untied input gradients only in selected rows; tied example has gradients in all ten rows; shared gradient equals the sum of the two separate contributions |
| E06 | `exercise-06` | 16,384,000 parameters per exercise table; 31.25 MiB in bf16; verify real-model counts and logit storage |
| P01 | optional | PPMI from the published word–context counts; information/data is about 0.0944 bits |
| P02 | optional | Train skip-gram on templates and inspect neighbors/analogies; these are toy results |
| P03 | optional | Check a model-table row using pinned configuration/tokenizer metadata and cross-review another pair's evidence |

Students with an older working notebook should rename it before clicking the
Notebook link again. The launcher intentionally preserves their existing
work. It copies the supplied assets into a fresh working notebook directory.

## CS336 connections and scope

[CS336 Lecture 2](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)
is the closest match: tensors, memory accounting, gradients, and the training
loop. This lecture introduces shapes, bytes per value, unique parameter counts,
and the size of materialized logits. It includes a brief dense-projection FLOP
estimate; detailed roofline analysis, GPU kernels and distributed execution
remain for later weeks.

We retain sigmoid/softmax and cross-entropy as prerequisites. We do not repeat
the complete Naive Bayes/logistic-regression sequence. Full word2vec gradient
walkthroughs, PPMI arithmetic, analogy benchmarks, GloVe/SVD/fastText, and document
embedding history are optional. TF-IDF returns with retrieval. A short retrieval
preview explains why a document vector differs from a token-table row.

## Mapping from the first draft in PR #157

This supersedes the first-version requirement to reproduce every Spring slide
unchanged. The instructor requested compression and a stronger connection to
modern LMs and CS336 after reviewing that draft.

| First-draft slides | Revised treatment |
| --- | --- |
| 3–32: classification | Logits, softmax and cross-entropy integrated into slides 25–29; other background in optional reading |
| 34–57: meaning and counts | Slides 6–10; detailed PPMI moved from old E01 to P01 |
| 59–100: word2vec | Slides 11–15, one example and immediate autograd; full diagrams linked from optional reading |
| 101–122: evaluation and static methods | Slide 16 plus optional reading and P02 |
| 124–127: retrieval models | Slides 6, 17 and 46 distinguish lookup, contextual states and retrieval representations |
| 129–136: PyTorch | Expanded across the lecture, including shifted targets, training and resource accounting |
| Old E02 | Remains E02; only one gradient is required by hand |
| Old E03 | Lookup becomes E01; loss becomes E03; tied-gradient comparison is new E05 |
| Old E04 | Becomes E06, with pinned Qwen3-0.6B and Qwen3-8B evidence |
| Old P01–P02 | Combined into optional P02; old P03 training loop becomes core E04 |

## Technical points to emphasize

- Uniform logits give loss exactly `log(V)`; arbitrary random initialization
  need not. Use the deliberately zero output table for the controlled check.
- Use `F.logsigmoid` for skip-gram and raw logits with `F.cross_entropy` for
  next-token training. The notebook demonstrates float32 cancellation.
- Untied lookup and tied input/output weights have different gradient paths.
  The gradient experiment keeps forward values identical to isolate tying.
- The tiny-batch check requires labels compatible with the model's context and
  enough capacity. Successful fitting does not establish generalization.
- Distinguish tokenizer entries, maximum token ID, and allocated table rows.
  Use the latter for parameter counts. Do not count added-token IDs twice.
- Parameter storage excludes gradients, optimizer state and activations. Logit
  storage is an illustrative materialization cost, not a peak-memory promise.

## Sources and reproducibility

Sources are in the slide notes, notebook, optional reading and
[asset provenance](assets/README.md). The selected model metadata records
immutable revisions and SHA-256 hashes; no weight files are stored.
`tests/test_lecture_03.py` executes every notebook code cell without network
access, checks the mathematical examples and teaching-figure data, and tests
that the asset lookup works from a copied notebook directory.

The complete model timeline belongs to [issue #154](https://github.com/baojian/llm-26-fall/issues/154)
as a separate document. P03 invites one verified contribution at a time,
with a combined **Sources and Notes** column and peer checking.
