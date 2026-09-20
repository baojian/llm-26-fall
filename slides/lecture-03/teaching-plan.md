# Lecture 03: Embeddings and PyTorch for LLMs

**Date:** Wednesday, September 23, 2026

**Central question:** How do discrete tokens become trainable representations?

The core deck has **60 slides**, including **exactly ten classical-bridge
slides (11–20)**. Students follow one computation from token
IDs to next-token loss, train a small model, and reason about its parameters
and memory. The classical bridge explains context counts, PPMI, SVD/GloVe,
CBOW/skip-gram, one checked gradient, fastText, and the limits of static-vector
similarity. It connects these ideas to current token tables and contextual
states. Longer derivations and classification history remain in
[optional reading](classical-reading.md), with links to the complete Spring
lecture. The instructor requested this bounded expansion on September 20.

## Learning objectives

By the end, students can:

- Trace lookup, context states, output logits, shifted targets, and loss shapes.
- Run a PyTorch training step and inspect gradient pathways.
- Explain what the classical objectives preserve and what contextual states add.
- Evaluate a held-out coverage probe without modifying parameters or gradients.
- Distinguish token embeddings, contextual states, and retrieval vectors.
- Explain weight tying and account for parameter, gradient, and optimizer storage.
- Inspect configuration and tokenizer metadata without downloading weights.

The running `TinyLM` is explicitly a **bigram model with a vector bottleneck**.
It does not use the full prefix. A feedforward context network is introduced
next week; Transformer architecture comes later in the course.

## Three-period sequence

Times include the exercises. Breaks are outside the 135 teaching minutes.

| Period | Minutes | Slides | Teaching and activity |
| --- | --- | --- | --- |
| 1 | 0–8 | 1–6 | Recall n-gram prediction; introduce the token-to-loss shapes and the three representation types |
| 1 | 8–18 | 7–10 | Trainable lookup, one-hot equivalence, dtype and device; **E01, 4 min** |
| 1 | 18–43 | 11–20 | Ten-slide classical bridge listed below; **E02, 4 min**, followed immediately by autograd |
| 1 | 43–45 | 21 | Contrast a static lookup with a context-dependent state |
| 2 | 0–10 | 22–25 | Shift sequences into next-token pairs; **E03, 4 min** |
| 2 | 10–24 | 26–34 | Bigram limitation, output projection, raw-logit cross-entropy, safe reshaping, uniform baseline, numerical stability |
| 2 | 24–38 | 35–38 | Inspect gradients, run SGD, and read the toy loss curve; **E04, 6 min** |
| 2 | 38–45 | 39–41 | Run the held-out coverage probe; distinguish evaluation mode, gradient recording, fitting, and generalization |
| 3 | 0–9 | 42–45 | Weight tying and its gradient pathways; **E05, 4 min** |
| 3 | 9–26 | 46–55 | Unique parameters, pinned Qwen configurations, tokenizer rows, fp32 training-memory ledger, measured optimizer state, projection FLOPs; **E06, 4 min** |
| 3 | 26–30 | 56–60 | Retrieval preview, optional student contribution, exit questions; leave reading slides for reference |
| 3 | 30–45 | — | **Quiz 1**, 15 minutes, as published on the course website |

Quiz content and grading remain in the private instructor repository. E01–E06
and P01–P03 are **ungraded practices** and use no Assignment A1 data or code.
The same activities and expectations apply to all students.

## The ten-slide classical bridge

Keep this block to about 25 minutes including E02. Explain the transferable
idea on each slide; optional reading supplies the full derivations. Static
embeddings remain useful baselines and components, while the core course goal
is to understand token lookup inside a trainable language model.

| Slide | Topic | Connection carried forward |
| ---: | --- | --- |
| 11 | Distributional hypothesis and toy counts | Learn from context statistics |
| 12 | PMI and PPMI | Compare association with a frequency-aware baseline |
| 13 | Truncated SVD and GloVe | Compression and the choice of training objective |
| 14 | CBOW versus skip-gram | Specify what predicts what; distinguish causal training |
| 15 | Center and context tables | Parameters can serve different input/output roles |
| 16 | Negative sampling | Distinguish pair discrimination from normalized next-token likelihood |
| 17 | E02: one gradient | Connect a loss to a parameter update |
| 18 | The same gradient with autograd | Verify the hand calculation with executable evidence |
| 19 | fastText and character n-grams | Compare subword composition with an LLM tokenizer |
| 20 | Similarity and analogy limitations | Inspect vectors without mistaking geometry for an LM evaluation |

The following contextual-state slide closes the bridge into modern LMs; it
belongs to the 50 other slides. No Naive Bayes/LR survey, full word2vec backward
derivation, or catalog of benchmark scores is restored to the classroom deck.

## Notebook correspondence

The notebook runs on a CPU with the core `uv sync` environment and makes no
network requests. Predict the result before running each cell. There are
26 minutes of timed pair practice within the schedule above.

| ID | Slide ID | Expected response or computation |
| --- | --- | --- |
| E01 | `exercise-01` | Table `(10,4)`, IDs `(2,3)`, lookup `(2,3,4)`, one-hot `(2,3,10)`; product equals lookup; integer IDs and floating-point weights |
| Bridge demo | `count-to-ppmi`, `count-to-dense` | Invented counts total 35; PMI(tea, drink) is 0.4150 bits; truncated SVD produces a `(3,2)` word table |
| E02 | `exercise-02`, `autograd` | Positive-context gradient `(-0.2689,-0.1345)`; full loss 1.2873; autograd agrees with the hand calculation |
| E03 | `exercise-03`, `reshape-batch` | Last targets 5 and 9; embeddings `(2,4,4)`, logits `(2,4,10)`; eight predictions; zero-logit loss `log(10)`; reshape preserves label order |
| E04 | `exercise-04`, `training-curve`, `evaluation-step` | Run 200 SGD updates on eight compatible pairs; final loss below 0.1; evaluate unseen input IDs 0 and 9 without an update |
| E05 | `exercise-05`, `tied-gradient` | Untied input gradients only in selected rows; tied example has gradients in all ten rows; shared gradient equals the sum of the two separate contributions |
| E06 | `exercise-06`, `training-memory`, `inspect-training-state`, `projection-work` | 16,384,000 parameters per exercise table; 31.25 MiB in bf16; 250 MiB fp32 Adam/gradient/parameter subtotal; inspect actual small-model state; toy projection is about 640 forward FLOPs |
| P01 | optional | PPMI from the published word–context counts; information/data is about 0.0944 bits |
| P02 | optional | Train skip-gram on templates and inspect neighbors/analogies; these are toy results |
| P03 | optional | Check a model-table row using pinned configuration/tokenizer metadata and cross-review another pair's evidence |

Students with an older working notebook should rename it before clicking the
Notebook link again. The launcher intentionally preserves their existing
work. It copies the supplied assets into a fresh working notebook directory.

## CS336 connections and scope

[CS336 Lecture 2](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)
is the closest match: tensors, memory accounting, gradients, and the training
loop. This lecture introduces shape/dtype/device contracts, non-contiguous
target slices, unique parameter counts, and the size of materialized logits.
Students see an explicit training-memory ledger and inspect initialized Adam
state. A worked dense-projection example distinguishes stored parameters from
repeated computation. Detailed roofline analysis, GPU kernels and distributed
execution remain for later weeks.

We retain sigmoid/softmax and cross-entropy as prerequisites. We do not repeat
the complete Naive Bayes/logistic-regression sequence. Full word2vec gradient
walkthroughs, extended PPMI arithmetic, analogy benchmarks, and document
embedding history are optional. SVD, GloVe, CBOW, and fastText receive bounded
coverage in the ten-slide bridge. TF-IDF returns with retrieval. A short retrieval
preview explains why a document vector differs from a token-table row.

## Mapping from the first draft in PR #157

This supersedes the first-version requirement to reproduce every Spring slide
unchanged. The instructor requested compression and a stronger connection to
modern LMs and CS336 after reviewing that draft.

| First-draft slides | Revised treatment |
| --- | --- |
| 3–32: classification | Logits, softmax and cross-entropy integrated into slides 29–34; other background in optional reading |
| 34–57: meaning and counts | Lookup in slides 6–10; toy counts, PPMI, and compression in 11–13; richer count exercise remains P01 |
| 59–100: word2vec | Slides 14–18, one example and immediate autograd; full diagrams linked from optional reading |
| 101–122: evaluation and static methods | SVD/GloVe in 13, fastText in 19, similarity limits in 20; extended reading and P02 |
| 124–127: retrieval models | Slides 6, 21 and 56 distinguish lookup, contextual states and retrieval representations |
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
- The two held-out pairs are selected coverage probes for unseen input IDs,
  not representative corpus evaluation. `eval()` and `no_grad()` serve
  different purposes; neither takes an optimizer step.
- Distinguish tokenizer entries, maximum token ID, and allocated table rows.
  Use the latter for parameter counts. Do not count added-token IDs twice.
- The fp32 ledger explicitly includes parameters, gradients and two Adam moment
  tensors, but excludes activations, scalar counters and temporary buffers.
  Measured tensor payload is not allocator or peak device-memory usage.
  Logit storage is an illustrative materialization cost.

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
