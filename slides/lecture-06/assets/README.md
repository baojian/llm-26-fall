# Lecture 06 asset provenance

## Sources and authorship

The lesson adapts the instructor-supplied `lecture-06-slides-gpts.pptx`,
130 slides, dated April 16, 2026. Source SHA-256:
`0a252f6b509ef10b910c58d447c4ffeda48b2acc4854271e5ea90f8e5445c6c5`.
The [public Spring copy](https://github.com/baojian/llm-26/blob/main/slides/lecture-06-slides/lecture-06-slides-gpts.pptx)
has different bytes. Page numbers in the [teaching plan](../teaching-plan.md)
refer to the instructor-supplied snapshot. The source acknowledges Jacob Devlin,
Anoop Sarkar, Stanford CS224n, and Jurafsky & Martin. No embedded Spring images
are redistributed in this lecture: the diagrams are newly drawn and plots are
calculated or measured as identified below. The original PowerPoint is unchanged.

`toy_lm.py` adapts the model definitions in Lecture 05's implementation notebook,
including their initialization order. Training and checkpoint utilities are new
supplied teaching examples. They use their own toy interfaces and data, separate
from the graded A2 starter/reference functions.

## Diagram and chart inventory

| Files | Origin and interpretation |
| --- | --- |
| `noa-model.svg` / `.excalidraw` | Lecture 05 architecture; 4,608 unique parameters |
| `shifted-targets.svg` / `.excalidraw` | New Noa drawing of the Spring slide 12 next-token mechanism |
| `training-update.svg` / `.excalidraw` | New eight-stage CPU training schematic |
| `checkpoint-state.svg` / `.excalidraw` | New state diagram for the supplied continuation experiment |
| `resume-timeline.svg` / `.excalidraw` | New 40 versus 20+20 experiment timeline |
| `generation-loop.svg` / `.excalidraw` | New drawing of the Spring slides 20–24 autoregressive mechanism |
| `greedy-tree.svg` / `.excalidraw` | Invented complete two-step tree; joint probabilities .30,.30,.36,.04 |
| `learning-rate.json` | Calculated 40-update schedule, warmup 4, max .02, min .002 |
| `tiny-fit.json` | All 201 recorded fitting losses, including initialization; analytic floor shown separately |
| `development-loss.json` | All 21 full-split train/dev evaluations, every ten completed updates |
| `gradient-norm.json` | All 200 measured pre-clipping norms from the same development run |
| `temperature.json`, `sampling.json` | Calculated four-token distributions, base probabilities (.4,.3,.2,.1) |
| `evidence-summary.json` | Curated provenance, settings, comparison results, and prepared generations |
| `corpus.json` | 24 course-authored toy documents, fixed train/dev split |

`build_assets.py` writes each SVG and editable Excalidraw scene from the same
coordinates and text. These conceptual diagrams use the course's existing
Arial font and colors. Plotly inputs are local SVG-based traces and render in
browser PDF export. E05's diagnostic curves are explicitly schematic in the
notebook and are not substituted into any measured figure.

## Recorded environment and measurement boundaries

Recorded UTC: **2026-10-12T04:44:43.604850+00:00**.

- Hardware: **Apple M1, 16 GiB unified memory**; CPU only, no GPU hours.
- OS: `macOS-26.3-arm64-arm-64bit`.
- Python `3.11.12`, PyTorch `2.14.0`, CPU float32, two threads.
- Base revision: `0e6990280eb122cbf2efee1a95e56f04e389bcc5`. The new source files were uncommitted at
  measurement time; exact content hashes are recorded in `evidence-summary.json`.
- Full-batch fitting loop: **0.280 seconds**, excluding imports and
  model/optimizer construction; 200 updates, 4,000 targets.
- Development loop, including periodic full-split evaluation:
  **0.389 seconds**, excluding imports and construction; 200
  updates, **1021** sampled targets.
- Restart comparison: **6.194 seconds**, including all four
  worker process startups, their imports, checkpoint I/O, and comparison.
  Parent interpreter startup/imports are excluded. Dependencies and files were
  already available locally; this is not a first-install timing.
- Largest worker peak process RSS: **286.9 MiB**,
  measured with the OS process high-water mark, including imports and training.
  Workers run sequentially; this number excludes the parent process and is not
  a measurement of total system RAM usage. Individual worker records are in
  the summary. Other machines may differ.

The earlier standalone pilot took 10.72 seconds for its restart comparison;
the recorded final run above is a separate measurement. These are observed
preparation runs, not classroom runtime guarantees. Prepared results are
available if installation or subprocess startup would interrupt the exercise.

## Exact protocols and results

All models have vocabulary 12, context 10, width 16, four heads, two pre-LN
blocks, ReLU FFN width 32, final affine LayerNorm (epsilon 1e-5), learned
positions, no Linear biases/dropout, and tied token/readout weights. Matrix
initialization is normal with standard deviation .02; norm scales are one and
biases zero. The unique parameter count is **4,608**.

### Fitting fixture

Two documents: `[0,1,2,3,4,5,6,7,8,9,11]` and
`[0,1,2,3,4,5,6,7,8,10,11]`. Inputs/targets are the first/last ten tokens.
Seed 7, B=2,T=10, 200 full-batch updates, constant LR .02, AdamW betas
(.9,.999), epsilon 1e-8, no decay/clipping/schedule. The final recorded mean
NLL is **0.069821998**, compared with infimum ln(2)/10. There is no
held-out set. This reproduces the Lecture 05 pre-LN result under the stated setup.

### Restart

Randomly sample one of those two whole documents per update using a dedicated
CPU Generator with seed 17. Model seed 7, 40 updates, warmup 4, max LR .02, minimum
ratio .1, global clip 1, AdamW betas (.9,.999), epsilon 1e-8, decay 0. Schedule and
batch sequence are fixed across comparisons. Save only after complete updates.

Full restoration has maximum final parameter difference **0** from continuous
training; declared tolerance is absolute 1e-7, relative 0. The control restores
weights, schedule, and future batches while resetting optimizer history. Its
first forward-loss difference is 0, second forward-loss difference
**0.0284600**, and final maximum parameter difference **0.1278004**. The next
forward pass precedes the changed optimizer update, which explains the initial
agreement. This is a same-environment result.

### Separate development corpus

Sixteen training and eight development document strings are disjoint. The fixed
12-token vocabulary covers both; prefixes and words can overlap. Split at the
document level before forming whole-document windows. EOS is appended; no BOS,
padding, or cross-document windows. Same scheduled optimizer settings, with 200
updates and warmup 20. One uniform random document is sampled per update.

Full-split evaluation is token-weighted over 82 train and 48 dev targets. Final
NLL is **0.515295583 / 1.402659148**.
The stochastic per-document objective and full-split token-weighted evaluation
use different weights. Preserve that distinction. This template corpus supports
a classroom demonstration, not a broad natural-language evaluation claim.

### Prepared samples and browser interaction

The prefix ends at great: IDs `[0,1,2,3,4,5,6,7,8]`. All three generations use the
200-update **fitting** checkpoint, seed 29 and at most 12 new tokens. Greedy yields
`friend EOS`; tau=.7,p=.9 and tau=1,p=1 both yield `cat EOS`. The checkpoint file
hash and full next-logit vector are recorded; weights remain in ignored output.

The browser interaction uses four invented logits `log(.4,.3,.2,.1)`, with
positive temperature and inclusive top-p. It uses Mulberry32 for deterministic
classroom draws, distinct from PyTorch's CPU RNG. Reset restores tau=1, p=0.6, seed 29.
This browser example demonstrates the probability operation; it does not execute
the trained model. The fresh-page PDF contains the initial bars and static result.

## Reproduce

From the repository root, choose a new ignored output directory for each run:

```sh
uv sync --extra tokenization
uv run --extra tokenization python slides/lecture-06/assets/collect_evidence.py --out workspace/lecture-06/my-run
uv run --extra tokenization python slides/lecture-06/assets/build_assets.py --evidence workspace/lecture-06/my-run/evidence.json
```

The first command for collecting evidence refuses an existing output directory.
Raw traces and checkpoint files stay under `workspace/`. The asset builder
extracts only the curated teaching figure inputs and provenance into this folder.
To rerun only the continuation check without retaining checkpoints:

```sh
uv run --extra tokenization python slides/lecture-06/assets/resume_demo.py
```

The notebook can be rerun offline after dependencies are installed. Source hashes
in the summary cover `toy_lm.py`, `training.py`, `resume_demo.py`,
`collect_evidence.py`, and `corpus.json`. No model weights, student work, grades,
raw run logs, or private assessment solutions are part of the lecture release.
