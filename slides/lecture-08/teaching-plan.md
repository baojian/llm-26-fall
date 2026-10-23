# Lecture 08 teaching plan

**Data quality and model evaluation** · October 28, 2026 · 135 teaching minutes.

Central question: **Did better training data actually improve the model?**

This preparation combines data preparation with evaluation in one lecture.
It draws on the instructor's Spring benchmark material and updates the examples
against primary sources checked on October 23, 2026. The website's Lecture 09
schedule has not been reassigned by this draft.

## Outcomes and preparation

Students should be able to audit duplicate families and split overlap, compare
data policies under a fixed processed-token budget, specify a benchmark protocol,
and qualify an apparent accuracy improvement using paired predictions.

Prerequisites: Lecture 02's likelihood and metric units; Lecture 06's causal
language-model training; Lecture 07's compute accounting and independent checks.
The model-interface slide briefly introduces masked-token scoring for BERT.
The examples reinforce A2's data-policy comparison without changing its published
requirements. All five classroom exercises are ungraded and apply equally to
all students.

The core requires Python 3.11+ and a CPU. The notebook uses the standard library
and supplied fixtures. GPU memory and total GPU-hours: **0**. No model downloads,
external API calls, or full benchmark runs are required. The published studies'
compute scales are background, not course allocations.

## Teaching sequence

The deck has 45 slides, with stable IDs. Each block includes discussion and the
exercise's answer review. Times are planning estimates, not a live rehearsal.

| Time (Asia/Shanghai) | Slides | Teaching purpose |
| --- | --- | --- |
| 13:30–13:36 | 1–4, `title`–`running-comparison` | Establish the question and the fixed comparison. |
| 13:36–13:44 | 5–7, `data-lineage`–`quality-filters` | Trace records through extraction and selection. |
| 13:44–13:52 | 8–9, `exact-duplicates`–`near-duplicates` | Calculate exact keys and word-shingle similarity. |
| 13:52–13:58 | 10, `exercise-01` | E01: audit duplicate families and split overlap. |
| 13:58–14:05 | 11–12, `split-construction`–`contamination-paths` | Connect grouping to the generalization question. |
| 14:05–14:15 | 13–15, `fixed-budget`–`data-evidence` | Match budgets, inspect mixtures, connect to papers. |
| 14:25–14:30 | 16–18, `outline-benchmarks`–`historical-tasks` | Define a benchmark as data plus protocol. |
| 14:30–14:39 | 19–21, `historical-shortcuts`–`completion-scoring` | Interpret a historical ablation and scoring interfaces. |
| 14:39–14:45 | 22, `exercise-02` | E02: compare summed and per-token scores. |
| 14:45–14:51 | 23–24, `knowledge-benchmarks`–`hle-diamond` | Explain question coverage and version identity. |
| 14:51–14:59 | 25–28, `code-and-interaction`–`pass-at-k` | Inspect current protocols and derive the attempt metric. |
| 14:59–15:04 | 29, `exercise-03` | E03: calculate pass@1 and pass@5. |
| 15:04–15:10 | 30, `graders` | Discuss grader validation and student questions. |
| 15:25–15:28 | 31–32, `outline-evidence`–`comparison-record` | Return to the frozen comparison. |
| 15:28–15:34 | 33, `exercise-04` | E04: compute the ranking before and after an audit. |
| 15:34–15:38 | 34–35, `evaluation-demo`–`loss-and-task` | Toggle the chart; distinguish task accuracy from loss. |
| 15:38–15:46 | 36–38, `prompt-sensitivity`–`paired-items` | Identify uncertainty sources and the paired unit. |
| 15:46–15:54 | 39, `exercise-05` | E05: run the paired bootstrap and interpret its scope. |
| 15:54–16:01 | 40–41, `bootstrap-result`–`error-analysis` | Inspect the interval and choose a useful next check. |
| 16:01–16:10 | 42–45, `project-evaluation`–`references` | Draft an evaluation record; summarize and discuss readings. |

If discussion runs long, shorten the benchmark catalog and reading walkthrough.
Preserve the worked calculations and interpretation of E04/E05. Students can
read detailed benchmark descriptions afterward; memorizing dataset names is not
a learning objective.

## Checked examples and limitations

| Exercise | Expected response |
| --- | --- |
| E01, 6 min | Exact groups: `train-1/train-2` and `train-4/dev-1`. The second crosses splits after NFC normalization. Red/blue word-bigram Jaccard is `2/6 = 1/3`; the changed color can be semantically important. |
| E02, 6 min | Sums: A `−2.4`, B `−3.2`, so A wins. Per-token means: A `−1.2`, B `−0.8`, so B wins. No correct task label is supplied. |
| E03, 5 min | With `n=10`, `c=2`, pass@1 is `0.2`; pass@5 is `1−56/252 = 7/9`. The attempt budget changes; the weights do not. |
| E04, 6 min | All 120: A `79/120 = 65.83%`, B `75/120 = 62.5%`. Clean 100: A `60%`, B `64%`. Exposed items were marked by an independent audit. |
| E05, 8 min | B−A is `+4` percentage points. The 5,000-draw paired percentile interval with seed `20261028` is `[−5, +13]` points. It crosses zero and excludes uncertainty from training seeds, prompt selection, and grader errors. |

All classroom outcomes are synthetic. They do not measure a trained model or
estimate the causal effect of real-world contamination. The only plotted paper
measurements are the explicitly dated BERT-Large shortcut ablations.

For a newly constructed dataset, group related records before assigning splits.
For a fixed external benchmark, preserve its split and audit training overlap
under a declared policy. Do not tune filters on test answers or discard items
because the preferred model fails them. The helper code groups records in memory;
it does not remove source files.

The bootstrap conditions on two fixed systems and independent evaluation items
from the relevant population. Related rows may need cluster resampling. Its
percentile interval is an instructional approximation, not a universal test.
An interval crossing zero does not prove equivalence. The fixed synthetic counts
cannot reveal variability across training seeds.

## Adaptation from the Spring source

Source: `lecture-07-slides-bert-benchmarks.pptx`, 35 slides, cover dated
April 23, 2026; instructor Baojian Zhou, with the original credit to Anoop Sarkar.
The accompanying `lecture-07-bert-benchmarks.ipynb` supplies the model-interface
comparison. The originals remain unchanged and are not copied into this package.

| Spring slides | Treatment in this lecture |
| --- | --- |
| 1–2, 12, 19, 30, 35: cover, outlines, references | Replace with the Fall course format and current sources. |
| 3–11: SuperGLUE task catalog | Compress into `historical-tasks`; retain label/reading-comprehension distinctions. |
| 13–17: SWAG and HellaSwag | Retain the shortcut question; redraw Figure 4 as an editable chart in `historical-shortcuts`. |
| 18: historical model scores | Omit the leaderboard comparison; use dated protocols rather than an old rank ordering. |
| 20–29: other benchmarks | Retain representative task interfaces and HumanEval's attempt metric; update the knowledge and code examples. |
| 31: structured prediction catalog | Leave as background; the classroom focus is evaluation design. |
| 32–34: saturation, BBH, ARC Prize | Develop into versioning, grading repair, and interactive evaluation examples. |
| Companion notebook: BERT, GPT-2, Qwen zero-shot evaluation | Preserve the interface contrast; replace automatic downloads with transparent offline calculations. |

The data-policy and uncertainty sections are new course material. Current
benchmark sources and dates are listed in [the reading guide](optional-reading.md).
The source hash, historical values, and synthetic construction are documented in
[asset provenance](assets/README.md).

## Delivery and verification

```sh
uv sync
uv run python slides/lecture-08/assets/build_assets.py
uv run python -m pytest -q tests/test_lecture_08.py
uv run python scripts/slides.py serve
node slides/lecture-08/check-demo.mjs http://127.0.0.1:8000
npm --prefix slides run check -- lecture-08
npm --prefix slides run pdf -- lecture-08
```

Open `/slides/lecture-08/`. The shared **Notebook** launcher creates a personal
copy with the supplied assets. The evaluation button switches between the mixed
120-item and clean 100-item fixtures; **Reset** restores the mixed set. PDF output
uses that initial state; E04 prints both results. Browser dependencies and fonts
are local assets. Review images and PDF output live in ignored
`slides/.checks/lecture-08/`.

Numerical checks cover Unicode overlap, length-sensitive decisions, pass@k
against exhaustive subsets, paired resampling, and execution from a fresh
notebook directory.

Preparation check, October 23, 2026 (Asia/Shanghai): all six numerical tests
passed, and all eight notebook code cells ran in a fresh kernel. The 45 slides
passed three viewport checks; all slide images and all 45 PDF pages were
visually reviewed. Chart units, evaluation switching/reset, offline browser
loading, and the notebook launcher passed their checks. Classroom timings
remain estimates, and the draft's combined scope awaits curriculum review.
