# Lecture 10 teaching plan

Data preparation, evaluation and experimental design · November 11, 2026.
**120 teaching minutes, then 15 minutes for Quiz 4.** Breaks are separate.
The revised material combines data quality with evaluation after the taught
InstructGPT/reward-model and REINFORCE/PPO lectures.

Central question: **Which data decision changed the behavior, under which controls?**

## Classroom route

There are 45 core slides through `methods-reading`, then 24 optional extension
slides beginning at `outline-benchmarks`. The final `references` slide serves both.
The deck is a material bank. GLUE/QNLI construction and SWAG/HellaSwag shortcut
audits are core; the open-data catalog, FineWeb case, model-interface comparison
and detailed candidate scoring are now optional. `source-card` and `data-evidence`
also remain in the appendix. Two core bridges connect the taught SFT/RM/PPO
workflow to evaluation boundaries and an incomplete reward test. Preserve exercise assumptions.
Timing is a planning estimate, not a classroom rehearsal.

| Time (Asia/Shanghai) | Route | Purpose |
| --- | --- | --- |
| 13:30–13:37 | `title`–`posttraining-data-boundaries` | Retrieve SFT/RM/PPO; declare a claim and audit evaluation exposure across stages. |
| 13:37–13:53 | `data-lineage`–`historical-shortcuts` | Schemas, extraction, distractors and shortcut checks. |
| 13:53–14:03 | `exact-duplicates`–`exercise-01` | Normalization, similarities; E01 (6 min). |
| 14:03–14:15 | `outline-curation`–`data-exercise-03` | Match families and QNLI-style article lineage; E02 (8 min). |
| 14:15–14:25 | Break | |
| 14:25–14:35 | `contamination-paths`–`fixed-budget` | Group before splitting; retain IDs; match processed tokens. |
| 14:35–14:47 | `training-mixture`–`data-exercise-04` | Sampling exposure; E03 (7 min). |
| 14:47–14:58 | `outline-evidence`–`mixtures` | Freeze protocols; distinguish loss, task metrics and aggregation. |
| 14:58–15:10 | `graders`–`evaluation-demo` | Incomplete reward-test walkthrough; audit prompts; E04 (6 min). |
| 15:10–15:25 | Break | |
| 15:25–15:39 | `uncertainty-sources`–`bootstrap-result` | Paired resampling; E05 (8 min). |
| 15:39–15:50 | `error-analysis`–`project-evaluation` | Failure slices and a feasible experiment record. |
| 15:50–15:55 | `summary`–`methods-reading` | Retrieve the evidence needed for a claim. |
| 15:55–16:10 | Quiz 4 | Previously taught material; questions remain instructor-only until the quiz. |

If discussion runs long, summarize the extraction illustration and schema table,
and leave the corpus/method catalogs in the appendix. Use notes for candidate
scoring. Do not compress the paired-comparison exercise into
a claim that overlapping uncertainty intervals prove equivalence.

## Exercises and checked outcomes

| Exercise | Expected response |
| --- | --- |
| E01, 6 min | Exact groups `train-1/train-2` and `train-4/dev-1`; NFC exposes the cross-split match. Red/blue word-bigram Jaccard is 1/3. |
| E02, 8 min | Components {A,B,C} and {D}; A/B/C must stay together. The separate QA fixture passes a question-ID audit but fails an article-ID audit; inherit article splits. One held-out group is insufficient for a broad evaluation. |
| E03, 7 min | With 9M/1M source tokens and 60M processed tokens: natural weights .9/.1 imply 6/6 passes; square-root weights .75/.25 imply 5/15 passes. |
| E04, 6 min | Mixed 120 items: A 79/120, B 75/120. Independently audited clean 100: A 60%, B 64%. Exclusion changes the evaluated population. |
| E05, 8 min | B−A is +4 percentage points; paired bootstrap interval approximately [−5,+13] pp. This does not establish a positive population difference at the stated level. |
| Optional P01 | Sum log likelihood selects the short candidate; token mean selects the long one. Neither convention is interchangeable. |
| Optional P02 | For n=10, c=2: pass@1=.2, pass@5=7/9. State the sampling assumptions. |
| Optional P03 | Jaccard 1/2, two of four signatures match, two bands of two rows give candidate probability 7/16. |

All data are authored synthetic fixtures unless a chart explicitly cites a
historical study. The notebook follows this exercise order and runs offline
with Python 3.11+ standard library on CPU; GPU memory and total GPU-hours are 0.
No model or benchmark download is required. These are ungraded practices for
all students, not new assignment requirements.

The additional notebook walkthrough compares `return 4` with `return x * x`.
Both pass the training reward case (input 2); only the latter passes the
independent case (input 3). This is an executed counterexample to an inference
from reward to general correctness, not a trained-policy result. Keep it to
two minutes; E01–E05 and P01–P03 retain their IDs and order.

Lecture 11 builds on this audit to introduce DPO and GRPO. Those algorithms
were prepared as Lecture 09 supplements but were not yet taught. See the
[remaining lecture plan](../../docs/course-revision.html#lecture-11-dpo-grpo-and-verifiable-rewards).

## Source adaptation

The [reading guide's CS336 map](optional-reading.md#cs336-data-and-evaluation-materials)
distinguishes the data/evaluation methods from the instructor's historical
benchmark material and our authored fixtures. It pins the Spring 2026 lecture
sources; full Stanford data-processing infrastructure is outside this lesson.

The earlier data-quality deck was prepared from the instructor's Spring
`lecture-07-slides-bert-benchmarks.pptx` (35 slides), with the five editable
pipeline diagrams and modern protocol readings added in October. On November 10
it moved from Lecture 08 to Lecture 10. Stable slide IDs and the analytical
fixtures were preserved. MinHash is now optional; paired uncertainty is core.
The source hash and historical chart values remain in [asset provenance](assets/README.md).
The benchmark catalog illustrates protocol design rather than a live ranking.

The instructor also supplied the **52-slide October 29, 2025**
`lecture-07-slides.pptx`. The November 10 integration uses it as follows:

| 2025 source slides | Lecture 10 destination | Preparation lesson |
| --- | --- | --- |
| 3–13, 15–23 | Core `data-schema`; notes Sections 2 and 5 | Define labels, preserve IDs and choose the scoring unit. |
| 11 | Core `task-construction`; notebook E02 | Materializing question–sentence rows must preserve article lineage. |
| 25–31 | Core `negative-construction`, `historical-shortcuts` | Construct negatives, validate ambiguity and audit context-free shortcuts. |
| 34 | Optional `lambada-construction`; notes Section 7 | Contrast full and limited context; keep source-book splits. |
| 41 | Optional `minimal-pair-construction`; notes Section 7 | Keep grammatical contrasts, pair IDs and generation templates. |
| 44–47 | Optional `benchmark-task-fields` | Make preprocessing checks match the task's annotations. |
| 48 | Optional `collection-rounds` | Record collection rounds and preserve independent evaluation. |
| 33, 35–42, 49–51 | Existing task/protocol appendix | Retain task distinctions; use versioned modern readings for protocol changes. |

Historical model rankings are not imported as current results. The source's
STS-B lower bound, CoLA label/metric distinction and LAMBADA training-set
description are corrected against primary sources. The original file remains
unchanged; credits and exact corrections are in the asset provenance file.

## Verification

```sh
uv run python -m pytest -q tests/test_lecture_10.py tests/test_lecture_10_data.py
npm --prefix slides run check -- lecture-10
npm --prefix slides run pdf -- lecture-10
uv run python scripts/slides.py serve --port 8010
node slides/lecture-10/check-demo.mjs http://127.0.0.1:8010
npm --prefix slides run check:notebook -- http://127.0.0.1:8010 lecture-10
npm --prefix slides run build:lecture-notes
npm --prefix slides run check:notes
```

The demo switches between mixed and clean evaluation and resets to the mixed
state. PDF output uses that initial state; E04 prints both results. Review
artifacts belong in ignored `slides/.checks/lecture-10/`. See the notes for a
full experimental-record template and the reading guide for source versions.

Local verification on November 10, 2026 (Asia/Shanghai), against base
`d930c2ca2b973973816770892ac3308c7156301a` plus the lecture/plan edits:
10 Python checks passed, including execution in a fresh notebook working copy;
all 69 slides passed three viewport checks and were visually inspected;
the PDF has 69 pages. The mixed/clean demo and notebook launcher passed on
the existing local server at port 8005. Reading-page checks passed for notes,
equations, mobile/print layouts and all six future lecture links in English
and Chinese. No GPU run or live-model comparison was performed. Timing and
future live-demo budgets remain planning estimates.
