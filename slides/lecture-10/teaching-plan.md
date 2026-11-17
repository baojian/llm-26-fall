# Lecture 10 teaching plan

Data preparation, evaluation and experimental design · November 11, 2026.
**120 teaching minutes, then 15 minutes for Quiz 4.** Breaks are separate.
The revised material combines data quality with evaluation after the taught
InstructGPT/reward-model and REINFORCE/PPO lectures.

Central question: **What changed in the data, model or evaluated system, and what evidence supports the improvement?**

## Classroom route

The revised material bank has **95 slides: 57 core slides through
`methods-reading`, then 38 optional slides starting at `outline-benchmarks`**.
The five exercises retain their IDs, order and checked assumptions. The new
history and HLE/SWE-bench/Terminal-Bench/ALE cases are in the core. Broader 2026
benchmark comparisons and detailed likelihood/pass@k/MinHash methods remain
optional. This timing is a planning estimate, not a rehearsal measurement.

| Time (Asia/Shanghai) | Route | Minutes and purpose |
| --- | --- | --- |
| 13:30–13:42 | `title`–`current-model-boundary` | 12: evaluation history, four paper excerpts, inference-budget and source boundaries. |
| 13:42–14:02 | `outline-data`–`exercise-01` | 20: provenance, extraction evidence, shortcuts, exact/near duplicates; E01 (6). |
| 14:02–14:15 | `outline-curation`–`data-exercise-03` | 13: connected families/article lineage; E02 (8, including debrief). |
| 14:15–14:25 | Break | 10, outside teaching budget. |
| 14:25–14:38 | `fixed-budget`–`mixture-exposure` | 13: actual FineWeb ablation, matched tokens, sampling exposure; E03 (7). |
| 14:38–15:00 | `outline-evidence`–`agent-comparison-check` | 22: four benchmark cases and complete agent protocol; 90-second oral check. |
| 15:00–15:10 | `graders`–`prompt-sensitivity` | 10: order-swap probe, independent verifier and prompt selection. |
| 15:10–15:25 | Break | 15, outside teaching budget. |
| 15:25–15:35 | `exercise-04`–`evaluation-demo` | 10: contamination reversal; E04 (6) plus demo/debrief. |
| 15:35–15:49 | `uncertainty-sources`–`bootstrap-result` | 14: paired uncertainty; E05 (8) and interpretation. |
| 15:49–15:55 | `error-analysis`–`methods-reading` | 6: failure slices, feasible experiment and retrieval summary. |
| 15:55–16:10 | Quiz 4 | 15, previously taught material; questions stay private. |

Teaching minutes total **120**. The route is deliberately concise; rehearse
before delivery. If time runs short, skip `gpt1-task-adaptation` (summarized by
the timeline), `swebench-family`, `terminal-versioning`, and
`ale-harness-evidence`; retain the four task objects and all five exercises.
Do not make up time by dropping paired uncertainty or implying that an interval
crossing zero proves equivalence. Use the full atlas for independent reading.

Each source figure has a question in its notes: identify the metric, changed
factor, controls and claim boundary. Historical scores are never plotted on a
single cross-generation axis. The requested GPT-6.2 endpoint is explicitly
unverified; official sources establish the separately named GPT-6/6.1 examples.
All embedded figure assets load locally. No live benchmark/API run is required.
Use the native takeaway text to teach the ALE and GPT-4 plots; their fine
internal labels are source detail for close reading, not text students must
read from the back of the classroom.

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

E03 now places its solved exposure diagram after prediction, both in slides and
the notebook Check cell. The calculations and seven-minute exercise are unchanged.

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

The November 11 instructor update confirms that DPO and GRPO were covered
across Lectures 8–10. Lecture 11 now builds on this evaluation audit through
efficient adaptation and distillation. See the
[remaining lecture plan](../../docs/course-revision.html#lecture-11-efficient-adaptation-and-distillation).

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
| 3–13, 15–23 | Optional `data-schema`; notes Sections 2 and 5 | Define labels, preserve IDs and choose the scoring unit. |
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

## November 11 revision

The instructor requested three iterations, each informed by three independent
reviewers. The [revision record](revision-review.md) records decisions and
validation. The expanded history, 14 locally bundled source figures/illustrations,
four benchmark families, 2026 atlas and judge/agent probes build on the existing
data-quality and uncertainty exercises. Original exercise values remain unchanged.

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

Prior baseline verification on November 10, 2026 (Asia/Shanghai), against base
`d930c2ca2b973973816770892ac3308c7156301a` plus the lecture/plan edits:
10 Python checks passed, including execution in a fresh notebook working copy;
all 69 slides passed three viewport checks and were visually inspected;
the PDF has 69 pages. The mixed/clean demo and notebook launcher passed on
the existing local server at port 8005. Reading-page checks passed for notes,
equations, mobile/print layouts and all six future lecture links in English
and Chinese. No GPU run or live-model comparison was performed. Timing and
future live-demo budgets remain planning estimates.

November 11 final revision verification: all three iterations and nine
independent reviews are complete. Ten Python checks pass, and all 95 slides
pass three viewport checks. The exported PDF has 95 pages; every page was
visually inspected, with the corrected currency answer reinspected after the
final export. All 14 source-asset hashes match their provenance manifest.
Demo, notebook-launcher and notes checks pass. See the
[revision record](revision-review.md#release-validation) for commands, version
and working-tree state, artifacts and remaining limits. No GPU or live-model
run was performed; the 120-minute classroom route still needs rehearsal.
