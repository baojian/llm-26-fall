# Lecture 10 teaching plan

Data preparation, evaluation and experimental design · November 11, 2026.
**120 teaching minutes, then 15 minutes for Quiz 4.** Breaks are separate.
The revised material combines data quality with evaluation after the taught
InstructGPT/reward-model and REINFORCE/PPO lectures.

Central question: **Which data decision changed the behavior, under which controls?**

## Classroom route

There are 45 core slides through `methods-reading`, then 15 optional extension
slides beginning at `outline-benchmarks`. The final `references` slide serves both.
The deck is a material bank: summarize the open-data examples and FineWeb case
briefly rather than reading every row. Preserve the exercises and assumptions.
Timing is a planning estimate, not a classroom rehearsal.

| Time (Asia/Shanghai) | Route | Purpose |
| --- | --- | --- |
| 13:30–13:37 | `title`–`running-comparison` | Retrieve SFT/RM/PPO; declare a claim and controlled comparison. |
| 13:37–13:50 | `data-lineage`–`filter-audit` | Trace provenance, extraction and selection errors. |
| 13:50–14:03 | `exact-duplicates`–`exercise-01` | Normalization, similarities; E01 (6 min). |
| 14:03–14:15 | `outline-curation`–`data-exercise-03` | Family grouping and leakage; E02 (6 min). |
| 14:15–14:25 | Break | |
| 14:25–14:35 | `contamination-paths`–`fixed-budget` | Group before splitting; retain IDs; match processed tokens. |
| 14:35–14:48 | `training-mixture`–`data-exercise-04` | Sampling exposure; E03 (7 min). |
| 14:48–15:00 | `outline-evidence`–`mixtures` | Freeze protocols; distinguish likelihood, task metrics and aggregation. |
| 15:00–15:10 | `graders`–`evaluation-demo` | Audit grader/prompt choices; E04 (6 min). |
| 15:10–15:25 | Break | |
| 15:25–15:39 | `uncertainty-sources`–`bootstrap-result` | Paired resampling; E05 (8 min). |
| 15:39–15:50 | `error-analysis`–`project-evaluation` | Failure slices and a feasible experiment record. |
| 15:50–15:55 | `summary`–`methods-reading` | Retrieve the evidence needed for a claim. |
| 15:55–16:10 | Quiz 4 | Previously taught material; questions remain instructor-only until the quiz. |

If discussion runs long, skip the FineWeb case and open-data catalog, and use the
notes for candidate scoring. Do not compress the paired-comparison exercise into
a claim that overlapping uncertainty intervals prove equivalence.

## Exercises and checked outcomes

| Exercise | Expected response |
| --- | --- |
| E01, 6 min | Exact groups `train-1/train-2` and `train-4/dev-1`; NFC exposes the cross-split match. Red/blue word-bigram Jaccard is 1/3. |
| E02, 6 min | Components {A,B,C} and {D}; A/B/C must stay together under the declared grouping rule. One held-out group is insufficient for a broad real evaluation. |
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

## Source adaptation

The earlier data-quality deck was prepared from the instructor's Spring
`lecture-07-slides-bert-benchmarks.pptx` (35 slides), with the five editable
pipeline diagrams and modern protocol readings added in October. On November 10
it moved from Lecture 08 to Lecture 10. Stable slide IDs and the analytical
fixtures were preserved. MinHash is now optional; paired uncertainty is core.
The source hash and historical chart values remain in [asset provenance](assets/README.md).
The benchmark catalog illustrates protocol design rather than a live ranking.

## Verification

```sh
uv run python -m pytest -q tests/test_lecture_10.py tests/test_lecture_10_data.py
npm --prefix slides run check -- lecture-10
npm --prefix slides run pdf -- lecture-10
uv run python scripts/slides.py serve --port 8010
node slides/lecture-10/check-demo.mjs http://127.0.0.1:8010
```

The demo switches between mixed and clean evaluation and resets to the mixed
state. PDF output uses that initial state; E04 prints both results. Review
artifacts belong in ignored `slides/.checks/lecture-10/`. See the notes for a
full experimental-record template and the reading guide for source versions.
