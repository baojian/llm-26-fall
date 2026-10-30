# Lecture 08 teaching plan

**Data preparation and quality** · October 28, 2026 · 135 teaching minutes.

Central question: **Can better data improve a fixed model under a fixed training budget?**

## Assessment and scope

The initial draft had sound offline calculations and useful benchmark examples,
but devoted two of three periods to evaluation. That overlapped with the published
Lecture 09 topic and left little time to explain data preparation itself.

The revised core follows the published schedule: extraction and filtering;
near-duplicate matching and split construction; sampling and a controlled
comparison. Five editable diagrams and three additional data exercises make the
pipeline inspectable. The benchmark material remains in a clearly marked optional
extension. Its existing slide IDs remain valid. Evaluation and experimental design
remain the Lecture 09 topic; assignment dates and requirements are unchanged.

This is a preparation draft for instructor review. The timings below are planning
estimates, not a live classroom rehearsal.

## Outcomes and preparation

Students should be able to audit extraction and selection decisions, distinguish
exact keys from approximate matching, group related records before splitting,
calculate source exposure, and design a data-policy comparison at a fixed budget.
The final evaluation audit provides a bridge to Lecture 09.

Prerequisites: Lecture 02 likelihood and metric units; Lecture 06 causal-LM
training; Lecture 07 compute accounting. Retrieve the difference between available
corpus tokens and processed training tokens. The examples reinforce A2's data-policy
comparison without changing its published requirements. All practice is ungraded
and applies equally to all students.

The notebook requires Python 3.11+ and a CPU. It uses the standard library and
supplied fixtures. GPU memory and total GPU-hours: **0**. No model downloads,
external API calls, or full benchmark runs are required. Published studies'
compute scales are background, not course allocations.

## Teaching sequence

There are **40 core slides and 20 optional extension slides**. The main route ends
at `methods-reading`; the extension begins at `outline-benchmarks`. The final
`references` slide serves both routes. Each period includes student work and
answer discussion; do not add the optional exercises to the 135-minute schedule.

| Time (Asia/Shanghai) | Slides | Teaching purpose |
| --- | --- | --- |
| 13:30–13:36 | 1–4 | Retrieve the scaling-law context; establish the fixed comparison. |
| 13:36–13:43 | 5–7 | Trace source records; inspect the evidence available from open data projects. |
| 13:43–13:50 | 8–9 | Identify extraction errors in an authored web page. |
| 13:50–13:56 | 10–11 | Audit useful rejections and possible false rejections. |
| 13:56–14:03 | 12–13 | Calculate exact keys and word-shingle similarity. |
| 14:03–14:09 | 14, `exercise-01` | E01: find duplicates and cross-split overlap. |
| 14:09–14:15 | 14, discussion | Review NFC and semantic differences; collect questions. |
| 14:15–14:25 | Break | |
| 14:25–14:32 | 15–16 | Explain MinHash through shared orderings and the union/intersection proof. |
| 14:32–14:37 | 17 | Derive the ideal LSH banding probability. |
| 14:37–14:42 | 18, `data-exercise-02` | E02: distinguish overlap, an estimate, and candidate probability. |
| 14:42–14:48 | 19–20 | Build connected families; choose a grouping boundary. |
| 14:48–14:54 | 21, `data-exercise-03` | E03: repair a split without deleting records. |
| 14:54–15:01 | 22–23 | Trace contamination paths and preserve IDs through chunking. |
| 15:01–15:06 | 24–25 | Read the FineWeb deduplication comparison with its limits. |
| 15:06–15:10 | 26 | Distinguish retained tokens, processed tokens, and passes. |
| 15:10–15:25 | Break | |
| 15:25–15:32 | 27–29 | Derive token sampling weights and expected source exposure. |
| 15:32–15:39 | 30, `data-exercise-04` | E04: compare natural and square-root weights. |
| 15:39–15:42 | 31 | Separate training weights from evaluation aggregation. |
| 15:42–15:46 | 32 | Declare the controlled comparison before looking at results. |
| 15:46–15:52 | 33, `exercise-04` | E05: calculate the ranking before and after an independent audit. |
| 15:52–15:55 | 34 | Toggle the interactive audit; explain why the conclusion changes. |
| 15:55–16:00 | 35–36 | Distinguish loss from task metrics and name uncertainty sources. |
| 16:00–16:06 | 37–38 | Inspect failures and draft a feasible project comparison. |
| 16:06–16:10 | 39–40 | Retrieve the five decisions and select a reading question. |

If discussion runs long, shorten the open-data catalog and reading walkthrough.
Keep the worked examples and their assumptions. The exhaustive 24-permutation
check is an optional notebook follow-up; do the four-order calculation in class.

## Board work and expected responses

For MinHash, write the union and intersection first. In a uniformly random order,
the earliest union element is equally likely to be any union member. The two
minima agree exactly when that element belongs to the intersection. For banding,
write one-band match probability, then the probability that every band misses.

For mixtures, write available tokens `n_i`, their fractions `q_i`, sampling
probabilities `p_i`, the total processed budget `D`, and finally `D p_i / n_i`.
Keep the units visible and ask students to check that allocations sum to `D`.

| Exercise | Expected response |
| --- | --- |
| E01, 6 min | Exact groups: `train-1/train-2` and `train-4/dev-1`. NFC makes the second cross splits. Red/blue word-bigram Jaccard is `2/6 = 1/3`; the changed color may matter. |
| E02, 5 min | Exact Jaccard `1/2`; two of four displayed signature positions match. Two independent bands of two rows give candidate probability `7/16`. Across all 24 uniform orders, 12 minima match. |
| E03, 6 min | Components `{A,B,C}` and `{D}`. Train A/B/C and development D avoids the declared crossing. It does not require deleting A/B/C or provide a useful real evaluation size. |
| E04, 7 min | Natural weights `0.9/0.1`: `54M/6M` processed tokens, `6/6` expected passes. Square-root weights `0.75/0.25`: `45M/15M`, `5/15` passes. Both process `60M` tokens. |
| E05, 6 min | All 120 items: A `79/120 = 65.83%`, B `75/120 = 62.5%`. Clean 100: A `60%`, B `64%`. An independent exposure audit changes the evaluated population, not those clean predictions. |

Optional practice, outside the core time budget:

| Exercise | Expected response |
| --- | --- |
| P01, 6 min | Sums: A `−2.4`, B `−3.2`, so A wins. Token means: A `−1.2`, B `−0.8`, so B wins. No correct task label is supplied. |
| P02, 5 min | With `n=10`, `c=2`, pass@1 is `0.2`; pass@5 is `7/9`. The attempt budget changes; the weights do not. |
| P03, 8 min | B−A is `+4` percentage points. The 5,000-draw paired percentile interval with seed `20261028` is `[−5, +13]` points. It omits training-seed, prompt-selection, and grader uncertainty. |

## Assumptions worth making explicit

- All classroom texts, budgets, and outcomes are authored fixtures. They do not
  measure a trained model or the causal effect of real-world contamination. The
  only plotted paper measurements are the dated BERT-Large shortcut ablations.
- MinHash's ideal probability uses uniform random orders. LSH banding additionally
  assumes independent rows. Matching is not proof of semantic equivalence.
- Connected components keep declared match edges inside a split. They do not
  establish that every pair is redundant or detect every paraphrase. Retention
  and split grouping are separate decisions.
- For a newly constructed dataset, group related records before splitting. For
  an existing benchmark, keep its published split and audit training overlap.
  Preserve parent/family IDs through chunking and pack within each split.
- Mixture weights are token probabilities. Expected passes assume uniform
  within-source sampling with replacement, ignoring packing overhead. Changing
  exposure does not establish better quality; test it at a comparable budget.
- Do not tune filters on final test answers or exclude items because a preferred
  model fails them. Preserve negative results and report retained counts.
- P03 conditions on two fixed systems and independent items from the relevant
  population. Related rows may need cluster resampling. The percentile interval
  is an instructional approximation, not a posterior or proof of equivalence.

## Adaptation from the Spring source

Source: `lecture-07-slides-bert-benchmarks.pptx`, 35 slides, cover dated
April 23, 2026; instructor Baojian Zhou, with the original credit to Anoop Sarkar.
The accompanying `lecture-07-bert-benchmarks.ipynb` supplies the model-interface
comparison. The originals remain unchanged and are not copied into this package.

| Spring slides | Treatment in the optional extension |
| --- | --- |
| 1–2, 12, 19, 30, 35: cover, outlines, references | Replace with the Fall course format and current sources. |
| 3–11: SuperGLUE task catalog | Compress into `historical-tasks`; retain label/reading-comprehension distinctions. |
| 13–17: SWAG and HellaSwag | Retain the shortcut question; redraw Figure 4 as an editable chart in `historical-shortcuts`. |
| 18: historical model scores | Omit the leaderboard comparison; use dated protocols rather than an old rank ordering. |
| 20–29: other benchmarks | Retain representative task interfaces and HumanEval's attempt metric; update the knowledge and code examples. |
| 31: structured prediction catalog | Leave as background; the classroom focus is evaluation design. |
| 32–34: saturation, BBH, ARC Prize | Develop into versioning, grading repair, and interactive evaluation examples. |
| Companion notebook: BERT, GPT-2, Qwen zero-shot evaluation | Preserve the interface contrast; replace automatic downloads with transparent offline calculations. |

The data-policy sections and uncertainty example are new course material. Current
benchmark sources and dates are listed in [the reading guide](optional-reading.md).
The source hash, historical values, and synthetic construction are documented in
[asset provenance](assets/README.md).

## Delivery and verification

```sh
uv sync --locked
uv run python slides/lecture-08/assets/build_assets.py
uv run python slides/lecture-08/assets/build_diagrams.py
uv run python -m pytest -q tests/test_lecture_08.py tests/test_lecture_08_data.py
uv run python scripts/slides.py serve --port 8008
node slides/lecture-08/check-demo.mjs http://127.0.0.1:8008
npm --prefix slides run check:notebook -- http://127.0.0.1:8008 lecture-08
npm --prefix slides run pdf -- lecture-08
```

Open `/slides/lecture-08/`. The shared **Notebook** launcher creates a personal
copy with supplied assets. The evaluation button switches between the mixed
120-item and clean 100-item fixtures; **Reset** restores the mixed set. PDF output
uses that initial state; E05 prints both results. Dependencies and fonts are local.
Review images and PDF output live in ignored `slides/.checks/lecture-08/`.

Numerical checks include exhaustive MinHash permutations and LSH binary outcomes,
transitive split groups, conserved sampling budgets, Unicode overlap, pass@k
against exhaustive subsets, paired resampling, and a fresh notebook working copy.
All 11 code cells are intended to run in order without network access.

Preparation check, October 26, 2026 (Asia/Shanghai): ten numerical tests passed;
all 11 notebook code cells executed in a fresh kernel. All 60 slides passed three
viewport checks, and all slide images and PDF pages were visually reviewed.
Chart switching/reset, offline browser loading, and notebook launch/reuse passed.
All 45 original slide IDs and relative documentation links were checked.

The ignored `slides/.checks/lecture-08/validation-record.json` records the local
revision, dirty state, file hashes, commands, environment, seed, and artifact
locations. Classroom timing remains an estimate; the package is ready for
instructor review.
