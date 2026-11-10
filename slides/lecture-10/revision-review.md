# Lecture 10: three-iteration revision record

Requested November 11, 2026 (Asia/Shanghai): revise the slides in three
iterations, each guided by three independent reviewers. Reviewers assess a
frozen version without reading each other's reports; the author synthesizes
priorities and owns implementation and verification.

Baseline: `eb5b75418acb147c64cb9474a2ab93d36965d602`, clean working tree.
The working branch was based on merged `b8d21e0` before edits. No student
submissions, quiz questions or assessment policies are changed.

## Iteration 1: evidence and scope

Three independent reviews covered (1) historical/technical narrative,
(2) benchmark identity and current primary sources, and (3) pedagogy, density,
visuals and quantitative examples. All found the data/uncertainty core sound;
all identified missing history and appendix-only modern benchmark coverage.

Accepted changes:

- Add a methodological timeline and actual GPT-1/GPT-2/GPT-3/InstructGPT figures.
- Preserve the GPT-6.2 request as an explicit unverified endpoint; separately
  identify the official GPT-6/6.1 documentation that was verified.
- Add source figures for FineWeb, HumanEval, GPT-4, HLE, SWE-bench,
  Terminal-Bench and Agents’ Last Exam, with complete provenance.
- Put four benchmark families into the main route; add a dated 2026 atlas.
- Add agent protocol fields, a budget-mismatch discussion and a judge-order
  audit; retain E01–E05 and P01–P03 without changing numerical assumptions.
- Rebudget to 57 core and 38 optional slides, 120 teaching minutes plus 15
  quiz minutes. Correct stale bootstrap/MinHash speaker notes.

Validation of the first revision: 95 slides passed the existing three-viewport
browser checks, including local assets, math, links and interactions. Independent
baseline notebook/numerical review passed 10 Python checks. Source figures are
not new course experiments. Review images are retained under ignored `.checks`.

## Iteration 2: precision and projection

Three fresh independent reviewers inspected frozen commit `3c726cc` without
reading prior reviews. History/source and benchmark reviews found no core
factual blocker. The visual reviewer inspected all 95 screenshots and all 14
source figures at full size, finding a plotted-metric sign error that a
text-only reading of the paper caption missed.

Accepted changes:

- Correct GPT-4's plotted **negative** mean log pass rate and explain its
  direction in large native text, alt text, notes and provenance.
- Remove neighboring BLEU-axis fragments from GPT-2 without clipping the final
  CoQA parameter tick; record the outside-panel mask and new hash.
- Enlarge Codex's upper sampling panel and give ALE readable native labels
  for fixed factors and evaluator-only references; shorten figure headings.
- State that ALE's 12/5/3-configuration ranges are descriptive, with no
  balanced variance decomposition or significance claim. Record the source's
  public-set count discrepancy rather than choosing an unsupported denominator.
- Pin LiveBench's November 6 snapshot and October 25 weight change. Clarify
  AgentTime's November 2 additions and v1.1's 200-task population.
- Add cost per solved task to the existing authored checkpoint, including
  failed-attempt cost and the zero-success boundary.
- Move E03's solved diagram after the question in both slides and notebook;
  preserve every code cell, fixture and exercise ID. Reallocate three minutes
  from history to family/article splitting and add concrete figure questions.
- Correct stale statements describing the optional schema table as core.

Validation: 95 slides again passed three viewports; 10 numerical/notebook tests
passed. Demo, notebook launcher and reading-page checks passed in this round.
No inference/training experiment or measured teaching-time claim was added.

## Iteration 3: final content and rendered-output audit

Three fresh independent reviewers assessed frozen commit `bec21f4` without
reading one another's reports. The content/teaching and benchmark/protocol
reviewers passed the material without required factual or numerical changes.
Both independently checked primary sources, all 14 asset hashes and the 10
focused Python tests. The visual reviewer inspected all 95 browser slides,
all 14 source-figure slides at full size, and all 95 exported PDF pages.

Accepted correction: the answer on `agent-comparison-check` used paired dollar
signs that KaTeX interpreted as a math span. Replace table and answer amounts
with explicit USD labels. The final export passes the browser checks; visual
inspection and PDF text extraction confirm **USD 2 versus USD 10** on page 43.
The arithmetic and experimental interpretation are unchanged.

The apparent edge loss on a FineWeb review thumbnail was checked against the
full PDF page and was only a contact-sheet artifact. No source figure was
recropped to compensate. Fine ALE/GPT-4 source labels remain close-reading
detail; large native slide text carries the classroom interpretation.

### Independent review register

Each row is a separate reviewer session. Raw reports are retained locally in
ignored `workspace/lecture10-review/`; this tracked record preserves findings,
decisions and verification without treating reviewer suggestions as facts.

| Iteration | Frozen version | Reviewer session | Assessment and resulting action |
| --- | --- | --- | --- |
| 1 | `eb5b754` | `review_history_r1` | Add evaluation history and authentic figures; keep GPT-6.2 unverified. |
| 1 | `eb5b754` | `review_benchmarks_r1` | Teach four benchmark cases in the core and add a dated 2026 atlas. |
| 1 | `eb5b754` | `review_teaching_r1` | Increase evidence density; preserve exercises and rebudget teaching time. |
| 2 | `3c726cc` | `review_history_r2` | Content passes; remove unrelated neighboring GPT-2 axis fragments. |
| 2 | `3c726cc` | `review_benchmarks_r2` | Qualify ALE ranges; pin LiveBench/AgentTime versions; define cost. |
| 2 | `3c726cc` | `review_visuals_r2` | Correct GPT-4 metric sign, enlarge source detail, move E03's answer. |
| 3 | `bec21f4` | `review_content_r3` | Pass; calculations, teaching sequence and source boundaries verified. |
| 3 | `bec21f4` | `review_protocols_r3` | Pass; primary-source protocols and numerical assumptions verified. |
| 3 | `bec21f4` | `review_visuals_r3` | Approve after currency rendering fix; author verified the regenerated page. |

## Release validation

Completed November 11, 2026 (Asia/Shanghai). The final rendered content is
`bec21f48e987a8526d3522b76d279a6256114c1d` plus the iteration-3 currency fix;
the working tree was dirty with that authorized correction during export.
The review register and teaching-plan verification were finalized afterward.
The third revision commit records these changes together.

| Command or check | Result |
| --- | --- |
| `uv run python -m pytest -q tests/test_lecture_10.py tests/test_lecture_10_data.py` | 10 passed, including execution of every notebook code cell in a fresh working copy. |
| `npm --prefix slides run check -- lecture-10` | All 95 slides passed at three viewport sizes after iteration 2. |
| `npm --prefix slides run pdf -- lecture-10` | All 95 slides passed the same checks after the final fix; 95-page PDF exported. |
| `node slides/lecture-10/check-demo.mjs http://127.0.0.1:8005` | Mixed/clean results, reset and offline interactions passed. |
| `npm --prefix slides run check:notebook -- http://127.0.0.1:8005 lecture-10` | Notebook launcher, tab reuse, kernel and static fallback passed. |
| `npm --prefix slides run build:lecture-notes` | Generated HTML synchronized with the edited notes and course plan. |
| `npm --prefix slides run check:notes` | Notes, equations, diagrams, navigation, mobile, print and offline checks passed. |
| Figure provenance | All 14 local SHA-256 values match the manifest; source versions and crops recorded. |
| Visual audit | All 95 slides/pages inspected; final changed PDF page 43 reinspected. |
| `git diff --check` | Passed. |

The PDF, screenshots and machine-readable browser report are in ignored
`slides/.checks/lecture-10/`. The source is `slides.md`; the exercise notebook
is `practice.ipynb`. No code cell, data fixture, package manifest, lockfile,
shared theme or grading requirement changed. The numerical examples use the
existing authored data and seed `20261028` (5,000 paired bootstrap draws).
Validation is local CPU/browser work; model training, live inference and GPU
usage are zero. Publisher measurements are cited evidence, not reproduced
course experiments.

Remaining limits: 120 teaching minutes is an unmeasured plan; a classroom
rehearsal and projector check remain useful. Model/benchmark facts are a dated
November 11 snapshot. GPT-6.2 was not verified in official documentation; the
separately named GPT-6/6.1 interface examples are sourced. No common score
curve across model generations is asserted. The work is prepared for review
through a pull request; this record does not authorize merging it.
