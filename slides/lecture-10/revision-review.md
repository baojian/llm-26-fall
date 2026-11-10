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

## Iteration 2: pending independent review

## Iteration 3: pending independent review

## Release validation

Final commands, artifacts and limitations will be recorded after the third
iteration. This document does not authorize publication or a merge.
