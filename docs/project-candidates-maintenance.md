# Individual project catalog: review and maintenance

The [catalog](project-candidates.html) offers a student-proposed project route
and 48 suggested individual directions for Lecture 05 on October 10, 2026. It implements
[issue #265](https://github.com/baojian/llm-26-fall/issues/265), including the
instructor-requested Context Language Models candidate and its stable
`clm-context-management` link. The course goal of a small agent for math, code,
and science is recorded in [issue #269](https://github.com/baojian/llm-26-fall/issues/269).

## Scope and review status

The student-proposed card has its own section before the 48 suggested
candidates, which are grouped into nine topic sections. Four categories
describe approaches (case study, paper replication, new task, research
problem); a candidate can have several approach tags. Survey-only projects
are not eligible. The page supports combined
filters, keyword search, stable links, keyboard operation, and printing visible
candidates with details. Content is present in HTML without JavaScript and works
when opened directly from disk.

Every entry includes a question, scope/data, baseline/controls, evaluation,
minimum outcome, preparation, planning resources, fallback, sources, and review
status. Suggested entries are initially **Unreviewed**; the custom route is
marked **Scope to be proposed**. Source verification does not replace
instructor selection or hardware pilots. Proposed experiments and
resource estimates are not measured course results or staff allocations.
Upstream work can require paid agent trials, hosted environments, domain
expertise, and external review; each route has a local minimum outcome.

## Student page and resource screening

The instructor's October 9 refinement retains suggested readings without a
fixed reference-count requirement. Projects should fit the resources students
can access, and the course-pool caution is kept to one sentence. The student page
now has seven sections in the requested order: preparation and overview,
resource requirements, student-proposed projects, open-source contributions,
previous projects and papers, 48 candidate projects, and other guidance. A
sticky left contents panel links each section and the nine candidate topics.
It collapses on narrow screens;
topic links filter the candidate list, and ordinary anchors work without
JavaScript. Each
card displays its references, why to read them, device/memory target, total
GPU-time ceiling, and main cost risk before the detailed experiment plan.
The historical course comparison stays in these notes. Detailed upstream
contribution guidance is collapsed within the fourth section.

- All 49 cards retain their annotated readings. The open-choice card links
  planning guidance; students connect their question to relevant prior work
  without a numeric citation requirement.
- Ten minimum studies require no GPU. The other suggested studies have a
  one-GPU minimum. Speculative decoding and the tiny-model scaling study also
  offer optional two-GPU configurations; the custom route asks students to
  specify CPU, one-GPU, two-GPU, or larger needs. Three or more devices are
  flagged as high demand for the course pool. Resource labels distinguish
  minimum hardware from optional parallel runs.
- The initial resource screen proposes at most eight GPU-hours per project
  from the course pool,
  with inference cards usually targeting two or four. This numeric ceiling is
  provisional pending the instructor's budget choice; it is not an allocation
  or measured runtime. `resource_policy.status` records that distinction.
  Students may use resources they obtain themselves; the proposed pool budget
  does not cap their total external resource use.
- Every total includes all devices, baselines, tuning, evaluation, ablations,
  and reruns. Two GPUs for two hours consume four GPU-hours. Memory targets
  are per GPU, and two devices do not automatically pool their memory.
  Students should time a small pilot and reduce scope if necessary. Hardware
  compatibility and memory figures remain unpiloted planning targets.
- Byte patching now uses a CPU n-gram predictor. The reward project studies
  verifier errors offline. Chunking uses BM25 evidence retrieval; ALE and HLE
  have CPU infrastructure/parsing minima. Large online RL, full LLM training,
  and full benchmark reruns remain outside the suggested minimum scopes.
- The former survey candidate (#30) is now a CPU cache-memory measurement
  study: implement an estimator and allocation harness, compare predictions
  with measured tensor storage, and add regression tests. Its stable ID is
  preserved. Every project requires practical or experimental evidence.
- Adapter distillation and per-example unlearning start with much smaller
  samples. Tiny-model training retains matched controls and a total run cap.
- The visual-preparation study now asks about one small model. The skills
  study includes a generic prompt at matched length/budget. The visual
  retrieval study compares modalities within one encoder before comparing
  different systems.

Primary arXiv records for the 33 added research references were checked for
title, authors, and topic on October 9. Their years identify the arXiv release;
the catalog does not infer conference acceptance or claim full replications.
Reading notes identify their role in the proposed study, not reported course
results. The PyTorch tutorial is an official implementation reference; students
must match instructions to their installed backend. Existing recent model and
benchmark links retain the earlier source checks described below.

The renderer rejects missing notes for supplied readings, duplicate references
or reference URLs, invalid GPU-count ranges, inconsistent
CPU labels, survey categories, and shared-pool GPU-hour plans above the
proposed ceiling. Higher GPU counts are supported and labeled as high demand.
`min_gpus` defaults to `max_gpus`; when different, the card explains the
optional configuration. `gpu_memory_gb` is per device, and `gpu_hours` is the
total across all devices and runs. Optional `shared_gpu_hours` separates a
course-pool request from externally supplied time; it defaults to the total.
Human review is still needed for reference relevance and pilot feasibility.

## Changes from Spring 2026

Sources were the [previous final-project deck](https://baojian.github.io/llm-26/slides/final-project/index.html),
its [embedded example-task handout](https://baojian.github.io/llm-26/slides/final-project/media/example-tasks.pdf),
and its 20-paper list. Its group size, report length, dates, and grading
breakdown do not apply to Fall 2026.
Section 5, also linked from the sidebar, points students to this list and
suggests implementing and evaluating a paper's method under the current requirements.

| Earlier direction | Fall treatment |
| --- | --- |
| SemEval-2014 sentiment; Fake News Challenge; 2019/2020 commonsense validation/explanation | Retired as standalone prompts; usable as baselines or controlled robustness data. |
| Generic Kaggle, toxicity, essay scoring, fraud/product classification, or news similarity | Replaced by specific questions about sample efficiency, evaluation validity, retrieval, or robustness. This is not a claim that these applications are obsolete. |
| BLT; sparse attention; difference-aware fairness; cross-lingual intervention; formal-language pretraining; Infini-gram mini | Retained as reduced mechanism studies or local experiments, with explicit deviations from full-paper reproduction. |
| Reasoning faithfulness through unlearning | Updated with the ACL 2026 study of intervention artifacts and controls. |
| Group projects and broad paper reproduction | Replaced by one-student scopes with concrete minimum outcomes. |

New directions include 2026 Engram work, Context Language Models, thought
sufficiency, Qwen-Scope, code-switching retrieval, agent memory/security, dLLM,
small multimodal models, and upstream benchmark contributions. Publication
labels distinguish preprints, conference papers, software, and model cards.

## Upstream credit and current openings

On October 9 the instructor asked to consider additional project credit for
work accepted and merged into established repositories. The catalog records
this intent and requests evidence of substantive individual contribution and
external acceptance. Numeric points, a maximum-score change, an upstream merge
deadline, and acceptance without a GitHub merge have **not** been decided.
The existing rubric remains in force. Any later point policy must apply
equally to all students and be reconciled with `AGENTS.md` and the assessment
page before publication as a grading rule.

A sound local investigation remains assessable if a maintainer rejects a
contribution or cannot review it before the course deadline. The catalog does
not promise authorship, acceptance, or automatic extra points. Graded proposals,
reports, and experiment records still go to eLearning. A separately agreed
public contribution follows upstream requirements and must not expose private
course solutions, grades, or protected evaluation material.

Primary contribution sources checked on **October 9, 2026**:

| Destination | Finding and implication |
| --- | --- |
| [TB-Science guide, revision `8ef49c8`](https://github.com/harbor-framework/terminal-bench-science/blob/8ef49c8a64cf57f9913755b41ed6a0fe1a667192/CONTRIBUTING.md) | The 0.2 PR cutoff is October 5. New tasks need approved proposals and a confirmed future intake. Task-fix requests have an explicit route. Tasks must meet the [scientific-domain and difficulty rubric](https://github.com/harbor-framework/terminal-bench-science/blob/8ef49c8a64cf57f9913755b41ed6a0fe1a667192/rubrics/task-proposal.md). |
| [Agents' Last Exam](https://github.com/rdi-berkeley/agents-last-exam) | Public framework and tasks; documented agent integrations; hidden references staged after execution. Scope a compatible public subset rather than the full VM/GUI benchmark. |
| [HLE README, revision `22ed307`](https://github.com/centerforaisafety/hle/blob/22ed3074b1e7b134bcbc09028d0ba320839b0655/README.md) and [official site](https://lastexam.ai/) | Question feedback form and maintainer contact for HLE-Rolling; GitHub issues disabled. Confirm the contribution route and preserve benchmark canaries/training exclusion. Question acceptance need not involve a GitHub merge. |
| [Context Language Models, revision `18dc111`](https://github.com/facebookresearch/context-language-models/tree/18dc11115f50f261233c5bba7937834491e307e8) | Code is available; ContextBench is still listed as coming soon. Use a documented original generator and label reduced experiments as adaptations. |

Primary papers and official model/code releases establish starting methods and
artifacts, not proposed student runtime or outcomes. Recheck external access
before selection. Students should pin code/model/data revisions in their
private experiment records.

## Comparison with other courses

The initial 45 entries drew on the previous Fudan list and primary research.
On October 9, the instructor requested comparison with other courses. The
comparison is retained here for instructors; it is omitted from the
student page so students can focus on proposals and resource choices. Sources
are the official [CS336 Spring 2026 assignments](https://cs336.stanford.edu/),
[CS224N Winter 2026 gallery](https://web.stanford.edu/class/cs224n/project.html),
[CS224N report guidance](https://web.stanford.edu/class/cs224n/project/Project_Report_Instructions.pdf),
and [CMU Advanced NLP Fall 2025 project requirements](https://cmu-l3.github.io/anlp-fall2025/assignments/assignment3%264).
CMU is explicitly labeled as the 2025 offering; this review does not claim to
have checked a 2026 Advanced NLP syllabus or all linked student reports.

| Reference course | Pattern retained in our design |
| --- | --- |
| CS336, Spring 2026 | Start from working model/data code, then isolate a systems or scaling question. Projects 46–47 use local tiny-model studies. |
| CS224N, Winter 2026 | Use a concrete question about efficiency, reasoning, memory, or clarification; its gallery informed project 48. |
| CMU Advanced NLP, Fall 2025 | Move from literature review to a baseline, then an extension and analysis across the written checkpoints. |
| UMD CMSC 723, Fall 2026 | Connect personal motivation to prior work, data, baseline, evaluation, compute, and a schedule in the proposal. |

External course deadlines, team sizes, and grading rules do not transfer to
this course. These are design references for instructors rather than another
set of student requirements.

That review added projects 46-48: small-scale scaling prediction, training
profiling, and clarification before coding. Existing candidate numbers and IDs
were preserved. The [CMU RAG assignment](https://github.com/cmu-l3/anlp-fall2025-hw2)
also informed project 16; its student reading list now emphasizes the retriever
and evaluation references directly.
The new scopes are original adaptations with local fallback routes, not full
copies of the external assignments or claims of reproducing student results.

The instructor also set a six-page maximum for the final report's main content
on October 9. References and appendices remain outside the limit; code and an
experiment record are still required. This decision is recorded in `AGENTS.md`
and reflected in the homepage, translation, template, and generated catalog.

The instructor subsequently supplied the UMD CMSC 723 proposal template and
requested student-chosen interests as the first category. The open-topic entry
has the stable ID `student-proposed-project` and internal number 49, displayed
as **Open choice** in its own section before projects 01-48. `display_label`
supplies that label. Existing suggested-project IDs and numbers are unchanged.
The same individual-project rubric applies to this route.

The [UMD schedule](https://www.cs.umd.edu/~miyyer/cmsc723/schedule.html) was checked
on October 9 and links the [proposal template](https://www.overleaf.com/read/jrxnkfqvkjfm).
Its planning prompts inform the open-topic card: motivation, related work,
baseline, data, evaluation, resources, tools, and schedule. The template text
was supplied by the instructor; the web reader did not expose the Overleaf
document contents. UMD's group size, deadlines, proposal minimum length,
citation minimum, and AI-detector grading rule are not adopted. Our one-page
proposal, October 21 deadline, six-page final main report, and eLearning
submission process remain the applicable requirements.

## Files and rebuild

| File | Role |
| --- | --- |
| [project-candidates.json](project-candidates.json) | Editable entries, shared budgets, and source records. |
| [project-candidates.template.html](project-candidates.template.html) | Course guidance, milestones, contribution policy, and page structure. |
| [build_project_candidates.py](../scripts/build_project_candidates.py) | Standard-library renderer, escaping, and data validation. |
| [project-candidates.css](../assets/project-candidates.css) | Layout alongside shared reader styles. |
| [project-candidates.js](../assets/project-candidates.js) | Search, filters, deep links, detail expansion, and printing. |
| [test_project_candidates.py](../tests/test_project_candidates.py) | Build, data, escaping, and local-link checks. |

From the repository root:

```sh
uv run python scripts/build_project_candidates.py
uv run python scripts/build_project_candidates.py --check
uv run python -m pytest -q tests/test_project_candidates.py
uv run python scripts/slides.py serve
```

Open `http://127.0.0.1:8000/docs/project-candidates.html`. Commit JSON, template,
and generated HTML together. No Node build, model download, external font, or
runtime network request is needed. Content is English; the header preserves
the existing shared course-language control.

Check sidebar navigation, topic links under active filters, combined filters,
multi-tag entries, paper search, no-results recovery,
deep links after filtering, keyboard access, narrow screens, print, and
eLearning links. Verify dates against [the schedule](schedule.md). Do not infer
new grade weights or deadline exceptions from an upstream opportunity.
