# Individual project catalog: review and maintenance

The [catalog](project-candidates.html) prepares 45 individual project directions
for Lecture 05 on October 10, 2026. It implements
[issue #265](https://github.com/baojian/llm-26-fall/issues/265), including the
instructor-requested Context Language Models candidate and its stable
`clm-context-management` link. The course goal of a small agent for math, code,
and science is recorded in [issue #269](https://github.com/baojian/llm-26-fall/issues/269).

## Scope and review status

There are five candidates in each of nine topic areas. Topic and approach are
separate: a candidate can have several approach tags (survey paper, case study,
paper replication, new task, research problem). The page supports combined
filters, keyword search, stable links, keyboard operation, and printing visible
candidates with details. Content is present in HTML without JavaScript and works
when opened directly from disk.

Every entry includes a question, scope/data, baseline/controls, evaluation,
minimum outcome, preparation, planning resources, fallback, sources, and review
status. All entries are initially **Unreviewed**: source verification does not
replace instructor selection or hardware pilots. Proposed experiments and
resource estimates are not measured course results or staff allocations.
Upstream work can require paid agent trials, hosted environments, domain
expertise, and external review; each route has a local minimum outcome.

## Changes from Spring 2026

Sources were the [previous final-project deck](https://baojian.github.io/llm-26/slides/final-project/index.html),
its [embedded example-task handout](https://baojian.github.io/llm-26/slides/final-project/media/example-tasks.pdf),
and its 20-paper list. Its group size, report length, dates, and grading
breakdown do not apply to Fall 2026.

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

Check combined filters, multi-tag entries, paper search, no-results recovery,
deep links after filtering, keyboard access, narrow screens, print, and
eLearning links. Verify dates against [the schedule](schedule.md). Do not infer
new grade weights or deadline exceptions from an upstream opportunity.
