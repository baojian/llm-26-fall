# Lecture 10 reading guide

Read for a question you want to investigate; there is no fixed reference count.
The evaluation-history and benchmark sources below were checked on
**November 11, 2026 (Asia/Shanghai)**. First releases, later paper publications,
and dataset revisions are distinguished. Pin actual task IDs, data/model/harness
revisions, prompts, tools, budgets and scoring code before running a comparison.
The deck is a 95-slide material bank with a 57-slide classroom route; the
[teaching plan](teaching-plan.md) reserves 120 teaching minutes and Quiz 4.

## Evaluation history

The timeline is a teaching synthesis of changing evaluation protocols, not a
common leaderboard across generations. The deck embeds the actual source figures
listed below; crops, hashes and attribution are in
[asset provenance](assets/README.md#paper-figures-added-in-november-2026) and the
[figure manifest](assets/papers/manifest.json).

| Stage and primary source | Read | Evaluation lesson |
| --- | --- | --- |
| [GPT-1, Radford et al. (2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | §3.2–3.3; Fig. 1, page 4 | Evaluate after task-specific supervised adaptation; input transformations and label maps matter. |
| [GPT-2, Radford et al. (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Fig. 1 CoQA panel; §3.3 and §4 | Zero-shot transfer is distinct from zero pretraining exposure. Different tasks use different metrics. |
| [GPT-3, Brown et al. (2020), v4](https://arxiv.org/html/2005.14165v4) | Fig. 1.2 and 2.1; §2.4; §4 | Demonstrations and task descriptions change evaluation with fixed weights. The displayed curve is word manipulation. |
| [Codex/HumanEval, Chen et al. (2021), v2](https://arxiv.org/html/2107.03374v2) | §2; Fig. 5 (upper panel in slides); Appendix A | Executable tests, sampling temperature, attempt budget and the pass@k estimator. |
| [InstructGPT, Ouyang et al. (2022), v1](https://arxiv.org/html/2203.02155v1) | Fig. 1; §3.6 | Preference against a named reference on a defined prompt and rater population. Human evaluation differs from the reward used to train. |
| [GPT-4 Technical Report (2023), v6](https://arxiv.org/html/2303.08774v6) | Fig. 2 and 4; §4; Appendices A, C–D | Held-out capability prediction, simulated exams, modality, scoring and contamination. Exam percentiles are not occupational competence. |
| [o1 research report (September 12, 2024)](https://openai.com/index/learning-to-reason-with-llms/) | Train/test compute plots and sampling protocols | Distinguish longer reasoning, consensus and oracle candidate selection. |
| [Official OpenAI model catalog](https://developers.openai.com/api/docs/models) and [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol) | Snapshot/interface, reasoning effort, tools | GPT-5 through verified GPT-6/6.1 interfaces illustrate why the full configured system must be recorded. Documentation is not independent performance evidence. |

The requested **GPT-6.2 endpoint could not be verified** in exact official-document
searches and the fetched catalog on November 11, 2026. The material states this
boundary explicitly and does not invent a release, technical report or score.
It uses GPT-6/6.1 only as a separately named verified endpoint.

## Four benchmark case studies

| Family | Version and source | Read for |
| --- | --- | --- |
| Humanity's Last Exam | [Original paper v10](https://arxiv.org/html/2501.14249v10), Fig. 4; [release history](https://www.lastexam.ai/) | Initial release January 2025; finalized 2,500 questions April 3, 2025; Nature publication January 28, 2026. Model filtering, expert review and separate private questions. |
| HLE-Diamond | [September 22, 2026 methodology](https://labs.scale.com/leaderboard/hle-diamond) | 1,000 questions; 500 knowledge and 500 reasoning. Accuracy, calibration and tools. Different population from original HLE and HLE-Rolling; unrelated to GPQA Diamond. |
| SWE-bench | [Paper v3](https://arxiv.org/html/2310.06770v3), Fig. 1; [official sets](https://www.swebench.com/) | Issue + repository → patch → tests. Original 2,294; Lite 300; Verified 500; Multilingual 300 across nine programming languages. These are different denominators. |
| SWE-Bench Pro V2 | [September 22, 2026 release](https://labs.scale.com/blog/swe-bench-pro-v2), [paper](https://arxiv.org/abs/2610.24206) | 642 public tasks and 51-task Hard subset; repair, network policy and pristine re-grading. Separate from Verified; paper first posted October 16. |
| Terminal-Bench | [Paper v1](https://arxiv.org/html/2601.11868v1), Fig. 2; [release index](https://www.tbench.ai/benchmarks) | Figure describes TB2-era container architecture, not TB4. Instruction, environment, tests and oracle solution. |
| Terminal-Bench 3/4 | [TB3](https://www.tbench.ai/news/terminal-bench-3-0), [TB4](https://www.tbench.ai/news/terminal-bench-4-0) | Separate agent/verifier containers; resource calibration; rerun/regrade/reuse semantics. TB4's eight-hour timeout is not a course allocation. |
| Terminal-Bench Science 0.1 | [August 27, 2026 release](https://www.tbench.ai/news/terminal-bench-science-0-1) | 70 scientific workflows, with task-specific artifacts and verification. Discuss evaluation structure; no scientific wet-lab procedure is part of this lecture. |
| Agents’ Last Exam | [Paper v4](https://arxiv.org/html/2606.05405v4), Figs. 6 and 12; [project](https://agents-last-exam.org/) | First paper June 3, 2026; v4 October 25. GUI/CLI workflows, full-pass versus partial-credit scores, model/harness controls. Fig. 12 compares unequal 12/5/3 configuration sets: descriptive ranges, not a variance decomposition. Use QC-approved task IDs, not changing collected-task counts. |

Terminal-Bench first releases: 1.0 May 19, 2025; 2.0 November 7, 2025.
The paper is dated January 17, 2026. The 2026 releases are 2.1 (May 6),
Challenges (June 18), 3.0 (July 30), Science 0.1 (August 27), and 4.0 (August 28).
These are not repeated measurements of the same population under the same budget.
LiveBench’s within-v2 weight change is documented by its official [October 14](https://livebench.ai/data/20261014/categories.json), [October 25](https://livebench.ai/data/20261025/categories.json), and [November 6](https://livebench.ai/data/20261106/categories.json) category metadata.

## More 2026 benchmarks

| Source and date | Distinct question | Reproducibility requirement |
| --- | --- | --- |
| [ProgramBench](https://programbench.com/), [paper v1 May 5](https://arxiv.org/html/2605.03546v1) | Reconstruct 200 programs from executable behavior and documentation. Full resolution differs from ≥95% tests passed. | Task/test revision; query and execution budget. |
| [SWE Atlas](https://scale.com/blog/swe-atlas), [full suite May 7](https://scale.com/blog/swe-atlas-complete) | Beyond issue repair: codebase QA, test writing and refactoring; initial release March 4, 284 tasks in complete suite. | [Current methodology](https://labs.scale.com/leaderboard/swe-atlas); November v1.1 changes judge/network protocol, so name the version. Exact November day not established. |
| [OSWorld-V2](https://github.com/xlang-ai/OSWorld-V2), [paper](https://arxiv.org/abs/2606.29537) | Long desktop workflows and end states; 2.0 released June 26, paper June 28, 2.1 bug-fix release September 16. | Code, task files, assets, mocked websites and VM image together. |
| [ARC-AGI-3](https://arcprize.org/arc-agi/3), [launch March 25](https://arcprize.org/blog/arc-agi-3-launch) | Interactive exploration and human-relative action efficiency. | Task partition, action policy, feedback and score definition. Distinct from AI2 ARC. |
| [LiveBench 2.0](https://livebench.ai/), first release October 14; current checked release November 6 | Longer tasks producing reports, software and decks. | October 25 changed suite weights: Long Agentic 2, Coding/Design 1.5, others 1. Match release, coverage, weights and grader even within v2. |
| [AgentTime data](https://agenttimebench.com/data/), [benchmark map](https://agenttimebench.com/benchmarks/) | Duration instruction-following on tasks drawn from other benchmarks; initial data September 26; added agent results November 2; current v1.1 uses 200 tasks. | Pin data release; distinguish requested duration, actual duration and upstream task quality. The [method](https://agenttimebench.com/method/) also covers forecasting/retrospection; duration following is our selected example. |

Historical MMLU/GPQA/HumanEval/LiveCodeBench/MT-Bench/Arena still provide useful
diagnostics. They are not all new in 2026. A benchmark portfolio should address
specific claims and failure modes, rather than maximizing the number of names.
None of these full suites is required for the course projects.

## CS336 data and evaluation materials

Checked **November 10, 2026**, against the [Spring 2026 syllabus](https://cs336.stanford.edu/).
Lecture links below pin revision `de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e`.
They are teaching references, not copies of Stanford's full workload.

| CS336 source | Useful material | Where it fits in our course |
| --- | --- | --- |
| [Lecture 12: Evaluation](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_12.py) | Perplexity; exam, chat and agent benchmarks; realism and validity | Lecture 10's frozen protocol, task/metric choice, grader audit and interpretation; agent evaluation returns in Lecture 14. |
| [Lecture 13: Data](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_13.py) | Pretraining sources, datasets and composition | Lecture 10's provenance and counts; the optional corpus catalog. |
| [Lecture 14: Data processing](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_14.py) | Quality filtering; exact and approximate deduplication; mixtures, repetition and synthetic data | Lecture 10's filtering, duplicate families and token exposure; MinHash/LSH in optional P03. Broad synthetic-data generation remains further reading. |
| [Assignment 4: Data](https://github.com/stanford-cs336/assignment4-data) | Web-text extraction, filtering and deduplication before model training | The Lecture 02 MinHash/LSH task adapts its deduplication exercise; A2 Part 4 credits the data assignment for its fixed-budget comparison. Full Common Crawl processing is not required. |

The Lecture 02 deduplication adaptation used [A4 revision `0555bea`, Section 3.2](https://github.com/stanford-cs336/assignment4-data/blob/0555bea66369872d912652debf10b115ca0688c8/cs336_assignment4_data.pdf).
The separate Lecture 02 DSIR task draws on the DSIR paper and earlier
[CS336 data-selection examples from 2025](https://github.com/stanford-cs336/spring2025-lectures/blob/b98b08a98d9d47a69bbdcb4e96a58aa48ee4d13b/lecture_14.py#L143).
Do not describe DSIR as a required exercise in the current A4.

Our GLUE/QNLI and SWAG/HellaSwag construction cases come from the instructor's
2025/2026 decks and the primary papers below. The small corpora, source-family
audits, reward-test counterexample, plots and notebook fixtures are course-authored.
Paired uncertainty draws on statistical evaluation guidance, including Dror et al.;
it is not attributed wholesale to CS336. All five core exercises run on CPU
without downloading a Stanford corpus or training a large model.

## Core: data preparation and quality

| Reading | Focus |
| --- | --- |
| [Lee et al., Deduplicating Training Data Makes Language Models Better (2022)](https://aclanthology.org/2022.acl-long.577/) | Deduplication methods and experiments on memorization and train/evaluation overlap. Ask which findings transfer to your own corpus and budget. |
| [Penedo et al., The FineWeb Datasets (2024)](https://arxiv.org/abs/2406.17557) | Dataset-construction ablations: extraction, filtering, and deduplication. Study the comparison method rather than reproducing the full training scale. |
| [Soldaini et al., Dolma (2024)](https://arxiv.org/abs/2402.00159) and [curation toolkit](https://github.com/allenai/dolma) | Inspect source composition, transformation rules, and intermediate data states. A toolkit license and the underlying documents' rights are different metadata. |
| [Li et al., DataComp-LM (2024)](https://arxiv.org/abs/2406.11794) | Compare data policies under shared training and evaluation recipes. A full-scale benchmark run is outside the classroom budget. |

Start with FineWeb Section 3 and its [illustrated author report](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html).
Trace one ablation from its changed data policy to its retained corpus and model
comparison. Then use Lee et al. to examine overlap and memorization. Dolma and
DCLM provide alternative designs to compare, not a ranking of the latest corpora.
No external data download is required for the lecture.

The core notebook exercises E01–E05 cover exact keys, connected split groups,
token exposure, an evaluation audit, and paired uncertainty. The sampling
numbers and outcomes are authored examples. More exposure to a source is not
evidence that it improves a model; a controlled run must test that hypothesis.

## Benchmark construction as data preparation

The instructor's October 29, 2025 benchmark deck provides the following cases.
The first four readings support the classroom route; the remaining cases are
optional. Read how the data were made before interpreting the model scores.

| Reading | Section and preparation question |
| --- | --- |
| [GLUE (Wang et al., 2019)](https://arxiv.org/html/1804.07461v3) | Section 3 and Appendix A: what do labels mean, and how does converting QA into question–sentence rows change the unit? |
| [STS (Cer et al., 2017)](https://aclanthology.org/S17-2001/) | Section 2, Table 1 and Section 8: the annotation rubric uses 0–5, including zero; preserve real-valued targets. |
| [SuperGLUE (Wang et al., 2019)](https://arxiv.org/html/1905.00537v3) | Sections 3.2–3.3 and Table 3: MultiRC needs option decisions and question-level answer-set scoring. |
| [SWAG (Zellers et al., 2018)](https://aclanthology.org/D18-1009/) and [HellaSwag (2019)](https://aclanthology.org/P19-1472/) | SWAG Section 2; HellaSwag Section 3.2 and Figure 4: generation, adversarial filtering, human checks, and shortcut ablations. |
| [LAMBADA (Paperno et al., 2016)](https://aclanthology.org/P16-1144/) | Sections 3.1–3.2: select a context-dependent prediction task, and distinguish filtered evaluation passages from raw training novels. |
| [BLiMP (Warstadt et al., 2020)](https://aclanthology.org/2020.tacl-1.25/) | Sections 2–3: template generation, minimal contrasts and validation. A grammaticality pair is not automatically a response-preference pair. |
| [Dynabench (Kiela et al., 2021)](https://aclanthology.org/2021.naacl-main.324/) | Sections 2–3: model-guided collection and validation; record rounds because their populations can differ. |

E02's additional source-ID check uses six authored rows, with no benchmark
download. A question split can still share an article. The notes and slides
distinguish these course design recommendations from each benchmark's released
protocol. Do not silently re-split a benchmark while claiming its official score.

## Core: evaluation methods

| Reading | Focus |
| --- | --- |
| [Dror et al., The Hitchhiker's Guide to Testing Statistical Significance in NLP (2018)](https://aclanthology.org/P18-1128/) | Test selection and assumptions, especially dependence between predictions and the evaluation unit. |
| [EleutherAI language-model evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness) | Read the model interface and the exact task configuration: likelihood requests, generation, normalization, and aggregation. |
| [Gao et al., Scaling Laws for Reward Model Overoptimization (2023)](https://arxiv.org/abs/2210.10760) | Review of the post-training sequence: distinguish optimizing a proxy reward from independent evidence of improvement. Our square-function counterexample is an authored illustration, not a reproduction of these experiments. |

The core E05 seeded paired percentile bootstrap is a small teaching example.
It does not account for training-run variation. For related examples, decide
whether documents, repositories, or another grouping should be the sampling unit.

## Appendix: from historical examples to current protocols

These readings supplement the four core protocol cases and the optional atlas.
The main Lecture 10 route combines data quality and evaluation in 120 teaching
minutes, reserving 15 minutes for Quiz 4. MinHash is optional P03.

| Source | Version used in the discussion | What to inspect |
| --- | --- | --- |
| [MMLU-Pro paper](https://arxiv.org/abs/2406.01574) and [author repository](https://github.com/TIGER-AI-Lab/MMLU-Pro) | Introduced 2024 | Subject coverage, answer choices, and experiments on prompt sensitivity. |
| [GPQA paper](https://arxiv.org/abs/2311.12022) and [author repository](https://github.com/idavidrein/gpqa) | Original publication 2023 | Expert validation and the named subset used by a result. |
| [HLE project](https://www.lastexam.ai/) and [Diamond methodology](https://labs.scale.com/leaderboard/hle-diamond) | Diamond announced September 22, 2026 | Question-set identity, partition scores, answer extraction, calibration, and tools. |
| [LiveCodeBench paper](https://arxiv.org/abs/2403.07974) and [official repository](https://github.com/LiveCodeBench/LiveCodeBench) | Release and date window must be selected explicitly | Problem dates, model exposure, test configuration, and sample budget. A new date window is not proof of zero contamination. |
| [SWE-Bench Pro V2 release](https://labs.scale.com/blog/swe-bench-pro-v2) and [public methodology](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2) | September 22, 2026 | Task corrections, network policy, and independent re-grading. Distinguish this release from SWE-bench Verified. |
| [ARC-AGI-3](https://arcprize.org/arc-agi/3), [launch](https://arcprize.org/blog/arc-agi-3-launch), and [technical report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf) | Launched March 25, 2026 | Interaction, action efficiency, and the evaluation environment. |
| [LiveBench](https://livebench.ai/) and [archived version 1](https://livebench.ai/v1/) | First v2 October 14; checked release November 6, 2026 | Changed suites, artifact grading and October 25 suite weights. Match exact release and aggregation within v2. |

These sources supply examples of contemporary evaluation design, not a claim
that every benchmark is unsaturated or suitable for every small model. We do
not reproduce a live ranking as a stable lecture fact. Read access, API pricing,
environment setup, and total evaluation cost need a separate check before a run.

## Historical foundations and model interfaces

- [SuperGLUE (Wang et al., 2019)](https://arxiv.org/abs/1905.00537): task definitions
  and the different metrics within a suite.
- [HellaSwag (Zellers et al., 2019)](https://arxiv.org/abs/1905.07830): **Figure 4**,
  the source of the BERT-Large shortcut chart. Interpret the training/evaluation
  conditions before comparing the bars.
- [BERT (Devlin et al., 2019)](https://aclanthology.org/N19-1423/): **Section 3**,
  masked-token learning and downstream adaptation. A masked-token score is not
  an autoregressive sequence probability.
- [Evaluating Large Language Models Trained on Code (Chen et al., 2021)](https://arxiv.org/abs/2107.03374):
  **Section 3 and Appendix A**, HumanEval and the pass@k estimator.
- [Judging LLM-as-a-Judge (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685):
  the judge-bias and human-agreement experiments. Agreement in one study does
  not validate every future judge, rubric, and application.

Both supplied PPTX decks credit Anoop Sarkar. The [teaching plan](teaching-plan.md)
maps the source pages to the updated lecture. No historical leaderboard is
presented as a current result.

## A feasible follow-up

Start with a small task that matches your claim and save predictions by item ID.
Declare the baseline, data revision, prompt, parser, and budget before the final
comparison. Inspect disagreements and report relevant slices. A useful experiment
can run on CPU or a short, explicitly budgeted single-GPU session; a full frontier
benchmark is not required. Record memory per GPU and total GPU-hours across all
runs, mark unmeasured budgets as estimates, and keep a smaller fallback.

The [classroom notebook](practice.ipynb) itself is offline and CPU-only, with zero
GPU-hours. It creates no additional graded assignment.
