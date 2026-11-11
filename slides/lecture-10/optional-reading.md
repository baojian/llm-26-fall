# Lecture 10 reading guide

Read for a question you want to investigate; there is no fixed reference count.
Data sources and dated benchmark releases were checked on **October 26, 2026**.
The 2025 deck's benchmark-construction sources were checked on **November 10, 2026**.
Pin the actual dataset,
model, task configuration, and scorer revisions when running an experiment.
A release date or a benchmark name alone does not fully specify a result.

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

These readings support the protocol case studies in the optional extension.
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
| [LiveBench](https://livebench.ai/) and [archived version 1](https://livebench.ai/v1/) | Version 2.0 dated October 14, 2026 | Changed task suites and artifact grading. Do not equate a v1 aggregate with a v2 aggregate. |

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
