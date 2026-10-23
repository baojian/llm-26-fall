# Lecture 08 reading guide

Read for a question you want to investigate; there is no fixed reference count.
Primary sources were checked on **October 23, 2026**. Pin the actual dataset,
model, task configuration, and scorer revisions when running an experiment.
A release date or a benchmark name alone does not fully specify a result.

## Data preparation and statistical comparisons

| Reading | Focus |
| --- | --- |
| [Lee et al., Deduplicating Training Data Makes Language Models Better (2022)](https://aclanthology.org/2022.acl-long.577/) | Deduplication methods and experiments on memorization and train/evaluation overlap. Ask which findings transfer to your own corpus and budget. |
| [Penedo et al., The FineWeb Datasets (2024)](https://arxiv.org/abs/2406.17557) | Dataset-construction ablations: extraction, filtering, and deduplication. Study the comparison method rather than reproducing the full training scale. |
| [Dror et al., The Hitchhiker's Guide to Testing Statistical Significance in NLP (2018)](https://aclanthology.org/P18-1128/) | Test selection and assumptions, especially dependence between predictions and the evaluation unit. |
| [EleutherAI language-model evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness) | Read the model interface and the exact task configuration: likelihood requests, generation, normalization, and aggregation. |

The notebook's seeded paired percentile bootstrap is a small teaching example.
It does not account for training-run variation. For related examples, decide
whether documents, repositories, or another grouping should be the sampling unit.

## From the Spring examples to current protocols

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

The Spring PPTX credits Anoop Sarkar. The [teaching plan](teaching-plan.md)
records which topics were retained or compressed. No Spring leaderboard is
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
