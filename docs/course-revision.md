# Course progress and the remaining six lectures

## Current progress: November 10, 2026

Weeks 8 and 9 reflect the actual teaching: InstructGPT/SFT/reward models,
then REINFORCE/PPO. GRPO and DPO remain prepared supplementary material,
not topics already taught. Week 10 combines data preparation and quality with
evaluation and experimental design. The [current schedule](schedule.md) and
[notes index](README.md) are the active references. The next six meetings are
Lectures 11–16. They advance from the taught post-training foundations to
preference optimization, retrieval, inference, diffusion, agents and research synthesis.
Existing assessment dates and weights remain unchanged.

Lectures 01–10 have slide and notebook packages. The sections below are
**preparation plans for Lectures 11–16**, not a claim that six new decks or demos
are ready. Lecture 10 has 45 core slides, five offline exercises and an optional
appendix; its [teaching plan](../slides/lecture-10/teaching-plan.md) reserves
15 minutes for Quiz 4 after 120 minutes of instruction.

The [schedule adjustment proposal](schedule-adjustment-proposal.md) records the
approved assessment dates and the October 9 proposal-deadline extension.
The [course page](../index.html#schedule)
and [dated schedule](schedule.md) carry the matching sequence. All planning
dates and class times use **Asia/Shanghai (UTC+08:00)**.

## Design decisions

- Assume Python and an introductory machine-learning course. Teach the PyTorch
  details needed for the common exercises, and provide a preparation notebook.
- Plan 16 teaching meetings. The Week 5 meeting moves from October 7 to
  Saturday, October 10. Teach Transformers there and advance later topics by
  one meeting. Use Week 16 for research synthesis and project revision.
- Retain NLP foundations and the useful spring notebooks. Add explicit data,
  training, resource-accounting, and evaluation work from CS336.
- Use one individual project per student. No presentations, posters, oral
  defenses, assigned project pods, or shared group deliverables are required.
- Allow substantial AI assistance. Grade the question, design, evidence,
  interpretation, and reproducibility. More experimental iteration is encouraged;
  code volume and application feature count are not learning objectives.
- Keep diffusion as one late course session and an advanced project option.
  Remove the requirement to interpret every early topic as a denoising process.
- Write teaching materials in English, following the repository guidelines.

## Teaching sequence

Each meeting has 135 minutes of instruction, excluding breaks. The default is
three 45-minute periods: concepts/worked example, experiment/interpretation,
and guided practice. Quizzes take 15 minutes from practice and cover material
already taught. Advanced work is an extension choice rather than a second
mandatory curriculum delivered during every meeting.

| Week | Date | Topic | Evidence students should be able to produce |
| ---: | --- | --- | --- |
| 1 | Sep 9 | Introduction and tokenization | A correct BPE trace and bilingual tokenization comparison |
| 2 | Sep 16 | Language models, cross-entropy, and perplexity | A bigram baseline and a checked loss calculation |
| 3 | Sep 23 | Embeddings and PyTorch | A tensor trace through a small neural LM |
| 4 | Sep 30 | Neural LMs and recurrent attention | A tiny training loop, recurrent-state trace, and additive source alignment |
| 5 | Sat, Oct 10 | The Transformer as a working model | A decoder block with shape and causal-mask checks |
| 6 | Oct 14 | Pretraining and decoding | A training curve, resumed run, and diagnosed failure |
| 7 | Oct 21 | Compute budgets and scaling; Jiaxuan Zou guest session | A compute estimate and a scaling prediction checked at a held-out scale |
| 8 | Oct 28 | InstructGPT, SFT and reward models — taught | Response-token masking and a checked preference-pair likelihood |
| 9 | Nov 4 | REINFORCE, PPO and GAE — taught | A policy-gradient trace; old-policy versus reference-policy roles |
| 10 | Nov 11 | Data preparation, evaluation and experimental design | A controlled data-policy comparison, leakage audit and paired uncertainty |
| 11 | Nov 18 | DPO, GRPO and verifiable rewards | A preference-loss calculation, group-relative advantage and verifier audit |
| 12 | Nov 25 | Retrieval and RAG | Separate measurements of retrieval and generation failures |
| 13 | Dec 2 | Efficient inference and computation at test time | A latency/memory comparison and quality-versus-attempt-budget curve |
| 14 | Dec 9 | Masked diffusion language models | A sampling comparison with explicit assumptions and budget |
| 15 | Dec 16 | Agents and tool use | A bounded tool-use evaluation with inspected traces and costs |
| 16 | Dec 23 | Research synthesis and project revision | Project claims linked to evidence, limitations, and a revision plan |
| 17 | Dec 30 | Final submission; no lecture | Report, code, and experiment record |

BERT is introduced as a bidirectional architecture/objective contrast in Week 5,
revisited through classification in Week 10, and connected to retrieval and
masked generation in Weeks 12 and 14. It has no standalone survey week.
Week 15 uses a bounded agent example; Week 16 revisits course methods and
guides revision of project claims, evidence, and writing.

All 16 meetings retain instruction and practical work. Project checkpoints
are written submissions; the final synthesis session adds guided practice.

The instructor moved scaling to Week 7 for Jiaxuan Zou's guest talk,
[How to Build a Scientific Scaling Ladder](https://jiaxuanzou0714.github.io/blog/2026/how-to-build-scientific-scaling-ladder/),
in the third period. Data and evaluation are now combined in Week 10, following
the actual SFT/RM/PPO coverage in Weeks 8–9. Assessment dates
remain as listed in the [current schedule](schedule.md).

## Lecture 11: DPO, GRPO and verifiable rewards

**November 18 · 135 minutes.** When can a preference or reward improve the
behavior we actually care about? Start from the taught SFT/RM/PPO objectives
and Lecture 10's independent evaluation boundary.

| Period | Teaching route | Student evidence |
| --- | --- | --- |
| 1, 45 min | From a reference-relative preference margin to the DPO loss; distinguish offline pairs from on-policy samples | Compute one loss and explain which response is pushed up relative to the reference |
| 2, 45 min | GRPO: sample a response group, compare rewards, normalize advantages, retain the PPO ratio and clipping distinction | Calculate group advantages; diagnose an all-equal-reward group and its zero learning signal |
| 3, 45 min | Verifiable rewards, incomplete tests and reward exploitation; compare DPO/PPO/GRPO data requirements | Audit a toy verifier and specify independent evaluation, token budget and failure slices |

Read [DPO, Sections 3–4](https://arxiv.org/abs/2305.18290) and
[DeepSeekMath, Section 4](https://arxiv.org/abs/2402.03300).
[Reward model overoptimization](https://arxiv.org/abs/2210.10760) is an optional
empirical case. Reuse Lecture 09's DPO/GRPO supplements as source material;
they have not been taught. Prepare a CPU arithmetic notebook and a verifier
counterexample before adding optional live sampling. Full RL training is not
needed for these learning objectives.

## Lecture 12: Retrieval and RAG

**November 25 · 135 minutes.** When should a model retrieve evidence, and
which stage caused a wrong answer? Use a small course-notes collection with
document IDs carried from retrieval through citation scoring.

| Period | Teaching route | Student evidence |
| --- | --- | --- |
| 1, 45 min | BM25 and dense retrieval; query/document representations; Recall@k | Rank a small collection and identify missed supporting documents |
| 2, 45 min | Chunk boundaries, reranking and context assembly under a fixed budget | Compare two chunking or retrieval policies on the same questions |
| 3, 45 min | Retrieved evidence versus generated claims; citation support and abstention | Separate retrieval failure, unsupported generation and unanswerable cases |

Read [DPR, Sections 2–4](https://arxiv.org/abs/2004.04906) and
[RAG, Sections 2–3](https://arxiv.org/abs/2005.11401). Prepare checked relevance
labels and answerable/unanswerable questions, then preserve a held-out split.
The CPU route can inspect saved responses; a live generator is optional.
The existing project progress update is due that evening; the classroom
comparison adds no separate graded submission.

## Lecture 13: Inference and computation at test time

**December 2 · 135 minutes.** How should a fixed inference budget be spent?
Connect Lecture 07's compute allocation to Lecture 10's protocol and Lecture
11's verifier. Separate systems efficiency from spending more attempts.

| Period | Teaching route | Student evidence |
| --- | --- | --- |
| 1, 45 min | Prefill, decoding and KV-cache size; latency versus throughput; attention I/O | A dimension-checked cache-memory estimate and a workload definition |
| 2, 45 min | Batching, cache reuse and quantization; warm-up and repeated measurements | A correctness-checked timing/memory comparison with model, hardware and token counts |
| 3, 45 min | Best-of-N, majority vote and verification; actual selection versus oracle pass@k | Plot success against attempts, total tokens and wall time; state selection cost and limits |

Read [FlashAttention, Sections 2–3](https://arxiv.org/abs/2205.14135),
[PagedAttention/vLLM, Sections 3–4](https://arxiv.org/abs/2309.06180), and
[Snell et al., test-time compute](https://arxiv.org/abs/2408.03314).
Prepare a small CPU cache example and auditable candidate traces. An optional
GPU profile must report its own hardware; it cannot turn a CPU trace into a
measured GPU speedup. A3 keeps its published deadline that evening.

## Lecture 14: Masked diffusion language models

**December 9 · 120 teaching minutes + 15 minutes for Quiz 5.** What changes
when a generator can update multiple positions? Retrieve BERT's masked
prediction objective and the causal decoder as distinct starting points.

| Period | Teaching route | Student evidence |
| --- | --- | --- |
| 1, 45 min | A discrete masking process and time-dependent denoising loss | Trace one corruption/target pair; explain why BERT alone is not a complete diffusion generator |
| 2, 45 min | Iterative unmasking, conditioning and sampling schedules | Trace token states through a small sampler and identify its stopping rule |
| 3, 30 min + quiz | Compare quality and cost against autoregression; then Quiz 5 | State step count, token budget, model differences and measured timing limits |

Read [MDLM, Sections 3–4](https://arxiv.org/abs/2406.07524) and
[LLaDA, Sections 2–3](https://arxiv.org/abs/2502.09992). Prepare a CPU toy
sampler with explicit assumptions. Updating positions in parallel does not
establish an end-to-end speedup; report repeated passes and actual runtime.
Quiz 5 covers previously taught core material, not new diffusion content or
optional extensions; its questions remain in the instructor repository.

## Lecture 15: Agents and tool use

**December 16 · 135 minutes.** When is a tool-use loop worth its additional
cost? Extend a direct-answer baseline into a bounded observe–act–tool loop.

| Period | Teaching route | Student evidence |
| --- | --- | --- |
| 1, 45 min | State, tool schema, observations and stopping conditions | Trace one successful episode and one failed tool call |
| 2, 45 min | Tool permissions, untrusted tool text, recovery and action limits | Inspect a prompt-injection fixture and distinguish an observation from an authorized instruction |
| 3, 45 min | Direct generation versus a bounded agent on the same tasks | Report success, failures, tool calls, tokens, latency and stopping behavior |

Read [ReAct, Sections 2–3](https://arxiv.org/abs/2210.03629).
[SWE-bench](https://arxiv.org/abs/2310.06770) is an optional evaluation-design
case, not a required full benchmark run. Prepare an offline tool environment
and replayable traces; network access and paid APIs are unnecessary for the
core lesson. Training large agents or coordinating many agents is optional reading.

## Lecture 16: Research synthesis and project revision

**December 23 · 135 minutes.** Which project claims survive a careful check?
Bring one current result, its baseline and its experiment record; students
work individually and use the same criteria regardless of degree program.

| Period | Teaching route | Student evidence |
| --- | --- | --- |
| 1, 45 min | Map question → comparison → evidence → claim | Identify one unsupported inference and the smallest useful follow-up |
| 2, 45 min | Inspect leakage, uncertainty, resource accounting and negative results | Repair a missing control or narrow a claim to the tested population |
| 3, 45 min | Revise figures, methods and limitations; check reproducibility | A revision checklist for the report, code and experiment record |

Use the [project guidance](project-candidates.html),
[Lecture 10 experiment-report template](lecture-10-data-evaluation.html#11-a-minimum-useful-experiment-report)
and the project's own primary papers. Prepare an authored example report
and an evidence checklist, keeping student work private. The report allows
six pages of main content; references and appendices are excluded from the
limit. Final submission remains December 30, 23:59 (Asia/Shanghai).
No presentation, poster or new oral assessment is introduced.

## Preparation status and compute budgets

The six routes above are plans. Each needs a finished deck, checked examples
and a classroom rehearsal. Preserve a CPU route for every lesson. The following
optional live-demo budgets are **unmeasured planning estimates**, not course-pool
allocations or claims about an already-tested model.

| Lecture | Minimum route to prepare | Optional live rehearsal: memory per GPU; total GPU-hours |
| --- | --- | --- |
| 11 | CPU loss/advantage calculations and verifier audit; 0 GPU-hours | One 16–24 GB GPU; up to 0.5 GPU-hours for small-model sampling across all runs |
| 12 | CPU retrieval and saved responses; 0 GPU-hours | One 16–24 GB GPU; up to 0.25 GPU-hours for generation across all runs |
| 13 | CPU cache example and saved candidate traces; 0 GPU-hours | One 16–24 GB GPU; up to 0.25 GPU-hours for profiling across all runs |
| 14 | CPU toy masked sampler; 0 GPU-hours | One 16–24 GB GPU; up to 0.5 GPU-hours only if a suitable checkpoint fits; otherwise use the CPU route |
| 15 | CPU tools and replayed episodes; 0 GPU-hours | One 16–24 GB GPU; up to 0.25 GPU-hours for bounded generation across all runs |
| 16 | CPU inspection of existing project evidence; 0 GPU-hours | None required |

These are per-lecture totals across devices and attempts, not per-run allowances.
Pin the model and data before a live rehearsal; measure memory/runtime and reduce
scope if the estimate does not hold. Record any external compute source and cost
separately from a course-pool request. No multi-GPU training is required.

## Spring material to retain, shorten, and move

The following are changes to required fall coverage. Keep the spring materials
available as references; do not delete their source files.

| Spring material | Retain or reuse | Remove from core or relocate | Use the time for |
| --- | --- | --- | --- |
| L01 introduction/tokenization | Ambiguity, BPE, vocabulary decisions, bilingual examples | Long product/history tour; regex tutorial moves to preparation | Bytes/Unicode, round-trip checks, and tokenizer tradeoffs |
| L02 n-grams | Chain rule, MLE, sampling, held-out loss, additive smoothing, interpolation | Katz backoff and Good–Turing/Kneser–Ney details become optional; consolidate repeated NPLM introduction | Explicit NLL/cross-entropy connection and baseline interpretation |
| L03 classification/embeddings | Distributional hypothesis, embedding lookup, one skip-gram example, similarity, contextual representations | Repeated NB/LR derivations and full word2vec gradient sequence; GloVe/SVD/fastText survey optional; TF-IDF moves to retrieval | Batching, autograd, output projections, and shape reasoning |
| L04 neural LMs | Existing PyTorch introduction, small feedforward LM, tiny-batch training | Micrograd and full LSTM training remain optional; Q/K/V and causal self-attention move to L05 | Training-loop diagnosis, recurrent memory and gates, and Bahdanau alignment |
| L05 attention/Transformers | Attention intuition, Q/K/V, multiple heads, residuals, normalization | Repeated BPE walkthrough, detailed MT results, constituency parsing, long encoder–decoder tour | One consistent decoder implementation and modern component comparisons |
| L06 GPT/pretraining | Existing GPT training notebook, AdamW, scheduling, checkpoint use, HellaSwag example | ELMo history and repeated beam-search variants shortened; model-generation catalog optional | Make training configuration and failure diagnosis explicit outcomes |
| L07 benchmarks | Representative task examples and distinctions between loss and downstream scores | Dataset-by-dataset GLUE/SuperGLUE catalog and many leaderboard slides become reference tables | Contamination, prompt sensitivity, uncertainty, metrics, and error analysis |
| BERT notebook | Masked prediction, contextual embeddings, classification example | Standalone BERT/SpanBERT family survey | Integrate architecture comparison with evaluation and retrieval |
| SFT/RM/PPO notebook | SFT, response-token masking, preference data, reward diagnostics; PPO/GAE as taught | Full PPO training implementation optional; no required dual-family RL experiment | Weeks 8–9 foundations followed by Week 11 DPO/GRPO and reward audits |
| L10 retrieval/RAG | TF-IDF/BM25, dense retrieval, reranking, pipeline and evaluation | Detailed ColBERT internals optional | Evidence support, abstention, and retrieval-versus-generation failure analysis |
| Frontier sessions | Practical inference, one diffusion session, one agent task | Repeated RLHF content folded into alignment; broad frontier survey optional | Workload-specific measurement and focused project investigation |

Spring sources inspected:
[course outline](https://baojian.github.io/llm-26/),
[slide sources](https://github.com/baojian/llm-26/tree/main/slides),
[neural-LM notebook](https://github.com/baojian/llm-26/blob/main/lecture-04-neural-lms/lecture-04-neural-lms.ipynb),
[GPT notebook](https://github.com/baojian/llm-26/blob/main/lecture-06-gpts/lecture-06-gpts.ipynb),
and [post-training notebook](https://github.com/baojian/llm-26/blob/main/lecture-08-sft-rm-ppo/lecture-08-sft-rm-ppo.ipynb).

## CS336 incorporation map

Lecture numbering follows the [Spring 2026 CS336 schedule](https://cs336.stanford.edu/).
These are selected adaptations with local datasets, starter code, and compute
budgets. The full Stanford assignment workload and infrastructure are not
requirements for this course.

| Source | Fall use | Adaptation boundary |
| --- | --- | --- |
| L01 tokenization; A1 basics | Weeks 1–2; A1 | Small correctness-focused tokenizer exercise; no high-throughput implementation requirement |
| L02 PyTorch and resource accounting | Weeks 3, 6, 7 | Tensor mechanics and resource estimates; detailed hardware arithmetic optional |
| L03 architectures; A1 components | Week 5; A2 | One coherent decoder baseline; one modern-component ablation optional |
| L13–14 data; A4 | Week 10; Lecture 02 tasks, A2 or project | Small supplied corpus, filtering/deduplication, and controlled comparison |
| L09/L11 scaling; A3 | Week 7 | Fit supplied run results and optionally add small local runs; no Stanford API dependency |
| L12 evaluation | Week 10 and project | Choose appropriate tasks, audit validity, and qualify conclusions |
| L15–16 post-training; A5 | Weeks 8–9 and 11; A3 context | SFT/RM/PPO foundations, then DPO and GRPO worked examples; full RL training optional |
| L10 inference; A2 systems selections | Week 13 | Measure caching/batching/quantization; custom kernels and distributed training optional |

Source repositories:
[lectures](https://github.com/stanford-cs336/lectures),
[A1 basics](https://github.com/stanford-cs336/assignment1-basics),
[A2 systems](https://github.com/stanford-cs336/assignment2-systems),
[A3 scaling](https://github.com/stanford-cs336/assignment3-scaling),
[A4 data](https://github.com/stanford-cs336/assignment4-data),
[A5 alignment](https://github.com/stanford-cs336/assignment5-alignment).
Before preparing handouts, pin the exact source revision: public repositories
can change independently of a course schedule, and README year labels can lag.
The [Lecture 10 source map](../slides/lecture-10/optional-reading.md#cs336-data-and-evaluation-materials)
pins the data/evaluation readings and distinguishes Stanford adaptations,
the instructor's benchmark slides and our own fixtures.

## Individual projects and AI assistance

Students choose a meaningful uncertainty rather than merely a list of features
to implement. Building a system is welcome when it enables the investigation.
The proposal establishes importance, prior work, plausible alternatives,
evaluation, and feasibility. Careful replication, useful failure analysis, or
a sound negative result can earn full credit.

Written milestones are Week 7 proposal, Week 12 progress update (including a
baseline and initial results), and Week 17 final report. These replace the former posters, oral defenses,
pod comparison pages, and mandatory reviewer roles. Students can reuse and revise
checkpoint text in the final report. Peer discussion and individual office-hour
consultations remain available.

AI assistance is allowed for implementation, engineering, research exploration,
and iteration. Students verify sources, check computations, inspect examples,
and own their conclusions. Keep a concise record of important decisions and
traceable experiments; do not require a full chat transcript or an artificial
list of rejected AI suggestions. Quizzes and designated concept checks assess
individual understanding without AI.

The tiny model trained in A2 supports learning about pretraining. A supplied
pretrained checkpoint supports SFT and application experiments. Projects on
later topics can start from a supplied baseline; students need not wait for the
relevant lecture or depend on their own early training run succeeding.

## Assessment context and workload

Use the published category weights: quizzes 10%, practical assignments 45%,
and individual project 45%. A1, A2, and A3 each contribute 15%.
They establish common skills and can provide infrastructure for the project;
the final project identifies its additional question and experiments rather
than receiving duplicate credit for the same submission.

| Submission | Released | Due |
| --- | --- | --- |
| A1: Tokenization and language models | Week 2, Sep 16 | Week 4, Sep 30 |
| A2: Build and investigate a small LM | Week 5, Oct 10 | Week 8, Sat, Oct 31 |
| A3: Adapt and evaluate | Week 10, Nov 11 | Week 13, Dec 2 |
| Project proposal | Project introduced Week 1 | Week 7, Oct 21 |
| Project progress update, including baseline and initial results | Follows proposal feedback | Week 12, Nov 25 |
| Final report, code, experiment record | Reuses the written checkpoints | Week 17, Dec 30 |

Quiz weeks are 3, 5, 7, 10, and 14. All five quizzes count, each contributing
2% of the course grade. A2 and A3 each have a three-week work window. All
submission deadlines are at 23:59 (Asia/Shanghai); A2 is due on Saturday,
October 31. No practical deadline coincides with a project deadline or the
holiday. Core data-quality and scaling exercises can use supplied results
when extra training would exceed the pilot-tested budget.

Within the project grade, propose question/motivation 20%, experimental design
25%, evidence/reproducibility 25%, interpretation/iteration 20%, and written
communication/checkpoints 10%. Undergraduate and master's students have the same
required work, learning objectives, rubric, points, and maximum scores.
Optional practice carries no extension bonus or extra-credit grading track.

## Checks before each lecture release

1. Keep the classroom route, notebook exercise order, expected answers and
   reading sections consistent. Pilot the stated timing and compute budget.
2. Execute the CPU examples, inspect every rendered slide, and check links,
   equations, mobile reading pages and browser interactions.
3. Record commands, revisions, configuration, seeds, resource use and artifact
   locations for reported experiments. Label synthetic fixtures and estimates.
4. Keep quizzes, reference solutions and grading administration in the private
   instructor repository. Use the existing assessment dates and project rules.
