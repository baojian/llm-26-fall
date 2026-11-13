# Course progress and the remaining six lectures

## Current progress: November 11, 2026

The instructor confirms that Lectures 8–10 have covered InstructGPT, SFT,
reward models, REINFORCE/PPO, GRPO and DPO. These are prerequisites for the
remaining course. The GRPO/DPO examples collected in the Lecture 09 supplement
now serve as review; Lecture 11 will not repeat that introduction. Lecture 10's
data preparation, evaluation and experimental-design materials remain the
protocol reference for the experiments below.

The next six meetings, Lectures 11–16, cover efficient adaptation and
distillation, retrieval, inference and test-time compute, agents, masked
diffusion, and research synthesis. Agents move ahead of diffusion so students
can use the retrieval and inference lessons in a bounded system earlier.
The [current schedule](schedule.md) and [notes index](README.md) carry this
sequence. Existing assessment dates and weights remain unchanged.

Lectures 01–10 have slide and notebook packages. The sections below are
**preparation plans for Lectures 11–16**, not a claim that six new decks or demos
are ready. Lecture 10 has 57 core slides, five offline exercises and an optional
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
- Build on the completed post-training sequence. Use brief prerequisite checks
  instead of another SFT, PPO, DPO or GRPO derivation. Lecture 11 introduces
  new parameter-efficiency and teacher–student mechanisms.
- Carry one small, versioned task collection through adaptation, retrieval,
  inference and agents where suitable. Keep its evaluation questions held out
  of teacher generation and tuning. Each class changes one declared factor
  and produces an interpretable comparison; students need not change projects.
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
| 11 | Nov 18 | Efficient adaptation and distillation | A LoRA parameter count, teacher–student loss check and controlled adaptation comparison |
| 12 | Nov 25 | Retrieval and RAG | Separate measurements of retrieval and generation failures |
| 13 | Dec 2 | Efficient inference and computation at test time | A latency/memory comparison and quality-versus-attempt-budget curve |
| 14 | Dec 9 | Agents and tool use | A bounded tool-use evaluation with inspected traces and costs |
| 15 | Dec 16 | Masked diffusion language models | A sampling comparison with explicit assumptions and budget |
| 16 | Dec 23 | Research synthesis and project revision | Project claims linked to evidence, limitations, and a revision plan |
| 17 | Dec 30 | Final submission; no lecture | Report, code, and experiment record |

BERT is introduced as a bidirectional architecture/objective contrast in Week 5,
revisited through classification in Week 10, and connected to retrieval and
masked generation in Weeks 12 and 15. It has no standalone survey week.
Week 14 uses a bounded agent example; Week 16 revisits course methods and
guides revision of project claims, evidence, and writing.

All 16 meetings retain instruction and practical work. Project checkpoints
are written submissions; the final synthesis session adds guided practice.

The instructor moved scaling to Week 7 for Jiaxuan Zou's guest talk,
[How to Build a Scientific Scaling Ladder](https://jiaxuanzou0714.github.io/blog/2026/how-to-build-scientific-scaling-ladder/),
in the third period. Data and evaluation are collected in the Week 10 materials;
the instructor's latest coverage update applies to the post-training sequence
across Weeks 8–10 without assigning every algorithm to a particular meeting.
Assessment dates remain as listed in the [current schedule](schedule.md).

## Lecture 11: Efficient adaptation and distillation

**November 18 · 135 minutes.** How can we transfer useful behavior under a
small training budget? Start from the familiar SFT objective and independent
evaluation boundary. Distinguish which parameters we train from where the
training targets come from.

By the end, students should be able to count LoRA parameters, explain what
QLoRA stores in low precision, distinguish sequence-level and token-level
distillation, and design a comparison that measures both gains and regressions.

| Period | Teaching route | Student evidence |
| --- | --- | --- |
| 1, 45 min | Frozen base weights and low-rank updates; rank and target-layer choices; QLoRA's frozen quantized base versus trainable adapters | E01, 8 min: trace a tiny matrix update, count trainable parameters and identify memory that LoRA does not remove |
| 2, 45 min | Teacher-generated responses versus soft next-token targets; temperature and KL direction; student-generated prefixes as an on-policy distillation bridge | E02, 8 min: compare hard-target cross-entropy with a teacher–student loss and identify whose prefixes were sampled |
| 3, 45 min | Compare an unchanged checkpoint, ordinary adaptation and teacher-data adaptation; hold the student, token budget and evaluation protocol fixed | E03, 15 min: audit supplied run records, separate a teacher-data change from a rank change, and report one gain and one regression |

Read [LoRA, Section 4](https://arxiv.org/abs/2106.09685) and
[QLoRA, Section 3](https://arxiv.org/abs/2305.14314).
[Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649) supplies
the bridge from fixed teacher responses to teacher feedback on student-generated
sequences; its full training algorithm is optional reading.

Prepare a CPU matrix example, same-vocabulary probability fixtures and a small
set of saved teacher responses with provenance. Token-level KL needs aligned
token distributions; sequence-level text targets do not require identical
tokenizers. Include a teacher error and a held-out failure slice. Teacher
agreement alone is not evidence of task improvement. The classroom comparison
may use clearly labeled supplied results; any live run is an optional small
LoRA experiment, not a requirement to train a large teacher or repeat RLHF.
This is ungraded practice supporting A3 and the project, with no new A3 rule.

## Lecture 12: Retrieval and RAG

**November 25 · 135 minutes.** When should a model retrieve evidence, and
which stage caused a wrong answer? Use a small course-notes collection with
document IDs carried from retrieval through citation scoring.

Students should distinguish a knowledge-access problem from a parameter-update
problem, locate failures in the pipeline, and justify a retrieval change with
held-out evidence. Reuse Lecture 11's task protocol while changing access to
documents rather than the model weights.

| Period | Teaching route | Student evidence |
| --- | --- | --- |
| 1, 45 min | BM25 and dense retrieval; query/document representations; Recall@k | E01, 8 min: rank a small collection and identify missed supporting documents |
| 2, 45 min | Chunk boundaries, reranking and context assembly under a fixed budget | E02, 10 min: compare two chunking or retrieval policies on the same questions |
| 3, 45 min | No retrieval, retrieved context and supplied supporting context; citation support and abstention | E03, 12 min: separate retrieval failure, unsupported generation and unanswerable cases in saved responses |

Read [DPR, Sections 2–4](https://arxiv.org/abs/2004.04906) and
[RAG, Sections 2–3](https://arxiv.org/abs/2005.11401). Prepare checked relevance
labels and answerable/unanswerable questions, then preserve a held-out split.
The CPU route uses BM25, supplied embeddings and saved responses; a live
generator is optional. Supplying known supporting context is a diagnostic
control, not a deployable retrieval result. Keep the generator, prompt and
context-token budget fixed when comparing retrieval policies. Save an error
table with question ID, retrieved IDs, support label and failure category.
The existing project progress update is due that evening; the classroom
comparison adds no separate graded submission.

## Lecture 13: Inference and computation at test time

**December 2 · 135 minutes.** How should a fixed inference budget be spent?
Connect Lecture 07's compute allocation to the evaluation and verifier audits
in Lectures 8–10. Use the retrieval workload from Lecture 12 as one concrete
case. Separate systems efficiency from spending more attempts.

Students should estimate KV-cache memory, choose a metric that matches a
workload, and compare deployed answer-selection quality at a declared cost.

| Period | Teaching route | Student evidence |
| --- | --- | --- |
| 1, 45 min | Prefill, decoding and KV-cache size; latency versus throughput; attention I/O | E01, 8 min: estimate cache bytes from layers, KV heads, dimensions, sequence length, batch and dtype |
| 2, 45 min | Batching, cache reuse and quantization; warm-up and repeated measurements | E02, 10 min: audit a timing/memory comparison for correctness, hardware and input/output token counts |
| 3, 45 min | Best-of-N, majority vote and verification; actual selection versus oracle pass@k | E03, 12 min: plot selected-answer success against attempts, total tokens and wall time, including selection cost |

Read [FlashAttention, Sections 2–3](https://arxiv.org/abs/2205.14135),
[PagedAttention/vLLM, Sections 3–4](https://arxiv.org/abs/2309.06180), and
[Snell et al., test-time compute](https://arxiv.org/abs/2408.03314).
Prepare a small CPU cache example and auditable candidate traces. An optional
GPU profile must report its own hardware; it cannot turn a CPU trace into a
measured GPU speedup. Report time to first token, inter-token latency and
throughput separately. An answer selected with access to the correct answer
is an oracle diagnostic, not the measured performance of a deployed verifier.
A3 keeps its published deadline that evening; this class adds no requirement
to rerun or extend the assignment.

## Lecture 14: Agents and tool use

**December 9 · 120 teaching minutes + 15 minutes for Quiz 5.** When is a
tool-use loop worth its additional cost? Extend Lecture 12's retrieval tool
and Lecture 13's budget accounting into a bounded observe–act–tool loop.

Students should specify the agent state and stopping rule, recover from a
tool error, and explain whether a measured gain comes from tools or simply
from extra computation. PPO/GRPO are prior knowledge for an optional discussion
of training on trajectories; the core implements and evaluates a loop.

| Period | Teaching route | Student evidence |
| --- | --- | --- |
| 1, 45 min | State, tool schema, observations and stopping conditions | E01, 8 min: trace one successful episode and repair one failed tool call |
| 2, 45 min | Tool permissions, untrusted tool text, recovery and action limits | E02, 10 min: inspect a prompt-injection fixture and enforce the permitted actions and stopping rule |
| 3, 30 min + quiz | Direct generation, a fixed retrieval pipeline and a bounded agent on the same tasks; then Quiz 5 | E03, 10 min: compare success, failures, tool calls, tokens, latency and budget exhaustion |

Read [ReAct, Sections 2–3](https://arxiv.org/abs/2210.03629).
[SWE-bench](https://arxiv.org/abs/2310.06770) is an optional evaluation-design
case, not a required full benchmark run. Prepare an offline tool environment
and replayable traces; network access and paid APIs are unnecessary for the
core lesson. Include a timeout, a malformed tool result and an episode that
exhausts its action budget. Use saved model decisions to make the CPU replay
reproducible; distinguish replay validation from live agent performance.
Training large agents or coordinating many agents is optional reading.
Quiz 5 covers previously taught core material, not the new agent lesson or
optional extensions; its questions remain in the instructor repository.

## Lecture 15: Masked diffusion language models

**December 16 · 135 minutes.** What changes when a generator can update
multiple positions? Retrieve BERT's masked prediction objective and the causal
decoder as distinct starting points, then apply Lecture 13's cost accounting.

Students should connect a masking process to a denoising objective, execute an
iterative sampler, and state what a fair comparison with autoregression can
and cannot establish. This is one focused alternative-generation lesson.

| Period | Teaching route | Student evidence |
| --- | --- | --- |
| 1, 45 min | A discrete masking process and time-dependent denoising loss | E01, 8 min: trace a corruption/target pair and explain why BERT alone is not a complete diffusion generator |
| 2, 45 min | Iterative unmasking, conditioning and sampling schedules | E02, 10 min: trace token states and compare two unmasking schedules under the same toy denoiser |
| 3, 45 min | Compare quality, controllability and cost against autoregression; identify confounded model comparisons | E03, 12 min: audit step count, forward passes, token budget, hardware and measured runtime in a supplied comparison |

Read [MDLM, Sections 3–4](https://arxiv.org/abs/2406.07524) and
[LLaDA, Sections 2–3](https://arxiv.org/abs/2502.09992). Prepare a CPU toy
sampler with explicit assumptions and deterministic fixtures. Its traces
illustrate mechanics, not language quality. Updating positions in parallel
does not establish an end-to-end speedup; account for repeated forward passes.
Real-model results need matched workloads and disclosure of model, data and
training differences. No diffusion training or late project-topic change is
required; students may use the supplied comparison for classroom practice.

## Lecture 16: Research synthesis and project revision

**December 23 · 135 minutes.** Which project claims survive a careful check?
Bring one current result, its baseline and its experiment record; students
work individually and use the same criteria regardless of degree program.

Synthesize the course through a method-choice question: does the observed
failure call for better data, adaptation, retrieval, more inference compute,
tools, or a narrower claim? Students should justify that choice, identify a
missing control, and write a reproducible statement of their strongest result.

| Period | Teaching route | Student evidence |
| --- | --- | --- |
| 1, 45 min | Map failure → method → comparison → evidence → claim using the course's adaptation, retrieval, inference and agent examples | E01, 10 min: choose a justified intervention or narrow an unsupported inference |
| 2, 45 min | Inspect leakage, uncertainty, resource accounting and negative results | E02, 12 min: repair a missing control or narrow a claim to the tested population |
| 3, 45 min | Revise figures, methods and limitations; check reproducibility | E03, 15 min: prepare a revision checklist linking a report claim to code, configuration and experiment record |

Use the [project guidance](project-candidates.html),
[Lecture 10 experiment-report template](lecture-10-data-evaluation.html#11-a-minimum-useful-experiment-report)
and the project's own primary papers. Prepare an authored example report
and an evidence checklist, keeping student work private. Provide the authored
example as a fallback if a student's result is not ready. Prioritize analysis
and revision of existing evidence over launching large new experiments.
The report allows
six pages of main content; references and appendices are excluded from the
limit. Final submission remains December 30, 23:59 (Asia/Shanghai).
No presentation, poster or new oral assessment is introduced.

## Preparation status and compute budgets

The six routes above are plans. Each needs a finished deck, checked examples
and a classroom rehearsal. Exercise IDs and times are proposed authoring
targets, not claims that new notebooks are available; prepare expected answers
and validate them before release. All classroom activities are ungraded.
Preserve a CPU route for every lesson. The following
optional live-demo budgets are **unmeasured planning estimates**, not course-pool
allocations or claims about an already-tested model.

| Lecture | Minimum route to prepare | Optional live rehearsal: memory per GPU; total GPU-hours |
| --- | --- | --- |
| 11 | CPU matrix/loss fixtures, saved teacher responses and run records; 0 GPU-hours | One 16–24 GB GPU; up to 1 GPU-hour total for a short LoRA run on a model of at most 1B parameters, using saved teacher data |
| 12 | CPU retrieval and saved responses; 0 GPU-hours | One 16–24 GB GPU; up to 0.25 GPU-hours for generation across all runs |
| 13 | CPU cache example and saved candidate traces; 0 GPU-hours | One 16–24 GB GPU; up to 0.25 GPU-hours for profiling across all runs |
| 14 | CPU tools and replayed episodes; 0 GPU-hours | One 16–24 GB GPU; up to 0.25 GPU-hours for bounded generation across all runs |
| 15 | CPU toy masked sampler; 0 GPU-hours | One 16–24 GB GPU; up to 0.5 GPU-hours only if a suitable checkpoint fits; otherwise use the CPU route |
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
| SFT/RM/PPO notebook | SFT, response-token masking, preference data, reward diagnostics; PPO/GAE as taught | Full PPO training implementation optional; no required dual-family RL experiment | Post-training foundations completed across Weeks 8–10; efficient adaptation/distillation in Week 11 and agent evaluation in Week 14 |
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
| L15–16 post-training; A5 | Weeks 8–10; A3 context | SFT/RM/PPO, DPO and GRPO already covered; Week 11 builds on them with adaptation efficiency and distillation; full RL training optional |
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
