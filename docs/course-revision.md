# Fall 2026 course revision: review draft

Draft dated September 7, 2026. This is a proposed syllabus and assessment design
for instructor review. The [course page](../index.html#schedule) contains the
dated schedule. Revised fall slides, notebooks, and assignment handouts have not
yet been authored; linked spring and CS336 materials are sources for adaptation.

## Design decisions

- Assume Python and an introductory machine-learning course. Teach the PyTorch
  details needed for the common exercises, and provide a preparation notebook.
- Plan 15 teaching meetings. Week 5, October 7, is **skip**, with no lecture,
  quiz, or deadline. Keep calendar week numbers rather than renumbering lessons.
- Retain NLP foundations and the useful spring notebooks. Add explicit data,
  training, resource-accounting, and evaluation work from CS336.
- Use one individual project per student. No presentations, posters, oral
  defenses, assigned project pods, or shared group deliverables are required.
- Allow substantial AI assistance. Grade the question, design, evidence,
  interpretation, and reproducibility. More experimental iteration is encouraged;
  code volume and application feature count are not learning objectives.
- Keep diffusion as one late course session and an advanced project option.
  Remove the requirement to interpret every early topic as a denoising process.
- Write the review draft in English, following the repository guidelines.

## Proposed sequence

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
| 4 | Sep 30 | Neural LMs and attention | A working tiny training loop and checked attention computation |
| 5 | Oct 7 | **skip** | — |
| 6 | Oct 14 | The Transformer as a working model | A decoder block with shape and causal-mask checks |
| 7 | Oct 21 | Pretraining and decoding | A training curve, resumed run, and diagnosed failure |
| 8 | Oct 28 | Data preparation and quality | A comparison of two data policies under controlled conditions |
| 9 | Nov 4 | Compute budgets and scaling | A resource estimate checked against measurements |
| 10 | Nov 11 | Evaluation and experimental design | An audited evaluation and a qualified interpretation |
| 11 | Nov 18 | Supervised fine-tuning | A before/after adaptation comparison |
| 12 | Nov 25 | Preferences, rewards, and alignment | A checked preference loss and reward-failure analysis |
| 13 | Dec 2 | Retrieval and RAG | Separate measurements of retrieval and generation failures |
| 14 | Dec 9 | Efficient inference | A correctness-checked latency/throughput experiment |
| 15 | Dec 16 | Diffusion language models | A sampling comparison with explicit assumptions and budget |
| 16 | Dec 23 | Agents and research synthesis | A bounded tool-use evaluation and final project claim audit |
| 17 | Dec 30 | Final submission; no lecture | Report, code, and experiment record |

BERT is introduced as a bidirectional architecture/objective contrast in Week 6,
revisited through classification in Week 10, and connected to retrieval and
masked generation in Weeks 13 and 15. It no longer occupies a standalone survey
week. Week 16 uses a bounded agent example and project workshop; a broad
multimodal survey is optional.

The previous draft used presentation time in Week 12 and oral-defense time in
Weeks 15–16. Those meetings now retain instruction and practical work. The
holiday does not create a sixteenth teaching session elsewhere in this draft.

## Spring material to retain, shorten, and move

The following are changes to required fall coverage. Keep the spring materials
available as references; do not delete their source files.

| Spring material | Retain or reuse | Remove from core or relocate | Use the time for |
| --- | --- | --- | --- |
| L01 introduction/tokenization | Ambiguity, BPE, vocabulary decisions, bilingual examples | Long product/history tour; regex tutorial moves to preparation | Bytes/Unicode, round-trip checks, and tokenizer tradeoffs |
| L02 n-grams | Chain rule, MLE, sampling, held-out loss, additive smoothing, interpolation | Katz backoff and Good–Turing/Kneser–Ney details become optional; consolidate repeated NPLM introduction | Explicit NLL/cross-entropy connection and baseline interpretation |
| L03 classification/embeddings | Distributional hypothesis, embedding lookup, one skip-gram example, similarity, contextual representations | Repeated NB/LR derivations and full word2vec gradient sequence; GloVe/SVD/fastText survey optional; TF-IDF moves to retrieval | Batching, autograd, output projections, and shape reasoning |
| L04 neural LMs | Existing PyTorch introduction, small feedforward LM, tiny-batch training | Micrograd optional; RNN/LSTM motivation brief; no separate required LSTM training pipeline | Training-loop diagnosis and attention computation |
| L05 attention/Transformers | Attention intuition, Q/K/V, multiple heads, residuals, normalization | Repeated BPE walkthrough, detailed MT results, constituency parsing, long encoder–decoder tour | One consistent decoder implementation and modern component comparisons |
| L06 GPT/pretraining | Existing GPT training notebook, AdamW, scheduling, checkpoint use, HellaSwag example | ELMo history and repeated beam-search variants shortened; model-generation catalog optional | Make training configuration and failure diagnosis explicit outcomes |
| L07 benchmarks | Representative task examples and distinctions between loss and downstream scores | Dataset-by-dataset GLUE/SuperGLUE catalog and many leaderboard slides become reference tables | Contamination, prompt sensitivity, uncertainty, metrics, and error analysis |
| BERT notebook | Masked prediction, contextual embeddings, classification example | Standalone BERT/SpanBERT family survey | Integrate architecture comparison with evaluation and retrieval |
| SFT/RM/PPO notebook | SFT, response-token masking, preference data, reward diagnostics | Full PPO implementation/derivations optional; no required dual-family RL experiment | Separate SFT and preference/reward sessions, with DPO as a small worked example |
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
| L02 PyTorch and resource accounting | Weeks 3, 7, 9 | Tensor mechanics and resource estimates; detailed hardware arithmetic optional |
| L03 architectures; A1 components | Week 6; A2 | One coherent decoder baseline; one modern-component ablation optional |
| L13–14 data; A4 | Week 8; A2 or project | Small supplied corpus, filtering/deduplication, and controlled comparison |
| L09/L11 scaling; A3 | Week 9 | Fit supplied run results and optionally add small local runs; no Stanford API dependency |
| L12 evaluation | Week 10 and project | Choose appropriate tasks, audit validity, and qualify conclusions |
| L15–16 post-training; A5 | Weeks 11–12; A3 | SFT core, DPO worked exercise; GRPO optional or graduate extension |
| L10 inference; A2 systems selections | Week 14 | Measure caching/batching/quantization; custom kernels and distributed training optional |

Source repositories:
[lectures](https://github.com/stanford-cs336/lectures),
[A1 basics](https://github.com/stanford-cs336/assignment1-basics),
[A2 systems](https://github.com/stanford-cs336/assignment2-systems),
[A3 scaling](https://github.com/stanford-cs336/assignment3-scaling),
[A4 data](https://github.com/stanford-cs336/assignment4-data),
[A5 alignment](https://github.com/stanford-cs336/assignment5-alignment).
Before preparing handouts, pin the exact source revision: public repositories
can change independently of a course schedule, and README year labels can lag.

## Individual projects and AI assistance

Students choose a meaningful uncertainty rather than merely a list of features
to implement. Building a system is welcome when it enables the investigation.
The proposal establishes importance, prior work, plausible alternatives,
evaluation, and feasibility. Originality is not a prerequisite for undergraduate
success; careful replication, useful failure analysis, or a sound negative
result can earn full credit.

Written milestones are Week 7 proposal, Week 13 progress update (including a
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

## Proposed assessment and workload

Retain the existing category weights: participation 5%, quizzes 15%, practical
assignments 40%, project 40%. Split the practicals into A1 10%, A2 20%, A3 10%.
They establish common skills and can provide infrastructure for the project;
the final project identifies its additional question and experiments rather
than receiving duplicate credit for the same submission.

| Submission | Released | Due |
| --- | --- | --- |
| A1: Text and probability | Week 2 | Week 4, Sep 30 |
| A2: Build and investigate a small LM | Week 6 | Week 9, Nov 4 |
| A3: Adapt and evaluate | Week 11 | Week 14, Dec 9 |
| Project proposal | Project introduced Week 1 | Week 7, Oct 21 |
| Project progress update, including baseline and initial results | Follows proposal feedback | Week 13, Dec 2 |
| Final report, code, experiment record | Reuses the written checkpoints | Week 17, Dec 30 |

Quiz weeks are 3, 6, 8, 11, and 15; the lowest score is dropped. No practical
deadline coincides with a project deadline or the holiday. Core data-quality
and scaling exercises can use supplied results when extra training would
exceed the pilot-tested budget.

Within the project grade, propose question/motivation 20%, experimental design
25%, evidence/reproducibility 25%, interpretation/iteration 20%, and written
communication/checkpoints 10%. Master's students include one substantial
theoretical or empirical extension, assessed within the same rubric. The old
mandatory 25% graduate add-on to every assignment is removed.

## Review points before releasing to students

1. Confirm whether to retain the existing 40% practical / 40% project split or
   shift more weight toward the individual project. The current draft preserves
   the existing category weights.
2. Pilot the practicals and set model sizes, datasets, GPU-hour budgets, and
   comparable AI-tool access. Scope is based on attainable evidence rather than
   an assumed universal AI productivity multiplier.
3. Confirm the staff capacity for written feedback on the proposal and progress update.
   This is the main support requirement of the individual-project format.
4. The registered course allocation is 48 academic periods. This draft schedules
   15 meetings of three periods, or 45 periods. Any administrative arrangement
   for the remaining three periods is outside this proposed lecture sequence;
   the October 7 row remains **skip**.

After review, prepare the revised decks and handouts using the incorporation
map. BPE and attention are sensible first animation pilots; complete animation
production is not part of this syllabus revision.
