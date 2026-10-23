<!-- .slide: class="title-slide" id="title" -->

<p class="eyebrow">CS40008.01</p>

# Data quality and model evaluation

<p class="subtitle">Lecture 08 · NLP and LLMs</p>

<p class="byline">Baojian Zhou<br>School of Data Science<br>Fudan University<br>October 28, 2026</p>

Note:
Three 45-minute periods, with the usual breaks. The central question is whether a change to training data produces an improvement we can trust. This lecture connects data preparation with evaluation. The benchmark examples update the instructor's Spring Lecture 07, which credits Anoop Sarkar. The five exercises are ungraded and run offline on CPU. The modern benchmark protocols were checked on October 23, 2026. Sources and the adaptation map are in teaching-plan.md and optional-reading.md.

---

<!-- .slide: id="objectives" -->

## What you will be able to do

**Did better training data actually improve the model?**

- Audit duplicates and overlap between data splits.
- Compare data policies under a fixed training budget.
- Explain what a benchmark score measures.
- Check whether an apparent improvement survives an evaluation audit.

Note:
Connect Lecture 07's fixed compute comparison to this lecture's fixed model and tokenizer. The goal is to justify a claim with inspectable evidence. Students should leave with an evaluation plan they can use for A2 and their individual project. A2's published requirements and deadline remain the authority for the assignment. The classroom calculations do not require another training run.

---

<!-- .slide: class="outline-slide" id="outline-data" -->

## Lecture outline

<ol class="outline-topics">
  <li aria-current="step">Training data and clean splits</li>
  <li>Benchmarks and scoring protocols</li>
  <li>Controlled comparisons and uncertainty</li>
</ol>

<p class="caption">First period · 13:30–14:15</p>

Note:
The same data-policy comparison returns in period 3. First establish what data can enter each stage. Keep the model, tokenizer, and token budget fixed before comparing quality policies.

---

<!-- .slide: id="running-comparison" -->

## One comparison throughout the lecture

| Condition | Training data | Everything else |
| --- | --- | --- |
| A | Original corpus | Fixed model and tokenizer |
| B | Deduplicated corpus | Same training-token budget |

We need a common evaluation set and a declared scoring rule.

<p class="caption">Classroom outcomes are synthetic. They illustrate an audit, not a measured training result.</p>

Note:
Use the same initial weights, optimizer settings, and training schedule for a single paired run. Multiple training seeds are a separate source of uncertainty. A2 already supplies an exact-document deduplication comparison under matched processed-token budgets. This lesson uses small authored fixtures so students can inspect every assumption. Source: ../../assignments/a2-build-a-small-lm/README.md, Part 4.

---

<!-- .slide: id="data-lineage" -->

## A traceable data pipeline

| Stage | Record |
| --- | --- |
| Acquire and extract | Source, date, license, document ID |
| Normalize and group | Rules, duplicate family, language |
| Assign splits | Group boundaries and held-out IDs |
| Build training mixture | Filters, sampling weights, token counts |

The same source family should stay within one split.

Note:
Distinguish creating a new dataset from using an externally fixed benchmark. For a new collection, group related documents before assigning splits. For an established benchmark, preserve its split and audit training data against the held-out identifiers or texts under a fixed policy. Never use test answers to tune filters. Record the versions that produced each artifact. This is a course workflow informed by FineWeb's documented extraction, filtering, and deduplication ablations: https://arxiv.org/abs/2406.17557, dataset construction sections.

---

<!-- .slide: id="text-extraction" -->

## Text extraction changes the learning problem

| Input | Possible extraction error |
| --- | --- |
| Web article | Navigation repeated on every page |
| Code tutorial | Indentation or symbols lost |
| Table | Values detached from their headers |
| Multilingual text | Useful passages rejected by an English rule |

Inspect retained examples and rejected examples.

Note:
These are authored failure examples. Parsing HTML is only one step: preserving the meaningful sequence matters for a language model. Do not show an HTML-tag regex as a general web extractor. Normalization, content extraction, and quality selection make different changes. Keep original source identifiers so a suspicious example can be traced back. FineWeb provides a concrete large-scale case study, not a recipe students must reproduce at its full scale.

---

<!-- .slide: id="quality-filters" -->

## Quality depends on the intended use

- Repetition filters can catch boilerplate.
- Language filters change which communities remain represented.
- Model-based scores inherit the reference model's preferences.
- Aggressive thresholds can remove difficult but useful examples.

Compare the resulting models on relevant held-out tasks.

Note:
Ask students for an example of useful repetition, such as definitions in textbooks or common code syntax. A filtering score is a selection rule, and its effectiveness is an empirical question for a target task and budget. A held-out development set can guide threshold selection. Preserve a final set for the selected policy. Relevant ablations: FineWeb, https://arxiv.org/abs/2406.17557. The language-coverage cautions here are experimental design considerations rather than universal measured effects of every filter.

---

<!-- .slide: id="exact-duplicates" -->

## Exact duplicates depend on normalization

```python
text = unicodedata.normalize("NFC", text)
text = text.replace("\r\n", "\n").replace("\r", "\n")
key = text.strip()
```

Preserve case and internal whitespace in this exercise.

`"Noa reads the red book."` and its padded copy share a key.

Note:
This is the A2 normalization convention. A stable hash can index keys in a larger system, with collision handling appropriate to the application. The classroom audit groups text values directly. It records IDs without deleting any files. Explain that changing normalization can merge genuinely distinct code, quotations, or formatting-sensitive documents. Source: ../../assignments/a2-build-a-small-lm/README.md, Part 4.

---

<!-- .slide: id="near-duplicates" -->

## Near duplicates need an explicit similarity rule

$$J(A,B)=\frac{|A\cap B|}{|A\cup B|}$$

Compare word-bigram sets:

- Noa reads the **red** book.
- Noa reads the **blue** book.

Two shared bigrams, six in the union: **$J=1/3$**.

<p class="caption">A similarity threshold flags candidates for review.</p>

Note:
The two shared bigrams are “Noa reads” and “reads the.” These are whitespace word shingles for this English toy example, not model BPE tokens. Shingle size and normalization affect the score. A color substitution may change the answer to a question, so similarity alone does not establish semantic equivalence. Large collections can use approximate matching methods, whose algorithmic details are optional reading. Source: Lee et al., https://aclanthology.org/2022.acl-long.577/, deduplication methods.

---

<!-- .slide: class="exercise" id="exercise-01" -->

<p class="exercise-meta">Exercise E01 · 6 minutes · Notebook E01</p>

## Which records belong together?

Inspect the six supplied documents.

1. Find exact duplicate groups after normalization.
2. Identify the group that crosses train and development splits.
3. Explain why the red/blue pair needs separate consideration.

<div class="answer fragment"><p>Duplicates: train-1/train-2 and train-4/dev-1. The café pair crosses splits.</p></div>

Note:
Expected response: Unicode normalization equates the decomposed accent in train-4 with dev-1's composed accent. The padded Noa copy is a training duplicate. Red and blue produce distinct normalized texts, although they have shared word bigrams. For a new dataset, keep duplicate families within one split. For fixed external held-out data, exclude overlapping training records under the declared policy and audit remaining overlap. Preserve the source files. Fixtures: assets/toy-documents.json.

---

<!-- .slide: id="split-construction" -->

## A random row split can leak source content

| Related records | Useful grouping boundary |
| --- | --- |
| Copies of one article | Duplicate family |
| Chunks of one document | Original document |
| Revisions of one repository task | Task or repository lineage |
| Repeated records from one person | Person, when appropriate |

Choose the boundary that matches the generalization question.

Note:
Rows are not always independent examples. A chunk split asks a different question from a document split. A repository split can test transfer to new projects, while an issue split within a repository tests another setting. Grouping choices can also make evaluation harder or change its target population. Explain the intended deployment before choosing the boundary. These are course-authored design examples, not a claim that one grouping rule suits every dataset.

---

<!-- .slide: id="contamination-paths" -->

## Evaluation information can enter several stages

- Pretraining includes benchmark questions or solutions.
- Fine-tuning examples contain evaluation answers.
- Demonstrations reproduce a held-out item.
- Repeated test-set feedback guides model or prompt selection.
- A tool retrieves the solution during evaluation.

The audit must cover the full evaluated system.

Note:
Direct text overlap is only one route. Paraphrases and derived examples may escape exact matching. Repeated use of a public test set can also bias model selection without literal memorization. A permitted web search is part of a tool-enabled protocol, but looking up a benchmark solution may violate the intended task. State the tool policy before running. Lee et al. supports train/evaluation overlap analysis; the tool and grading case returns in the SWE-Bench Pro V2 slide.

---

<!-- .slide: id="fixed-budget" -->

## Match processed tokens when data size changes

| Condition | Corpus tokens | Processed tokens | Approximate passes |
| --- | ---: | ---: | ---: |
| A | 4 million | 12 million | 3 |
| B | 3 million | 12 million | 4 |

Keep the architecture, tokenizer, context length, and training recipe fixed.

<p class="caption">Illustrative counts. Equal epochs would change the training budget.</p>

Note:
Processed tokens count repeated exposures. The retained corpus size measures a different quantity. With fixed architecture, sequence shape, and implementation, matching processed tokens is a useful approximation to matching training FLOPs. Report measured runtime separately because throughput and data handling can differ. The pass count is approximate when packing or incomplete windows affect consumption. This is the same budget convention required by A2.

---

<!-- .slide: id="mixtures" -->

## An aggregate score reflects the language mixture

| Synthetic accuracy | English | Chinese | 80/20 mixture | Equal language weight |
| --- | ---: | ---: | ---: | ---: |
| A | 90% | 40% | 80% | 65% |
| B | 85% | 60% | 80% | 72.5% |

Report relevant slices and state their weights.

Note:
Compute A's 80/20 score as 0.8 times 90 plus 0.2 times 40. These are constructed values, not measurements of a real model or language. Equal language weighting is a macro average over these two language accuracies; an item-weighted average depends on the item counts. Neither weighting is universally right. Choose weights that answer the application question before comparing systems. Training mixtures and evaluation mixtures are separate choices.

---

<!-- .slide: id="data-evidence" -->

## Data policies need controlled evidence

**Lee et al., 2022:** investigate deduplication, memorization, and evaluation overlap.

**FineWeb, 2024:** document extraction, filtering, and deduplication ablations.

For our course: change one data policy and keep a comparable training budget.

Note:
These papers provide methods and empirical case studies. Their findings do not guarantee that every deduplication or quality threshold improves every model. Students can get a sound negative result. Read Lee et al., methods and experimental sections, https://aclanthology.org/2022.acl-long.577/; FineWeb, dataset-construction ablations, https://arxiv.org/abs/2406.17557. Full paper training budgets are outside the classroom scope. Allow a short transition before the break.

---

<!-- .slide: class="outline-slide" id="outline-benchmarks" -->

## Lecture outline

<ol class="outline-topics">
  <li>Training data and clean splits</li>
  <li aria-current="step">Benchmarks and scoring protocols</li>
  <li>Controlled comparisons and uncertainty</li>
</ol>

<p class="caption">Second period · 14:25–15:10</p>

Note:
Use a few familiar tasks to explain the scoring interfaces, then introduce current benchmark versions. Avoid treating the list as vocabulary students must memorize.

---

<!-- .slide: id="benchmark-definition" -->

## A benchmark includes a protocol

| Component | Example |
| --- | --- |
| Task | Select a plausible continuation |
| Dataset | A versioned set of contexts and choices |
| Protocol | Prompt, demonstrations, tools, budget |
| Scoring | Answer extraction and aggregation |

A model name and one number leave much of the experiment unspecified.

Note:
An evaluation harness implements this contract. Record the dataset revision, task configuration, model revision, tokenizer, prompt, generation settings, and scoring code. The EleutherAI harness distinguishes likelihood-based requests from generated responses; a backend's capabilities constrain which task interface can run. Source: https://github.com/EleutherAI/lm-evaluation-harness, model request types and reproducibility guidance. We inspect the concepts without requiring a new harness installation today.

---

<!-- .slide: id="historical-tasks" -->

## Established tasks still teach useful distinctions

| Spring example | What the model returns |
| --- | --- |
| SuperGLUE: BoolQ, RTE | A label about a passage or sentence pair |
| HellaSwag | A continuation choice |
| MMLU | A subject-question answer |
| GSM8K | A final numerical answer |
| HumanEval | Executable code |

<p class="caption">Retained from the Spring material. Each requires its own scoring rule.</p>

Note:
This compresses the Spring deck's dataset catalog. Different SuperGLUE tasks also use different metrics, so do not call every suite score plain accuracy. Older public tasks remain useful for small models, debugging, and studying specific failure modes. Strong scores alone do not establish broad capability. Source: supplied Spring PPTX slides 3–29, which credits Anoop Sarkar; SuperGLUE paper https://arxiv.org/abs/1905.00537; HellaSwag project https://rowanzellers.com/hellaswag/.

---

<!-- .slide: id="historical-shortcuts" -->

## Shortcut checks can expose dataset artifacts

<div class="plot" data-plotly="assets/historical-shortcuts.json" role="img" aria-label="Historical 2019 BERT-Large accuracy on SWAG and HellaSwag under default, ending-only, shuffled, and shuffled-ending-only conditions."></div>

<p class="caption">Historical BERT-Large results, 2019. Ending-only removes the context.</p>

Note:
The original Spring slide 17 includes Figure 4 of Zellers et al., HellaSwag. This editable replot preserves all eight values and uses a zero baseline. Models were trained and evaluated under the stated conditions. High ending-only performance suggests exploitable information in the candidate endings; it does not measure understanding of the missing context. Shuffled means token order within endings is shuffled. These are historical experimental results, not current model rankings. Source: https://arxiv.org/abs/1905.07830, Figure 4; supplied PPTX image16.png.

---

<!-- .slide: id="model-interfaces" -->

## The same task can use different model interfaces

| Interface | Sentiment example |
| --- | --- |
| BERT masked token | “It was [MASK].” Compare label words. |
| Causal LM likelihood | Score each candidate answer. |
| Chat generation | Generate a label, then parse it. |

Prompts, label words, and parsing rules are experimental choices.

Note:
Retain the useful comparison in the Spring companion notebook. BERT predicts a masked token using both sides of its input. A causal LM scores an answer token sequence conditioned on a prefix. A chat model generates a response that needs a declared extraction rule. A plain masked-token score is not an autoregressive sequence probability. Check label-token boundaries and label mappings. Do not copy the old notebook's automatic model downloads into the classroom core. Source: BERT, https://aclanthology.org/N19-1423/, Section 3; supplied Spring notebook, zero-shot evaluation section.

---

<!-- .slide: id="completion-scoring" -->

## Candidate likelihood and length normalization

$$s(y\mid x)=\sum_{t=1}^{|y|}\log p(y_t\mid x,y_{<t})$$

An alternative rule uses $s(y\mid x)/|y|$.

- Score the answer tokens under a fixed prompt.
- State how the denominator counts length.
- Choose the rule before inspecting test results.

Note:
The sum is the candidate's conditional log probability under the token sequence model. Dividing by answer length creates a different decision rule, and neither rule is universally appropriate for every benchmark. Our exercise uses token counts. Real harness tasks may normalize by characters, bytes, or another convention, so inspect the pinned task definition. Leading spaces and prompt/completion tokenization boundaries can also matter. Source: EleutherAI harness task configuration documentation, linked from https://github.com/EleutherAI/lm-evaluation-harness.

---

<!-- .slide: class="exercise" id="exercise-02" -->

<p class="exercise-meta">Exercise E02 · 6 minutes · Notebook E02</p>

## Which candidate wins under each rule?

| Candidate | Token log probabilities | Sum | Per token |
| --- | --- | ---: | ---: |
| A | −1.1, −1.3 | ? | ? |
| B | −0.7, −0.8, −0.8, −0.9 | ? | ? |

<div class="answer fragment"><p>Sum: A wins, −2.4 versus −3.2.<br>Per token: B wins, −0.8 versus −1.2.</p></div>

Note:
These are invented valid log probabilities. Larger, less negative scores win. Expected response: the choice flips because the protocol changed. No task label has been supplied, so students cannot infer which rule is more accurate from this calculation. The result does not imply that a normalized score is a joint probability. Compare only runs with the same declared scoring convention.

---

<!-- .slide: id="knowledge-benchmarks" -->

## Knowledge and reasoning benchmarks

| Benchmark | Useful distinction |
| --- | --- |
| MMLU-Pro | More reasoning-focused questions, up to ten choices |
| GPQA | Expert-written science questions |
| HLE-Diamond | A revised, frozen expert-question collection |

Inspect the subject coverage, answer format, and selected subset.

<p class="caption">Benchmark sources checked October 23, 2026.</p>

Note:
MMLU-Pro covers 14 subject domains and was introduced in 2024. GPQA's original main set contains 448 questions in biology, physics, and chemistry; reports using the Diamond subset must name that subset. Avoid treating either as a complete measure of reasoning in every domain. Sources: https://arxiv.org/abs/2406.01574 and https://github.com/TIGER-AI-Lab/MMLU-Pro; https://arxiv.org/abs/2311.12022. HLE-Diamond is the recent revision discussed next. Do not reuse the Spring MMLU leaderboard as a current ranking.

---

<!-- .slide: id="hle-diamond" -->

## HLE-Diamond: versioning the evaluation set

**Announced September 22, 2026.**

- 1,000 questions with stable identifiers.
- 500 reasoning questions and 500 knowledge questions.
- Revised from the original HLE pool and its reserve.
- Report the partition scores and the tool setting.

<p class="caption">HLE, HLE-Rolling, and HLE-Diamond are different evaluation versions.</p>

Note:
The publisher describes Diamond as a frozen, refined collection. Its protocol uses automatic answer extraction/judging and also reports calibration. A new question set changes the comparison population: do not place an original-HLE score beside a Diamond score as if they were the same experiment. Strong results support claims about these expert questions, not a general claim about autonomous research. Sources: https://www.lastexam.ai/, latest-news entry; https://labs.scale.com/leaderboard/hle-diamond, design and methodology. Counts and announcement date checked October 23, 2026.

---

<!-- .slide: id="code-and-interaction" -->

## Code and interaction need richer protocols

| Benchmark | Evaluated behavior |
| --- | --- |
| LiveCodeBench | Code tasks with dated problem windows |
| SWE-Bench Pro V2 | Repository changes checked by tests |
| ARC-AGI-3 | Learning through actions in unfamiliar environments |
| LiveBench 2.0 | Longer tasks with graded deliverables |

Record the available tools, time limits, and grading environment.

Note:
LiveCodeBench's official repository explains release versions and date-window selection. SWE-Bench Pro V2 and ARC-AGI-3 are developed on the next slides. LiveBench's publisher lists version 2.0 dated October 14, 2026 and tasks involving artifacts such as reports, software, and decks. Comparing its aggregate with version 1 requires checking the changed suite and grader. Sources: https://github.com/LiveCodeBench/LiveCodeBench; https://labs.scale.com/blog/swe-bench-pro-v2; https://arcprize.org/arc-agi/3; https://livebench.ai/. These are protocol examples, not required full-suite student runs.

---

<!-- .slide: id="swe-benchmark-repair" -->

## SWE-Bench Pro V2: task quality and grading

| Publisher's September 2026 revision | Evaluation issue |
| --- | --- |
| 731 public tasks reduced to 642 | Invalid task instances |
| Instructions checked by independent engineers | Ambiguous task specifications |
| Restricted network during agent work | Access to solution information |
| Re-grading on a pristine image | Changes to the grading environment |

Note:
This is a concrete current case where data curation and evaluation belong in one lesson. Scale's release describes checking reference-patch success and empty-patch failure, alongside manual task review. A second engineer solves a corrected specification without seeing the reference solution. Network restrictions cannot undo prior training exposure. Distinguish this named Pro V2 release from SWE-bench Verified and from other independently revised Pro datasets. Sources: https://labs.scale.com/blog/swe-bench-pro-v2, Sections 1, 3, and 5; https://labs.scale.com/leaderboard/swe_bench_pro_public_v2. Counts describe the publisher's release, not an independent audit by this course.

---

<!-- .slide: id="interactive-evaluation" -->

## ARC-AGI-3: actions become part of evaluation

- The agent explores an unfamiliar interactive environment.
- It infers goals and tests actions using feedback.
- Scoring considers efficiency relative to human action baselines.
- Replays help explain failures behind the final score.

Memory, tools, and the agent's control loop affect the measured system.

Note:
ARC-AGI-3 launched in March 2026. This is an introduction to evaluating an interactive system; agent implementation comes later in the course. Completion and action efficiency are different quantities, so a score is not simply the percentage of games solved. Public demo and semi-private results also represent different evaluation sets. Sources: https://arcprize.org/arc-agi/3; https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf, benchmark and scoring sections. Avoid carrying the launch-time model scores into a current leaderboard claim.

---

<!-- .slide: id="pass-at-k" -->

## More attempts change the success measure

With $c$ correct completions among $n$ samples:

$$\widehat{\mathrm{pass@}k}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}$$

This estimates success among $k$ attempts under the sampling protocol.

Report the sampling settings and total attempt budget.

Note:
This is the HumanEval estimator, with n at least k. The probability of selecting no successful completion from n fixed samples is the ratio of combinations. Under independent samples from a fixed generator, it estimates the probability that at least one of k attempts passes the tests. It does not give a deployment method for choosing the successful answer without a checker. Adaptive repair with feedback is a different protocol. Source: Chen et al., https://arxiv.org/abs/2107.03374, Section 3 and Appendix A.

---

<!-- .slide: class="exercise" id="exercise-03" -->

<p class="exercise-meta">Exercise E03 · 5 minutes · Notebook E03</p>

## How much does the attempt budget matter?

Ten sampled programs contain two that pass all tests.

1. Calculate estimated pass@1 and pass@5.
2. Does the difference show that the model weights improved?

<div class="answer fragment"><p>pass@1 = 20%. pass@5 = $7/9\approx77.8\%$. The sampling budget changed.</p></div>

Note:
Expected arithmetic: 1 minus choose(8,5) divided by choose(10,5) equals 1 minus 56/252. The same ten outcomes support both estimates. This example contains no generated executable code. Students should distinguish a task-level pass fraction from repeated attempts on a task, and should not compare pass@1 to pass@5 as if the compute were equal.

---

<!-- .slide: id="graders" -->

## A grader needs its own validation

| Grader | Example failure |
| --- | --- |
| Exact string match | Equivalent answer rejected |
| Unit tests | Untested behavior accepted |
| LLM judge | Rubric or presentation changes the judgment |
| Human rubric | Reviewers disagree on an ambiguous criterion |

Inspect disagreements and retain examples behind the score.

Note:
These are possibilities to audit, not measured error rates for a named grader. Use known correct and incorrect examples, check label mappings, and manually inspect a small sample. For an LLM judge, record its model revision, prompt, rubric, and reference-answer access. A grader's agreement with humans on a particular sample does not establish universal validity. HLE-Diamond's answer-extraction protocol and the SWE-Bench Pro V2 clean re-grading example provide concrete contexts. Detailed judge studies are linked in optional-reading.md.

---

<!-- .slide: class="outline-slide" id="outline-evidence" -->

## Lecture outline

<ol class="outline-topics">
  <li>Training data and clean splits</li>
  <li>Benchmarks and scoring protocols</li>
  <li aria-current="step">Controlled comparisons and uncertainty</li>
</ol>

<p class="caption">Third period · 15:25–16:10</p>

Note:
Return to the original versus deduplicated data comparison. We now have enough vocabulary to audit the result. Keep the exercise outcomes clearly labeled synthetic throughout.

---

<!-- .slide: id="comparison-record" -->

## A comparison needs a frozen record

| Training record | Evaluation record |
| --- | --- |
| Data policy and source revision | Dataset revision and item IDs |
| Model, tokenizer, initialization | Prompt and answer parser |
| Processed tokens and schedule | Sampling and tool budget |
| Training seeds | Metric, slices, and aggregation |

Use development evidence for choices, then run the declared final evaluation.

Note:
Fixing a record is a procedural decision, not a claim that files cannot change. Record hashes where useful and retain predictions by item ID. If a bug invalidates the final evaluation, document it and design a fresh valid comparison rather than silently treating a repeatedly optimized score as untouched evidence. This extends Lecture 07's separation of fitting and validation to data policies and task evaluation.

---

<!-- .slide: class="exercise" id="exercise-04" -->

<p class="exercise-meta">Exercise E04 · 6 minutes · Notebook E04</p>

## Does the evaluation set change the conclusion?

| Synthetic outcomes | Raw data A | Deduplicated data B |
| --- | ---: | ---: |
| 100 clean items | 60 correct | 64 correct |
| 20 exposed items | 19 correct | 11 correct |

Compute accuracy on all 120 items and on the clean 100.

<div class="answer fragment"><p>All: A 65.8%, B 62.5%. Clean: A 60%, B 64%.</p></div>

Note:
An independent audit has already marked the exposed items in this constructed example. The numbers show how sample composition can reverse the ranking. They do not establish how much of a real score is caused by memorization. Students must not exclude examples because a preferred model gets them wrong. Exclusion needs an independently justified rule, and changing the evaluated population must be reported. Full item-level correctness vectors are constructed in assets/evaluation.py.

---

<!-- .slide: id="evaluation-demo" -->

## The ranking changes with the evaluated items

<div class="demo-form">
  <button id="evaluation-toggle" type="button" aria-pressed="false">Mixed evaluation</button>
  <button id="reset-button" type="button">Reset</button>
</div>

<div id="evaluation-plot" class="plot" data-plotly="assets/contamination.json" role="img" aria-label="Synthetic accuracies for raw and deduplicated training data. Toggle between a mixed evaluation with 120 items and a clean evaluation with 100 items."></div>

<p class="caption" id="evaluation-status" aria-live="polite">Synthetic: 100 clean items plus 20 exposed items. Raw 65.8%, deduplicated 62.5%.</p>

Note:
Toggle once after students calculate both results, then reset. The chart uses a zero baseline and a fixed 0–100 percent scale. The evaluation size changes, while the predictions on the clean items stay fixed. This demonstration does not run inference or remove any files. In the PDF, the initial mixed set appears; the preceding exercise gives both outcomes.

---

<!-- .slide: id="loss-and-task" -->

## Language-model loss and task accuracy answer different questions

| Measure | What contributes |
| --- | --- |
| Held-out NLL or PPL | Probabilities of observed text tokens |
| Task accuracy | Whether the selected answer is correct |
| Per-language score | Behavior on a specified language slice |
| Cost per task | Resources consumed under the protocol |

A lower average loss does not guarantee every task improves.

Note:
Retrieve Lecture 02's units: PPL is exp(mean NLL in nats per scored token). Its direct comparison requires compatible tokenization and evaluation conventions. BPB changes the normalization unit but still requires the same text and scoring scope. Task metrics may use a subset of outputs and a discontinuous decision rule. A model can change answer probabilities without changing its selected answer, or improve one domain while losing another. Source: ../../docs/lecture-02-lm-metrics.md. No loss values are invented for the synthetic accuracy fixtures.

---

<!-- .slide: id="prompt-sensitivity" -->

## Prompt selection is part of model selection

| Synthetic accuracy on 100 items | Model A | Model B |
| --- | ---: | ---: |
| Template 1 | 60% | 64% |
| Template 2 | 62% | 58% |

Choose a prompt on development data, or report a prespecified set of prompts.

<p class="caption">Changing the prompt can change the model ranking.</p>

Note:
The table is a constructed counterexample, not empirical evidence about a model family. Both prompts use the same item population. Reporting only each model's best test prompt introduces selection bias. Shared prompts and model-specific prompts answer different questions; either needs a declared protocol. Choice ordering, chat templates, truncation, and answer extraction can also change results. MMLU-Pro's prompt sensitivity study is a concrete reading: https://arxiv.org/abs/2406.01574, experiments on prompt variations.

---

<!-- .slide: id="uncertainty-sources" -->

## What uncertainty does a result include?

| Source | Useful evidence |
| --- | --- |
| Which evaluation items appear | Item or cluster resampling |
| Training randomness | Independent training seeds |
| Generation randomness | Repeated samples with fixed settings |
| Prompt or grader choices | Prespecified robustness checks |

One interval cannot automatically cover all four.

Note:
The next bootstrap conditions on two fixed fitted systems and their fixed prompts. It estimates sampling uncertainty across independent items drawn from the same population. It does not include training-run variation. Related examples may require resampling document or task groups rather than individual rows. Source: Dror et al., https://aclanthology.org/P18-1128/, statistical testing assumptions and test selection. The resampling implementation and limitations are documented in assets/evaluation.py.

---

<!-- .slide: id="paired-items" -->

## Compare the same items together

| Clean synthetic items | Count |
| --- | ---: |
| Both A and B correct | 50 |
| Only A correct | 10 |
| Only B correct | 14 |
| Both wrong | 26 |

$$\Delta=\mathrm{accuracy}(B)-\mathrm{accuracy}(A)=\frac{14-10}{100}=4\text{ percentage points}$$

Note:
Students often compare two independent error bars even though both models answer the same questions. The paired difference uses the joint outcomes. Here 24 items disagree. Pair each item's two predictions before resampling; independent resampling of A and B loses that dependence. All counts derive from the explicit synthetic vectors used by E04.

---

<!-- .slide: class="exercise" id="exercise-05" -->

<p class="exercise-meta">Exercise E05 · 8 minutes · Notebook E05</p>

## How strong is the four-point improvement?

Resample the 100 paired items with replacement 5,000 times.

1. Compute the accuracy difference in each resample.
2. Read the central 95% percentile interval.
3. State which uncertainty this interval leaves out.

<div class="answer fragment"><p>The interval crosses zero. These items do not establish a reliable positive gain.</p></div>

Note:
The notebook fixes the seed at 20261028 and uses a transparent percentile implementation. It reports the exact computed endpoints and the point estimate. An interval crossing zero does not prove the models are identical. The small fixture is for understanding the procedure, not deciding a research result from one run. Expected limitations include training seeds, prompt choice, grader errors, and violations of item independence. Use this exercise to practice a qualified conclusion.

---

<!-- .slide: id="bootstrap-result" -->

## The paired differences include both signs

<div class="plot" data-plotly="assets/bootstrap.json" role="img" aria-label="Histogram of 5000 paired bootstrap accuracy differences from synthetic predictions. The distribution includes negative and positive differences around a four-percentage-point estimate."></div>

<p class="caption">95% interval: −5 to +13 points. Dashed line: zero. Synthetic items; fixed systems.</p>

Note:
Read the units as percentage points, not relative percent improvement. The plot is generated directly from the same seeded samples as Notebook E05. The density is a distribution of resampled differences, not a Bayesian posterior probability that B is better. More representative independent evaluation items or repeated training runs answer different uncertainty questions; choose the next experiment based on the dominant uncertainty and compute budget.

---

<!-- .slide: id="error-analysis" -->

## Error analysis turns a score into a hypothesis

| Inspect | Possible next comparison |
| --- | --- |
| Both systems fail the same items | Review labels and missing skills |
| One language loses after filtering | Audit retained language coverage |
| Answer text is right but parsing fails | Test a corrected parser on development data |
| Only long inputs fail | Check truncation and context limits |

Note:
These are hypotheses to investigate, not diagnoses guaranteed by the observed pattern. Inspect raw predictions alongside labels and item metadata. Avoid changing many parts of the pipeline at once. If label errors are discovered, keep an audit trail and report the revised dataset version. The purpose is to decide which evidence would discriminate among explanations, building on the course's individual-project requirements.

---

<!-- .slide: id="project-evaluation" -->

## An evaluation plan for your project

Write down:

- The claim and a relevant baseline.
- Evaluation examples, provenance, and split boundaries.
- The exact prompt, tool budget, and scoring rule.
- One controlled comparison and its resource cost.
- An uncertainty estimate and examples of failures.

Note:
Students can adapt the notebook's final short template to the existing project plan. This creates no new graded submission or reference-count requirement. A CPU study or a small supplied-result analysis can be useful if it tests a meaningful claim. A frontier benchmark may be too expensive or poorly matched to the project; use a smaller relevant comparison and state the scope. Keep the same learning goals and expectations for all students.

---

<!-- .slide: id="summary" -->

## What makes an improvement credible?

- A data policy with inspectable examples and provenance.
- A comparable training and evaluation budget.
- A versioned benchmark and explicit scoring protocol.
- A conclusion supported by uncertainty and error analysis.

**The result should answer the question the experiment was designed to test.**

Note:
Return to the original question. Deduplication can change exposure frequency and content coverage. Evaluation quality determines whether the comparison supports a claim. The synthetic example demonstrates how leakage and finite samples can undermine a tempting conclusion. Future sessions apply the same principles to adapting pretrained models. Leave time for students to name one change they will make to their own evaluation record.

---

<!-- .slide: class="references" id="methods-reading" -->

## Reading: data and evaluation methods

- [Lee et al., 2022](https://aclanthology.org/2022.acl-long.577/): deduplication methods and overlap analysis.
- [FineWeb, 2024](https://arxiv.org/abs/2406.17557): data-construction ablations.
- [Dror et al., 2018](https://aclanthology.org/P18-1128/): assumptions behind statistical comparisons.
- [EleutherAI evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness): task definitions and scoring interfaces.

Note:
Choose sections that support the student's question. These are relevant readings, not a fixed reference-count requirement or an invitation to reproduce full-scale training. The reading guide provides the Spring adaptation map and additional task-specific sources.

---

<!-- .slide: class="references" id="references" -->

## Reading: current benchmark protocols

- [HLE-Diamond](https://labs.scale.com/leaderboard/hle-diamond): revised expert questions.
- [SWE-Bench Pro V2](https://labs.scale.com/blog/swe-bench-pro-v2): task repair and grading controls.
- [ARC-AGI-3](https://arcprize.org/arc-agi/3): interactive evaluation.
- [LiveBench](https://livebench.ai/): dated benchmark releases.
- [Full reading guide](optional-reading.md): historical tasks, model interfaces, and code evaluation.

<p class="caption">Sources checked October 23, 2026. Preserve the exact benchmark version in any reported result.</p>

Note:
All named current releases have primary publisher references. Read protocols and dataset versions before consulting rankings. The course does not promise that a current leaderboard remains unchanged, or that benchmark access and full-suite runs fit the course pool. The reproducible classroom examples require no network, model download, or external API.
