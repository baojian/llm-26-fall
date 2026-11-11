# Lecture 10: Data preparation, evaluation, and experimental design

Week 10, November 11, 2026. This lecture connects the post-training workflow
to the evidence needed to trust an improvement. The central question is:
**does the measured gain survive a controlled comparison on relevant held-out data?**

## 1. Define the claim before building the pipeline

“Better data” needs a task, population, budget, and measurement. A useful
question is whether removing exact duplicates improves held-out instruction
following at a fixed training-token budget, while preserving performance on
short factual answers. A change in aggregate score alone does not explain
which population improved or which variable caused it.

Write an experiment record before running the final comparison: treatment,
baseline, frozen model/tokenizer revision, data version, split unit, training
budget, development-selection rule, evaluation protocol, and the smallest
effect that would matter. Keep evaluation data out of every earlier fitting
or selection stage.

## 2. A traceable data pipeline

| Stage | Output to preserve | Example failure |
| --- | --- | --- |
| Acquisition | Source, version, access conditions, source ID | Unknown provenance |
| Extraction | Text plus parent-document ID | Menus replace article content |
| Filtering | Decision and reason | Useful short answers rejected |
| Matching | Exact keys or approximate match evidence | False duplicate merges |
| Splitting | Family/group assignment | Related examples cross the split |
| Tokenization and packing | Frozen tokenizer, boundaries, masks | Prompt tokens scored as response targets |
| Sampling | Source weights and exposure counts | A small source repeated excessively |

Preserve source identifiers through chunking and augmentation. Several chunks
from the same document are not independent examples. A license or access record
is separate from an estimate of content quality. A filter score also does not
prove that text is factually correct.

For SFT, inspect instruction/response alignment and the response-token mask.
For preferences, retain pair provenance, the rubric, and correlated candidate
groups. For reward optimization, separate policy-training prompts, reward-model
training pairs, and evaluation prompts. All three can become leakage routes.

| Taught stage | Fitting data | Boundary to audit |
| --- | --- | --- |
| SFT, Lecture 08 | Demonstrations and response masks | Evaluation answers or related source documents in demonstrations |
| Reward model, Lecture 08 | Preference pairs and rubric | Evaluation prompt families in reward-model fitting or selection |
| PPO, Lecture 09 | Prompts, sampled responses and rewards | Evaluation tasks exposed during policy optimization |

A prompt held out from PPO is not necessarily held out from SFT or reward-model
training. Carry source-family IDs across stages and reserve final evaluation
before model or prompt selection. The three training pools need not be mutually
disjoint; the claim concerns their relationship to the evaluation population.
When pretraining exposure cannot be audited, record that uncertainty.

### Task definitions become data schemas

The instructor's 2025 benchmark lecture provides concrete preparation cases.
Start by specifying what one example contains, what the target means, and
which source identifiers must survive transformation.

| Case | Input and target | Preparation check |
| --- | --- | --- |
| CoLA | Sentence → binary acceptability label | Separate the label from the Matthews correlation metric, whose range is −1 to 1. |
| STS-B | Sentence pair → similarity score on a 0–5 scale | Preserve a real-valued target; define annotation anchors. |
| MNLI / RTE | Ordered premise and hypothesis → relation label | Store label names and the release-specific integer mapping. |
| MultiRC | Passage, question and answer option → binary decision | Keep passage/question IDs and the complete option set. |

These task distinctions come from [GLUE, Section 3](https://arxiv.org/html/1804.07461v3#S3),
[STS, Section 2 and Table 1](https://aclanthology.org/S17-2001/), and
[SuperGLUE, Sections 3.2–3.3](https://arxiv.org/html/1905.00537v3#S3).
The STS source establishes the lower endpoint as **0**; the earlier lecture
and GLUE paper's 1–5 wording should not be used to reject valid zero scores.

Write a short annotation rubric with positive, negative and borderline examples.
Pilot it with independent judgments, retain disagreements, and record how they
were resolved. Store the rubric revision alongside labels. For Week 8 preference
pairs, also preserve candidate order, ties and annotator agreement: a grammatical
sentence is not automatically the preferred response under a helpfulness rubric.
These are recommended course data practices, not a claim that each historical
dataset used the same annotation procedure.

## 3. Extraction and filtering

Extraction choices decide what becomes training text. Compare raw source,
extracted text, and the downstream example on a small audited sample. Look
for navigation, repeated footers, missing equations, broken lists, and lost
speaker roles. Preserve enough information to reproduce a decision.

Quality is conditional on use. A rule rejecting all short documents could
remove exact answers and concise code snippets. A language classifier can make
errors on mixed-language text. Inspect both retained and rejected examples,
including examples near the threshold. Report rejection rates by source or
slice, rather than only a global retained-token count.

Primary examples: [FineWeb](https://arxiv.org/abs/2406.17557),
[Dolma](https://arxiv.org/abs/2402.00159), and
[DataComp-LM](https://arxiv.org/abs/2406.11794). Their large published experiments
are methodological references, not the required course compute budget.

### Distractor construction is a quality decision

SWAG builds continuation choices using generated negatives and adversarial
filtering; HellaSwag revisits the remaining shortcuts. Read the construction
method in [SWAG, Section 2](https://aclanthology.org/D18-1009/) and the
ablation in [HellaSwag, Section 3.2 and Figure 4](https://aclanthology.org/P19-1472/).

For a course dataset, retain candidate provenance, use a development audit to
identify easy stylistic cues, then have people check that the intended answer is
supported and alternatives are unsuitable. A difficult distractor can instead
be another valid answer. Inspect length, formatting and answer-position cues;
try an answer-only baseline. Strong performance without the required context
is evidence to investigate a shortcut. Low performance does not certify that
all shortcuts are absent. Freeze the final evaluation after construction choices.

## 4. Exact and approximate duplicates

An exact key is only as meaningful as its normalization rule. Unicode NFC,
line-ending normalization, and trimming outer whitespace can remove superficial
differences. Lowercasing, punctuation removal, or aggressive whitespace changes
can collapse examples whose distinctions matter. State the rule explicitly.

For sets of word shingles $A$ and $B$, Jaccard similarity is

$$J(A,B)=\frac{|A\cap B|}{|A\cup B|}.$$

With a uniformly random ordering of the universe, the minimum element of
$A\cup B$ belongs to the intersection with probability $J(A,B)$. Thus
MinHash collisions estimate that similarity. A finite signature is an estimate,
and a match proposed by locality-sensitive hashing still needs the declared
verification rule.

Under ideal independent MinHash rows, $b$ bands of $r$ rows give candidate
probability $1-(1-s^r)^b$ at similarity $s$. This is not the actual similarity
of a particular document pair. Connected components of a match graph can also
contain endpoints below the pairwise threshold.

## 5. Split by the unit that can leak

Choose a grouping unit appropriate to the claim: source document, prompt
family, author, repository, patient, time period, or interaction environment.
For preference data, all pairs derived from one prompt should remain together.
For chunked documents, keep parent and related-family IDs available.

Establish split membership before generating dependent pairs, chunks, or
augmentations. Audit exact and approximate overlaps across splits. When
grouping must change, preserve the mapping and reasons. A random row split
does not make rows independent, and exact deduplication does not rule out
semantic overlap or prior pretraining exposure.

Development data selects thresholds, prompts, hyperparameters, and stopping
points. A final test is used after these choices are fixed. Repeatedly trying
options on the final test turns it into development data, even if no gradients
were computed on its text.

### QNLI-style conversion: one source produces many rows

QNLI converts question answering into question–sentence classification.
The historical conversion motivates this **authored illustration**, not a
reproduction or proposed re-split of released QNLI.

Article 17 contains: S1, “Noa stores the model on the shared drive.” S2,
“Mina writes the report.” Q1 asks where Noa stores the model; Q2 asks who
writes the report. Pairing questions with sentences produces:

| Article | Question | Sentence | Contains its answer? |
| --- | --- | --- | --- |
| 17 | Q1 | S1 | Yes |
| 17 | Q1 | S2 | No |
| 17 | Q2 | S1 | No |
| 17 | Q2 | S2 | Yes |

Q1 in training and Q2 in development have disjoint question IDs but share the
article. For a claim about unseen articles, assign article families to splits
before materializing these rows; inherit that assignment in every row. If
the claim concerns new questions about familiar articles, describe that different
setting explicitly. Preserve published benchmark splits when reproducing scores.
Source: [GLUE, Section 3.3 and Appendix A](https://arxiv.org/html/1804.07461v3#A1).

The scoring unit can also differ from the row unit. MultiRC reports answer-option
F1 and exact match of each question's entire answer set. Four correct options
out of five do not imply a correctly answered question. Keep the identifiers
needed for both aggregation and group-aware uncertainty.
[SuperGLUE, Table 3](https://arxiv.org/html/1905.00537v3#S5.T3).

## 6. Match exposure and compute

Removing records changes corpus size. Equal epochs then give different
training-token counts. A data-policy study should freeze the model, tokenizer,
optimizer, and processed-token budget, while reporting both retained tokens
and actual processed tokens.

For source $i$ with $N_i$ retained tokens and sampling probability $p_i$,
an expected exposure under budget $B$ is

$$B_i=Bp_i,\qquad E_i=\frac{Bp_i}{N_i}.$$

$E_i$ is an expected pass count under the stated token-sampling convention,
not a guarantee that every token is seen that many times. Sampling documents
uniformly creates different token weights when their lengths differ.

Keep total GPU-hours across all devices and runs, or CPU time for CPU studies.
Record model and dataset sizes separately from runtime. An external resource
has a source and cost; it is not an allocation from the course pool.

## 7. A benchmark is a complete protocol

The measurement includes dataset revision, task interface, prompt/template,
decoding budget, answer extraction, grader, aggregation, and uncertainty.
A familiar benchmark name does not pin all of these. Changing the number of
attempts or giving tools changes the evaluated system.

| Evaluation family | Example measurement | Limitation to inspect |
| --- | --- | --- |
| Language modeling | Token loss, PPL, BPB | Tokenizer and context conventions |
| Classification or candidate scoring | Accuracy, macro-F1 | Label mapping, imbalance, candidate normalization |
| Generated answers | Exact match, rubric, checked tests | Equivalent answers, weak graders |
| Preferences | Paired win rate | Ties, length/style bias, judge identity |
| Code or interaction | Tests passed, tasks completed, budget | Environment, retries, hidden feedback |

GLUE, SuperGLUE, and HANS remain useful teaching cases for task construction
and shortcut analysis. Modern benchmark examples are useful for studying
versioning, graded reasoning, tools, and interaction. They are not a substitute
for evaluation matched to a project's question.

### Controlled examples reveal what a dataset measures

**LAMBADA** selected passages through full-context and limited-context prediction
checks. Its 10,022 filtered passages form development and test sets; its separate
training text comes from different novels and is not filtered in the same way.
[Paperno et al., Sections 3.1–3.2](https://aclanthology.org/P16-1144/).
For a new context-use study, retain book/document IDs, document the context
window, and compare both conditions on the same items. Correct prediction of a
target word spanning several BPE tokens requires scoring the complete word.

**BLiMP** uses grammar-generated minimal pairs and checks which sentence receives
higher probability. [Warstadt et al., Sections 2–3](https://aclanthology.org/2020.tacl-1.25/).
An authored example is “The keys beside the folder are missing” versus “The keys
beside the folder is missing.” Preserve pair and template IDs, verify the intended
contrast, and check for lexical cues. Template-based generation can create many
related rows; a new-template generalization claim needs a corresponding split.
Do not equate grammatical acceptability with a Week 8 human-preference label.

**Dynabench** studies data collection with people challenging a model and
validating examples across rounds. [Kiela et al., Sections 2–3](https://aclanthology.org/2021.naacl-main.324/).
For a course adaptation, record the model, collection round, rubric and validation
status. Use discovered failures for development, preserve a separate final check,
and report performance by fixed evaluation round. Changing the test distribution
while changing the model prevents attributing the score difference to the model
alone. These design recommendations do not assume the historical web service
is currently available.

### Metadata depends on the task

The 2025 lecture's task catalog becomes a preparation checklist:

| Task family | Preserve before tokenization | Example audit |
| --- | --- | --- |
| Question answering | Source, question ID, answer spans or aliases | Answers remain supported after extraction. |
| Summarization | Document, reference and source-family IDs | Truncation has not removed reference evidence. |
| Translation | Aligned pair, languages and document IDs | Source/target alignment survives filtering. |
| Structured prediction | Text revision, token/span offsets, labels | Normalization has not shifted annotation offsets. |

This is a course-authored checklist, not an official scorer specification for
every dataset. Choose one family that matches the project claim. The optional
slides connect these older examples to the versioned modern protocols in the
reading guide, without treating historical leaderboards as current rankings.

## 8. Likelihood and generation answer different questions

Token perplexity is $\exp(H)$ for mean negative log likelihood $H$ in nats per
scored token. It is not directly comparable across tokenizers without controlling
the scoring unit and conventions. Bits per byte uses total negative log
likelihood divided by $\log 2$ and the original UTF-8 byte count.

For multiple-choice evaluation, sum response-token log probabilities under
each candidate, with the same context and boundary convention. Length-normalized
scores can rank candidates differently. State the choice rather than treating
normalization as an innocuous implementation detail.

For code, the HumanEval estimator for $n$ sampled completions, $c$ successes,
and $n\ge k$ is

$$\widehat{\mathrm{pass@}k}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}.$$

With $n=10,c=2$, pass@1 is $20\%$ and pass@5 is $7/9\approx77.8\%$.
The weights did not improve; the attempt budget changed. The measure also
does not supply a deployment method for choosing the successful answer without
a checker. Adaptive repair after test feedback is a different protocol.

Reading: [Chen et al., Section 3 and Appendix A](https://arxiv.org/abs/2107.03374).

## 9. Paired comparisons and uncertainty

Evaluate A and B on the same items and retain their paired outcomes. In a
synthetic set of 100 independent items, suppose 50 are correct for both,
10 only for A, 14 only for B, and 26 for neither. The accuracy difference is
$(14-10)/100=4$ percentage points.

A paired bootstrap resamples item indices and carries both systems' outcomes
together. Compute the difference for each resample, then a declared interval.
For the supplied fixture and seed, the central 95% percentile interval is
approximately -5 to +13 percentage points. It crosses zero: the observed
sample does not establish a reliable positive gain. This does not prove the
systems are equivalent.

The interval conditions on the fitted systems, prompts, and grader. It omits
training-seed variation, prompt selection, label error, and domain shift.
Related observations need resampling at the relevant group level. More rows
from the same family may add less independent evidence than new families.

Reading: [Dror et al. (2018), assumptions and test selection](https://aclanthology.org/P18-1128/).

## 10. Validate the evaluator

### Worked example: reward coverage

Suppose the task is to return the square of an input. The training reward only
checks input 2. Both `return 4` and `return x * x` receive full reward. An
independent check at input 3 expects 9 and rejects the hard-coded candidate.

| Candidate | Reward case: input 2 | Independent case: input 3 |
| --- | --- | --- |
| Hard-coded answer | 4: pass | 4: fail |
| Square function | 4: pass | 9: pass |

The notebook executes this authored fixture. It demonstrates an incomplete
verifier, not a trained-policy result. Even passing both tests does not prove
general correctness. Once a check guides a repair or candidate selection,
it is development evidence; retain a separate final evaluation for the claim.
This revisits the post-training objectives covered across Lectures 8–10 and
prepares the independent evaluation of Lecture 11's adaptation experiments.
For an empirical study of optimizing a proxy, read
[Gao et al., Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760).
Their reward-model experiments are distinct from this two-function illustration.

### Audit disagreements and failure slices

An exact matcher can reject an equivalent answer. A unit-test suite can miss
incorrect behavior. An LLM judge can change its decision when answer order or
presentation changes. Human assessors can disagree about an ambiguous rubric.
These are failure modes to investigate, not universal error rates.

Build a small checked set containing clear successes, clear failures, and
borderline cases. Test order swaps for pairwise judgments and state how ties
are counted. Record the judge model revision, prompt, rubric, and access to
reference answers. Keep its training and selection history separate from the
system being evaluated when possible.

Error analysis should produce a testable next question. “Long responses fail
more often” could reflect task difficulty, truncation, or a scoring artifact.
Inspect examples before changing the model. Choose the next experiment on
development data, then use a fresh or preserved final evaluation for the claim.

## 11. A minimum useful experiment report

State the question, treatment, and baseline. Document data provenance and
splits, actual training exposure, model/template revisions, and evaluator.
Report per-item or per-group outcomes, relevant slices, uncertainty, and a few
checked failures. Identify which choices were made before final evaluation.

A negative result can be informative when the implementation and comparison
are sound. A large score increase with leakage or a changed attempt budget
does not answer the intended question. Tie the conclusion to the tested
population and budget instead of claiming general model superiority.

## Practice and reading

The notebook provides five core offline exercises on duplicate/split audits,
source exposure, evaluation composition, and paired uncertainty. Additional
practice covers candidate scoring, pass@k, and MinHash. Every dataset is a
labeled teaching fixture; the core practice uses no model download or GPU.
E02 also checks why disjoint question IDs do not establish an unseen-article split.

- [Lecture 10 reading guide and versioned benchmark examples](../slides/lecture-10/optional-reading.md).
- [Lecture 08: SFT and reward-model data](lecture-08-instructgpt.html).
- [Lecture 09: policy-training and evaluation distributions](lecture-09-policy-optimization.html).
- [Lecture 11 plan: efficient adaptation and distillation](course-revision.html#lecture-11-efficient-adaptation-and-distillation).
- [FineWeb methods and controlled data comparisons](https://arxiv.org/abs/2406.17557).
- [DataComp-LM benchmark design](https://arxiv.org/abs/2406.11794).
