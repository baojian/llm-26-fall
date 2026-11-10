<!-- .slide: class="title-slide" id="title" -->

<p class="eyebrow">CS40008.01 · Week 8 · October 28, 2026</p>

# InstructGPT and reward models

<p class="subtitle">Supervised fine-tuning, preferences, and learned feedback</p>

<p class="byline">Baojian Zhou<br>Fudan University · Fall 2026</p>

Note:
This deck consolidates the Week 8 topics the instructor reported teaching. Use the notes for longer explanations; the material is organized for three 45-minute periods, not a claim about the exact historical minute-by-minute sequence. The PPO derivation belongs to Lecture 09. All unlabelled numerical examples are synthetic. Primary case study: Ouyang et al., https://arxiv.org/html/2203.02155v1, Sections 2–3.

---

<!-- .slide: id="outcomes" -->

## What you will be able to explain

- Separate demonstrations, preferences, and rewards.
- Write the response-only SFT loss and check its mask.
- Derive the pairwise reward-model loss.
- Distinguish policy improvement from reward-score improvement.

<p class="caption">Four ungraded exercises; offline CPU notebook.</p>

Note:
Ask students what changes after pretraining: the model family can stay the same while the data, objective, and evaluation change. Prerequisites are the causal next-token loss and the held-out evaluation conventions from Lectures 02 and 06. No new graded work is introduced by these exercises.

---

<!-- .slide: class="outline-slide" id="outline-sft" -->

## Lecture outline

<ol class="outline-topics"><li aria-current="step">Demonstrations and supervised fine-tuning</li><li>Preferences and reward models</li><li>Policy optimization and evaluation</li></ol>

Note:
Period 1. Begin with an actual instruction rather than starting with three unexplained acronyms. The first exercise checks shifted target positions before discussing reward modeling.

---

<!-- .slide: id="prediction-task" -->

## A likely continuation may ignore the instruction

**Prompt:** “Summarize this paragraph in one sentence.”

Possible continuations include an answer, another instruction, or a transcript.

Pretraining estimates text continuation probabilities. Instruction following
adds evidence about the behavior we want.

Note:
These are constructed possible continuations, not generated outputs or findings about a named model. Explain that an instruction-following model still predicts tokens; post-training changes which continuations receive probability. Do not imply that pretraining never produces instruction following.

---

<!-- .slide: id="instructgpt-stages" -->

## The InstructGPT workflow

| Stage | Data | Object learned |
| --- | --- | --- |
| SFT | Demonstrated responses | Response policy |
| Reward modeling | Ranked candidate responses | Scalar scorer |
| Policy optimization | Sampled responses and scores | Updated policy |

<p class="source">Ouyang et al. (2022), Figure 2 and Section 3</p>

Note:
Source: https://arxiv.org/html/2203.02155v1. These are different learning problems. The original study reports preference results under its prompt distribution and annotation procedure; those results should not be generalized into a universal model ranking. PPO is an algorithm within the third stage, not another name for the whole workflow.

---

<!-- .slide: id="response-probability" -->

## The policy still predicts the next token

$$\pi_\theta(y\mid x)=\prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{<t})$$

The prompt $x$ supplies context.

The response $y$ includes its termination decision under the chosen convention.

Note:
This factorization is unchanged by switching from pretraining to SFT or policy optimization. Name which positions belong to the response. Prompt tokens can affect hidden states without contributing direct response target loss. Padding is not a response action.

---

<!-- .slide: id="sft-data" -->

## A demonstration specifies desired behavior

| Field | Example content |
| --- | --- |
| Instruction | Summarize the supplied paragraph |
| Context | The paragraph and any allowed evidence |
| Response | A checked one-sentence summary |
| Provenance | Source, version, authoring procedure |

Check correctness and instruction following separately.

Note:
This schema is a teaching example, not the original InstructGPT collection format. A grammatically good answer can still violate the request or invent facts. Preserve source and prompt-family identifiers so later splitting does not leak paraphrases or examples derived from the same source.

---

<!-- .slide: id="sft-loss" -->

## Response-only supervised loss

$$L=-\frac{\sum_{i,t}m_{i,t}\log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})}{\sum_{i,t}m_{i,t}}$$

$m_{i,t}=1$ for a scored response target; otherwise zero.

Use the mask on **shifted targets**. Prompt and padding targets are excluded.

Note:
This slide declares a token-mean reduction, not a claim that every SFT implementation uses this exact reduction. Teacher forcing uses the demonstrated prefix. Explain that taking a mean over sequence means instead changes the relative contribution of different response lengths.

---

<!-- .slide: id="shifted-mask" -->

## Align inputs, targets, and masks

| Input | Prompt-end | Answer A | Answer B | EOS |
| --- | --- | --- | --- | --- |
| Target | Answer A | Answer B | EOS | PAD |
| Score target? | Yes | Yes | Yes | No |

The first response token is predicted from the prompt.

Note:
The names denote token roles, not a particular tokenizer vocabulary. Earlier prompt positions have masked prompt targets. Do not mask the first response target just because its input position is the final prompt token. Include EOS only under the declared training convention and distinguish it from a length-cap truncation.

---

<!-- .slide: class="exercise" id="exercise-01" -->

<p class="exercise-meta">Exercise E01 · 6 minutes · Notebook E01</p>

## Which loss is being averaged?

Response A has scored probabilities $(1/2,1/4)$.
Response B has one scored probability $1/4$.

Compute the token mean and the mean of response means.

<div class="answer fragment"><p>Token mean: $\frac53\log2$. Response mean: $\frac74\log2$.</p></div>

Note:
Expected reasoning: the total NLL is 5 log 2 across three scored tokens; the sequence losses are 1.5 log 2 and 2 log 2. All prompt and padding targets are excluded. The two answers differ because the reductions assign different weights. Checked in the notebook; give students two minutes of discussion after the calculation.

---

<!-- .slide: id="chat-template" -->

## Chat formatting changes the token sequence

- Inspect role separators and special tokens.
- Check the assistant-response boundary after tokenization.
- Record truncation and termination rules.
- Reuse the model's tokenizer and declared template.

<p class="caption">A printed conversation is not the encoded training example.</p>

Note:
This is an implementation principle, not advice to choose a new template for every evaluation. Pin the tokenizer and template revision. Show an encoded example in a practical project before training, particularly when a template adds assistant headers or end-of-turn tokens.

---

<!-- .slide: id="sft-limit" -->

## Lower demonstration loss is one piece of evidence

Teacher forcing uses demonstrated prefixes.

Generation uses the model's own prefixes.

Compare held-out generated behavior, including regressions and failures.

Note:
A good training fit does not establish robustness to generated prefixes or unseen instruction types. Keep decoding settings and the evaluated prompt population fixed in the comparison. This provides the conceptual bridge to Lecture 10 rather than adding another required training run.

---

<!-- .slide: class="outline-slide" id="outline-reward" -->

## Lecture outline

<ol class="outline-topics"><li>Demonstrations and supervised fine-tuning</li><li aria-current="step">Preferences and reward models</li><li>Policy optimization and evaluation</li></ol>

Note:
Period 2. Ask why people might find comparing two answers easier than writing an ideal answer from scratch. Comparison labels still depend on the rubric and the population providing them.

---

<!-- .slide: id="preference-data" -->

## A preference is a comparison

$(x,y_w,y_l)$ means response $y_w$ was preferred to $y_l$ for prompt $x$.

Record the rubric, candidate source, and disagreement.

A winner label does not supply an absolute numerical reward.

Note:
InstructGPT Section 3.2 and Appendix B describe human feedback and annotation procedures: https://arxiv.org/html/2203.02155v1. Do not erase ties by inventing a winner. Several pairs obtained from one ranking are correlated and should not be treated as independent prompts.

---

<!-- .slide: id="reward-scorer" -->

## A reward model produces one score per response

$$r_\phi(x,y)\in\mathbb R$$

The same scorer processes both candidates.

Training uses their score difference to predict the observed comparison.

Note:
An implementation can use a scalar head on a language-model representation, with an explicit choice of which response position is pooled. The score is not a token probability. Its numerical origin is not identified by same-prompt pair comparisons.

---

<!-- .slide: id="bradley-terry" -->

## Convert a score difference to a probability

$$P(y_w\succ y_l\mid x)=\sigma(r_w-r_l)$$

$$\sigma(d)=\frac{1}{1+e^{-d}}$$

A zero margin gives probability $1/2$.

<p class="source">Pairwise Bradley–Terry model; InstructGPT Section 3.2</p>

Note:
Source: https://arxiv.org/html/2203.02155v1. This is a modeling assumption for comparison labels. It does not assert that all annotators share one transitive utility or that the inferred probability is calibrated on another prompt population.

---

<!-- .slide: id="preference-curve" -->

## The margin controls preference probability

<div class="plot" data-plotly="assets/preference-probability.json" role="img" aria-label="Sigmoid preference probability versus reward margin, with a zero margin giving one half and log three giving three quarters."></div>

<p class="caption">Analytical curve, not a fitted reward model.</p>

Note:
Hover to inspect the values. The plotted function is exactly sigmoid(d), generated from the standard library. Use the horizontal symmetry to explain swapping winner and loser. The curve's shape is a consequence of the model, not evidence that preferences in a particular dataset follow it.

---

<!-- .slide: id="reward-loss" -->

## Fit the observed winner

$$L_{\mathrm{RM}}=-\mathbb E\log\sigma(r_w-r_l)$$

For margin $d$, $\frac{dL}{dd}=\sigma(d)-1$.

Use a stable log-sigmoid or softplus implementation.

Note:
At zero margin the derivative is minus one half. Gradient descent increases the margin contributed by this pair. Do not independently regress the winner score to 1 and the loser score to 0 and call that the same objective. Both candidate scores need gradients. Large negative margins require stable arithmetic.

---

<!-- .slide: class="exercise" id="exercise-02" -->

<p class="exercise-meta">Exercise E02 · 5 minutes · Notebook E02</p>

## What does a reward score identify?

Let $r_w=\log3$ and $r_l=0$.

Compute the preference probability and pair loss.
Then add 10 to both scores.

<div class="answer fragment"><p>Probability $3/4$; loss $\log(4/3)$. The common shift changes neither.</p></div>

Note:
The score difference is the identified quantity in this example. More generally, a prompt-dependent common shift cancels. This does not mean multiplying all scores by a constant leaves probabilities unchanged. Ask students to distinguish a shift from a scale change.

---

<!-- .slide: id="reward-shortcut" -->

## A shortcut can predict the training labels

Suppose preferred training responses are usually longer.

A scorer can reward length without detecting correct answers.

Test comparisons where answer quality and length disagree.

Note:
This is a hypothetical failure mode rather than a measured property of InstructGPT. Suggest a held-out slice with a concise correct answer against a long incorrect answer. Keep the rubric fixed and inspect annotator disagreement rather than assuming every length effect is necessarily a bug.

---

<!-- .slide: class="exercise" id="exercise-03" -->

<p class="exercise-meta">Exercise E03 · 6 minutes · Notebook E03</p>

## Did the reward-model split leak?

| Pair | Prompt family | Proposed split |
| --- | --- | --- |
| p1 | summary-17 | Train |
| p2 | summary-17 | Test |
| p3 | summary-29 | Test |

<div class="answer fragment"><p>p1 and p2 share a family. Assign complete families before deriving pairs.</p></div>

Note:
Expected response identifies the grouping boundary and explains why different pair IDs do not make independent examples. The notebook implements an overlap audit without deleting data. A separate held-out source or time period may be needed for a broader generalization claim.

---

<!-- .slide: id="reward-evaluation" -->

## Reward-model evaluation has several questions

| Measurement | Question |
| --- | --- |
| Pair accuracy | Is the winner ranked higher? |
| Pair log loss | How much probability is assigned to labels? |
| Slice audit | Does a shortcut explain the decisions? |
| Optimized outputs | Does maximizing the score remain useful? |

Note:
These questions require different data. Pair accuracy does not measure absolute calibration or robustness to optimization-induced distribution shift. Keep a checked collection of disagreement examples. Lecture 10 will connect this to paired evaluation and uncertainty.

---

<!-- .slide: class="outline-slide" id="outline-policy" -->

## Lecture outline

<ol class="outline-topics"><li>Demonstrations and supervised fine-tuning</li><li>Preferences and reward models</li><li aria-current="step">Policy optimization and evaluation</li></ol>

Note:
Period 3. Introduce the regularized objective and the different model roles, leaving the policy-gradient derivation to Lecture 09. Ask why optimizing a proxy might create examples unlike the reward model's original comparison data.

---

<!-- .slide: id="regularized-objective" -->

## Reward with a reference anchor

$$J=\mathbb E_{y\sim\pi_\theta}[r_\phi(x,y)]-\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})$$

Here the prompt is fixed; training also averages over prompts.

The reference is often the SFT policy.

Note:
Source: InstructGPT Section 3.2, https://arxiv.org/html/2203.02155v1. Both distributions are complete-response distributions conditioned on the same prompt. The KL penalty discourages movement from the anchor, but does not prove the reward model is correct or eliminate reward hacking.

---

<!-- .slide: id="sampled-kl" -->

## A sampled log ratio is not the full KL

$$\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}=\sum_t\log\frac{\pi_\theta(y_t\mid s_t)}{\pi_{\mathrm{ref}}(y_t\mid s_t)}$$

A single sampled value may be negative.

Its expectation under the current policy is nonnegative.

Note:
Assume reference support on the policy's allowed responses. The expectation of the sequence log ratio is the sequence KL. A token average changes the length-weighting convention and should be stated separately. Lecture 09 discusses what changes when differentiating a penalty on a fixed rollout batch.

---

<!-- .slide: id="model-roles" -->

## Keep the computational roles distinct

| Object | Role |
| --- | --- |
| Current policy | Receives the actor update |
| Old policy | Generated the rollout batch |
| Reference policy | Persistent regularization anchor |
| Reward model | Scores a completed response |
| Critic | Estimates remaining return from a prefix |

Note:
Some implementations share parameters or memory between roles; that does not merge their mathematical responsibilities. The reward model is fixed during an ordinary actor update. The critic is trained to a return target tied to the policy and reward convention. PPO details are in the next lecture.

---

<!-- .slide: class="exercise" id="exercise-04" -->

<p class="exercise-meta">Exercise E04 · 5 minutes · Notebook E04</p>

## Which denominator belongs in the update ratio?

For one sampled token:
$\pi_{\mathrm{old}}=0.20$, $\pi_{\mathrm{ref}}=0.10$, $\pi_\theta=0.24$.

Find the rollout ratio and the reference log ratio.

<div class="answer fragment"><p>Rollout ratio: $1.2$. Reference log ratio: $\log2.4$.</p></div>

Note:
The probabilities refer to the same token at the same prefix. The current-to-old ratio is 0.24/0.20; using the reference instead would produce 2.4 and change the update. The reference log ratio is the value for the current policy here; a rollout-time shaped reward would use cached old versus reference probabilities instead.

---

<!-- .slide: id="ptx" -->

## The training recipe includes auxiliary objectives

InstructGPT's PPO-ptx variant also includes a pretraining-likelihood term.

Record the objective mixture and evaluation regressions.

The name “PPO” does not fully specify a post-training experiment.

Note:
Source: https://arxiv.org/html/2203.02155v1, Section 3.2 and alignment-tax discussion. Do not add a pretraining mixture as an unannounced course requirement. This slide explains why two systems both called PPO can differ in their actual optimization objectives.

---

<!-- .slide: id="proxy-failure" -->

## More reward can accompany worse task performance

| Synthetic checkpoint | Reward score | Checked task accuracy |
| --- | ---: | ---: |
| Before update | 0.4 | 70% |
| After update | 1.2 | 62% |

Inspect outputs, the reward model, and the evaluator before claiming improvement.

Note:
These invented values illustrate a logically possible proxy failure, not measured InstructGPT results. An alternative explanation could be an evaluation bug. The appropriate next step is to inspect examples and compare under a frozen protocol, not conclude that all policy optimization is harmful.

---

<!-- .slide: id="three-claims" -->

## Match each claim to its evidence

| Claim | Needed comparison |
| --- | --- |
| Fits demonstrations | Held-out response likelihood |
| Predicts preferences | Held-out comparisons and slices |
| Improves the assistant | Controlled generated-output evaluation |

One optimization trace cannot establish all three claims.

Note:
Conclude by asking students where each evaluation split must be protected in the pipeline. The same held-out prompt family must not re-enter through SFT, reward-model pairs, or policy-training prompts. This is the bridge to Lecture 10.

---

<!-- .slide: class="references" id="references" -->

## Reading and next lecture

- [InstructGPT](https://arxiv.org/html/2203.02155v1): Sections 2–4; annotation appendix.
- [PPO](https://arxiv.org/abs/1707.06347): Sections 2–3; Algorithm 1.
- [Lecture 08 notes](../../docs/lecture-08-instructgpt.html): losses and worked examples.
- [Post-training reading guide](../../docs/rl-posttraining-and-agents.md).

Note:
Lecture 09 develops REINFORCE and PPO, then provides clearly labelled GRPO/DPO supplementary material. All practice here is optional and ungraded. Primary source URLs were checked November 10, 2026.
