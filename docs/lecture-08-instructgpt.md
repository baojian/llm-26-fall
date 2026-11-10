# Lecture 08: InstructGPT, supervised fine-tuning, and reward models

Week 8, October 28, 2026. These notes consolidate the topics covered in class.
The next lecture develops the policy-gradient and PPO stage. All worked
numbers below are teaching examples unless explicitly attributed to a study.

## 1. Prediction and instruction following

A pretrained causal language model estimates a distribution over continuations.
For a prompt $x$ and response $y=(y_1,\ldots,y_T)$,

$$\pi_\theta(y\mid x)=\prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{<t}).$$

Predicting text that occurs frequently in a corpus does not specify what a
helpful assistant should do. Given a question, a continuation might answer it,
repeat it, write another question, or continue an imagined conversation.
Instruction following introduces a target behavior and evidence about that
behavior. It does not remove the need to check factual correctness.

InstructGPT is the case study for this transition. Its workflow combines
demonstrations, human comparisons, and optimization against a learned reward.
The original study compares systems on its specified prompt distribution and
with its evaluation procedure. Its preference findings are not a universal
ranking of models on every population or task.

Reading: [Ouyang et al., Sections 2–3 and Figure 2](https://arxiv.org/html/2203.02155v1).

## 2. Three datasets, three learning problems

| Stage | Observations | Learned object | Immediate objective |
| --- | --- | --- | --- |
| Supervised fine-tuning, SFT | Prompt and demonstrated response | A response policy | Increase demonstrated response likelihood |
| Reward modeling | Prompt, preferred response, rejected response | A scalar scoring function | Predict comparisons |
| Policy optimization | Prompts, sampled responses, feedback | An improved response policy | Increase regularized expected reward |

The datasets need not contain the same prompts. Separate their purposes and
their split boundaries. A response selected by an annotator is an observation
of a preference under a rubric, not an automatically calibrated numerical reward.

Keep the original pretrained model, the SFT model, the reward model, and the
eventual optimized model distinct in experiment records. “RLHF” names a
workflow; PPO is one possible algorithm used within its policy-optimization stage.

## 3. The supervised fine-tuning loss

Let $m_{i,t}$ indicate a response target token in example $i$. Prompt tokens
provide context but receive no direct target loss in this response-only setup.
Padding also receives no loss. One token-averaged convention is

$$L_{\mathrm{SFT}}(\theta)=-\frac{\sum_{i,t}m_{i,t}\log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})}{\sum_{i,t}m_{i,t}}.$$

The mask refers to the **shifted target positions**, not simply the input
positions. A useful implementation check writes input tokens, next-token
targets, and the target mask on three aligned rows. The first response token
is predicted using the prompt; it must not be dropped accidentally.

Teacher forcing supplies earlier demonstrated response tokens as context.
At inference, the model sees its own earlier generated tokens. Lower training
loss therefore does not alone establish good free-running behavior.

### Worked mask and reduction example

Suppose one response has two scored tokens with target probabilities
$(1/2,1/4)$, and a second response has one scored token with probability $1/4$.
All prompt and padding targets are masked. The token mean is

$$L_{\mathrm{token}}=\frac{\log 2+\log 4+\log 4}{3}=\frac{5}{3}\log 2.$$

A mean of the two response means instead gives

$$L_{\mathrm{response}}=\frac{1}{2}\left(\frac{\log 2+\log 4}{2}+\log 4\right)=\frac{7}{4}\log 2.$$

Both reductions are definable; they weight examples differently. Name the
chosen convention. Include a termination token consistently, and distinguish
a genuine termination from a response cut at a length cap.

## 4. Chat formatting is part of the model input

The tokenizer and chat template determine the actual token sequence. System,
user, and assistant roles may introduce special tokens and separators. A model
trained with one serialization can behave differently when evaluated with another.

Inspect an actual encoded example before training. Check the assistant boundary,
the shifted mask, the termination token, and which text truncation removes.
Do not fit a new tokenizer on the evaluation set. Record the tokenizer and
template revisions alongside the model revision.

SFT data quality involves more than grammatical text. A response must follow
the instruction, use relevant evidence, and apply the intended refusal or
abstention policy. Duplicated templates, wrong answers presented confidently,
and inconsistent rubrics can all produce a low training loss.

## 5. Constructing preference comparisons

A pair $(x,y_w,y_l)$ says that response $y_w$ was preferred to $y_l$ for prompt
$x$ under the collection procedure. Useful metadata includes the source prompt,
candidate-generating model, sampling settings, comparison rubric, and whether
the annotators agreed. Preserve ties or disagreement rather than inventing
a confident winner when the collection protocol does not provide one.

Several pairs derived from the same prompt or candidate ranking are correlated.
Split by prompt family or source before deriving pairs. A random pair split can
place the same prompt and nearly identical answers in both training and test.

Inspect counterexamples to easy shortcuts. If preferred responses are almost
always longer, length alone can predict the labels without understanding the
task. Compare on a held-out set that separates answer quality from this cue.
An evaluation rubric should distinguish correctness, instruction following,
support, and style instead of assuming one scalar explains every disagreement.

## 6. Pairwise reward modeling

Let $r_\phi(x,y)$ be a scalar score. The Bradley–Terry model defines

$$P_\phi(y_w\succ y_l\mid x)=\sigma\bigl(r_\phi(x,y_w)-r_\phi(x,y_l)\bigr),\qquad\sigma(z)=\frac{1}{1+e^{-z}}.$$

For a dataset whose first response is the observed winner, minimize

$$L_{\mathrm{RM}}(\phi)=-\mathbb E\left[\log\sigma\bigl(r_\phi(x,y_w)-r_\phi(x,y_l)\bigr)\right].$$

Use a stable log-sigmoid or softplus implementation. A large negative margin
should produce a large finite loss, not overflow. Both winner and loser scores
participate in the derivative.

If $d=r_w-r_l$, then $dL/dd=\sigma(d)-1$. At $d=0$, this derivative is $-1/2$:
gradient descent pushes the margin in the observed preference direction.
This describes the pair's gradient contribution; shared parameters and other
pairs can affect the actual change in each individual score.

### Worked margin example

If $r_w=\log 3$ and $r_l=0$, the predicted preference probability is $3/4$
and the pair loss is $\log(4/3)$. Adding 10 to both scores changes neither.
More generally, a prompt-specific offset $c(x)$ cancels from every same-prompt
comparison. Pairwise training therefore does not identify an absolute reward
origin. Scores from separately fitted reward models are not automatically on
the same scale.

Reading: [InstructGPT, Section 3.2 and the reward-model loss](https://arxiv.org/html/2203.02155v1).

## 7. Evaluating a reward model

Held-out pair accuracy asks whether the sign of a score difference matches the
label. Pair log loss also uses the predicted probability. Neither establishes
that maximizing the score will produce good outputs outside the training
distribution. Check the effect of length, formatting, source, and prompt type;
retain representative disagreements for manual inspection.

For example, a scorer can rank ordinary responses reasonably well but assign
an unusually high score to an answer containing repeated praise. Optimization
may find that behavior even if it rarely appeared in the comparison dataset.
This is a distribution-shift problem between the candidates used to train the
reward model and the candidates produced by the optimized policy.

Keep evaluation prompts and their related variants out of SFT data, reward-model
training, policy-training prompts, and development-driven selection. A frozen
reward model can still leak evaluation information if its own training data
contains the evaluation cases.

## 8. Regularized policy optimization

With a fixed reward model and fixed reference policy, a common objective is

$$J(\theta)=\mathbb E_{x\sim D_x,y\sim\pi_\theta}[r_\phi(x,y)]-\beta\,\mathbb E_{x\sim D_x}D_{\mathrm{KL}}\bigl(\pi_\theta(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x)\bigr).$$

The reference is often the SFT policy. The regularizer discourages movement
away from that anchor, but does not certify truthfulness or make a misspecified
reward correct. Increasing reward alone is not enough to claim improvement.

For a complete response, the log-probability ratio is a sum over its tokens.
An individual sampled log ratio may be negative; the full KL expectation is
nonnegative. Do not call every sampled log ratio a nonnegative KL value.

In the original InstructGPT study, the PPO-ptx variant also mixes in a
pretraining-likelihood term. Record whether such an auxiliary objective is
present. “PPO” alone does not fully specify the training recipe.

## 9. Keep five computational roles separate

| Object | Purpose during a PPO iteration |
| --- | --- |
| Current policy | Receives the actor update |
| Old policy | Generated the current rollout batch; supplies fixed ratio denominators |
| Reference policy | Provides the persistent regularization anchor |
| Reward model | Scores completed responses |
| Value model or critic | Predicts expected remaining return from a prefix |

The old policy is refreshed between rollout batches. The reference usually
stays fixed across those batches. A reward model scores a completed response;
a critic estimates a policy-dependent expected continuation. They are not
interchangeable. Lecture 09 derives their different roles in the update.

## 10. A defensible before/after comparison

Use the same held-out prompts and declare the decoding budget. Report changes
in the target task together with regressions on a separate capability or
behavior slice. Record refusal/abstention criteria and inspect answers behind
the scores. If using human preference judgments, report the rubric and
population; if using an automated judge, validate it on checked examples.

Separate three claims: fitting demonstration text, predicting preference
labels, and improving generated behavior. Each requires a different comparison.
This distinction connects post-training to Lecture 10's data and evaluation
design. A successful optimization trace establishes that an objective changed;
the evaluation must establish whether that change is useful.

## Practice and reading

The companion notebook contains E01–E04: masked SFT loss, reward-margin
arithmetic, prompt-family split leakage, and the different roles of old and
reference policies. It runs offline with Python's standard library.

- [InstructGPT: methods, experiments, limitations, and annotation details](https://arxiv.org/html/2203.02155v1).
- [PPO: Sections 2–3 and Algorithm 1](https://arxiv.org/abs/1707.06347).
- [Course reading guide on post-training and agents](rl-posttraining-and-agents.md).
- [Lecture 09 notes](lecture-09-policy-optimization.html).
- [Lecture 10 notes](lecture-10-data-evaluation.html).
