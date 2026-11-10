<!-- .slide: class="title-slide" id="title" -->

<p class="eyebrow">Lecture 09 · November 4, 2026</p>

# Learning from rewards

<p class="subtitle">REINFORCE, advantages and PPO</p>

<p class="caption">GRPO and DPO follow as supplementary material.</p>

Note:
Reconstructed November 10 from the instructor's taught scope and the November 4 policy-optimization reference PDF. REINFORCE and PPO are the core. GRPO and DPO were not taught in Week 9; the supplement is prepared for later teaching and is not a Quiz 4 prerequisite. Use the companion notes for longer derivations.

---

<!-- .slide: id="objectives" -->

## Trace a policy update

- Derive a gradient without differentiating a sampled reward.
- Explain the role of a baseline.
- Compute PPO's objective for both advantage signs.
- Keep rollout, reference and current policies distinct.

Note:
The central question is how a sampled score changes next-token probabilities. Four short exercises run offline on CPU. The class uses gamma=1 for its language-model examples; explicit discount conventions are in the notes.

---

<!-- .slide: class="outline-slide" id="outline-gradient" -->

## Lecture outline

<ol class="outline-topics"><li aria-current="step">From sampled rewards to gradients</li><li>Advantages and clipped policy updates</li><li>A complete PPO iteration</li></ol>

Note:
Use the first period for REINFORCE and E01, the second for PPO and E02, and the third for GAE, implementation checks and E03–E04. The supplement has a separate teaching budget.

---

<!-- .slide: id="language-mdp" -->

## Next-token prediction becomes a policy

| Object | Language-model example |
| --- | --- |
| State $s_t$ | Prompt and generated prefix |
| Action $a_t$ | Next token |
| Transition | Append the token |
| Reward | Token feedback or terminal score |

Note:
Termination is EOS or a declared horizon. Padding is not an action. The reward model is one possible reward source; a verifier or human feedback can supply others. State the prompt distribution and decoding sampler.

---

<!-- .slide: id="objective" -->

## Optimize expected reward

$$J(\theta)=\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[R(x,y)]$$

$$\pi_\theta(y\mid x)=\prod_t\pi_\theta(y_t\mid x,y_{<t})$$

The reward is fixed in this derivation.

Note:
We suppress the outer expectation over prompts. The reward may be nondifferentiable with respect to the sampled text. A learned reward model is frozen during the policy update. Parameter-dependent KL penalties need their stated objective and estimator, not a silent application of this fixed-reward derivation.

---

<!-- .slide: id="score-function" -->

## Differentiate the sampling probability

$$\nabla_\theta p_\theta(y)=p_\theta(y)\nabla_\theta\log p_\theta(y)$$

$$\nabla J=\mathbb E_{\pi_\theta}[R(x,y)\nabla\log\pi_\theta(y\mid x)]$$

Increase the log probability of high-reward samples.

<p class="source">Williams (1992), REINFORCE</p>

Note:
Start with the finite sum J=sum_y p(y)R(y), differentiate, then multiply and divide by p. This assumes fixed support and differentiable positive probabilities. The expectation already samples according to the policy; multiplying a sampled contribution by its probability again is a mistake. Source: https://doi.org/10.1007/BF00992696.

---

<!-- .slide: id="token-scores" -->

## A response score distributes over its tokens

$$\nabla\log\pi_\theta(y\mid x)=\sum_t\nabla\log\pi_\theta(y_t\mid s_t)$$

With one terminal reward and $\gamma=1$, every token receives the same return.

That does not identify which token caused success.

Note:
The score-function estimator solves differentiation through sampling, not the credit-assignment problem. Token-level rewards and value estimates can improve the estimator. Sequence sums and token means also imply different length weighting.

---

<!-- .slide: id="returns" -->

## Use rewards from this action onward

$$G_t=\sum_{u=t}^{T-1}\gamma^{u-t}r_u$$

$$\nabla J=\mathbb E\left[\sum_t\gamma^tG_t\nabla\log\pi_\theta(a_t\mid s_t)\right]$$

For the worked language-model examples, set $\gamma=1$.

Note:
This formula corresponds to J=E[sum_t gamma^t r_t]. Past rewards have zero expected contribution to the current action's score, so they can be removed. Do not drop the outer gamma^t while claiming the same discounted trajectory objective. Indices start at zero in this formula.

---

<!-- .slide: id="baseline" -->

## An action-independent baseline cancels in expectation

$$\mathbb E_{a\sim\pi}[b(s)\nabla\log\pi(a\mid s)]=0$$

Subtract $b(s)$ from the return.

A suitable baseline can reduce variance without changing the expected gradient.

Note:
At a fixed state, factor out b(s) and use sum_a grad pi(a|s)=grad 1=0. Independence from the current sampled action matters. An arbitrary baseline need not reduce variance. A baseline fitted using the same actions can introduce dependencies; the supplement discusses the group-mean case.

---

<!-- .slide: id="reinforce-loss" -->

## A loss for the sampled update

$$L_{\mathrm{RF}}=-\frac1N\sum_{i,t}\operatorname{sg}[G_{i,t}-b(s_{i,t})]\log\pi_\theta(a_{i,t}\mid s_{i,t})$$

`sg` means stop-gradient. Minimize this loss.

Collect fresh samples as the policy changes.

Note:
Gamma is one here. At the rollout parameters this loss gives the desired negative score-function estimate. Treat rewards and advantages as fixed coefficients. A parameter-dependent weighted log likelihood with undetached weights generally has additional derivative terms. Reusing an uncorrected fixed batch for many updates no longer gives an on-policy estimator.

---

<!-- .slide: class="exercise" id="exercise-01" -->

<p class="exercise-meta">Exercise E01 · 6 minutes · Notebook E01</p>

## Check REINFORCE with two actions

$p=\sigma(\theta)=1/4$. Action 1 earns 1; action 0 earns 0.

Compute the expected gradient, with baseline 0 and then $1/2$.

<div class="answer fragment"><p>Both give $p(1-p)=3/16$.</p></div>

Note:
The action scores are 1-p and -p. Without baseline: (1/4)(1)(3/4)=3/16. With b=1/2: (1/4)(1/2)(3/4)+(3/4)(-1/2)(-1/4)=3/16. Ask students which term changes at the sample level. This exact enumeration checks the estimator, not a noisy training curve.

---

<!-- .slide: class="outline-slide" id="outline-ppo" -->

## Lecture outline

<ol class="outline-topics"><li>From sampled rewards to gradients</li><li aria-current="step">Advantages and clipped policy updates</li><li>A complete PPO iteration</li></ol>

Note:
Now address reuse of a rollout batch. An advantage estimates whether an action was better than expected in that state. PPO uses a local clipped surrogate; it is not a global guarantee on policy drift.

---

<!-- .slide: id="value-advantage" -->

## Advantage compares an action to its state's average

$$V^\pi(s)=\mathbb E[G_t\mid s_t=s]$$

$$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$$

Positive advantage favors increasing the sampled action's probability.

Note:
The scalar reward model scores a response according to preference data. The critic estimates expected future return under the policy. Those are different functions even if an implementation shares a backbone. A critic is not needed for the REINFORCE identity itself.

---

<!-- .slide: id="three-policies" -->

## Three policies have different jobs

| Policy | Purpose | During inner updates |
| --- | --- | --- |
| Current $\pi_\theta$ | Learn improved behavior | Changes |
| Old $\pi_{\mathrm{old}}$ | Generated this rollout | Fixed probabilities |
| Reference $\pi_{\mathrm{ref}}$ | Anchor behavior | Fixed anchor |

Note:
At the start of a rollout the current policy becomes the old sampler. The reference usually remains the SFT policy across rollouts. A live alias to the changing policy cannot serve as cached old log probabilities. If sampling uses temperature or truncation, the ratio's denominator must match the claimed sampling distribution.

---

<!-- .slide: id="importance-ratio" -->

## Measure the change on a sampled action

$$\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}$$

$$\rho_t=\exp(\log\pi_\theta-\log\pi_{\mathrm{old}})$$

At rollout collection, $\rho_t=1$.

Note:
For language-model PPO use a token ratio at each sampled prefix. The old state distribution remains fixed in this local surrogate. A single action ratio does not correct the distribution of whole trajectories, and multiplying all response ratios creates a different estimator. Compute via log probabilities for numerical stability.

---

<!-- .slide: id="clipped-objective" -->

## PPO clips the incentive to move further

$$J_{\mathrm{clip}}=\mathbb E_t\!\left[\min\left(\rho_t\hat A_t,\bar\rho_t\hat A_t\right)\right]$$

$$\bar\rho_t=\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)$$

Maximize this surrogate; minimize its negative.

<p class="source">Schulman et al. (2017), Equation 7</p>

Note:
Source: https://arxiv.org/abs/1707.06347. Old probabilities and estimated advantages are fixed during inner policy updates. The min matters: this is a pessimistic clipped surrogate. It does not clip the model probabilities themselves, impose a hard ratio bound, or guarantee monotonic improvement.

---

<!-- .slide: id="positive-advantage" -->

## Positive advantage: cap the gain from increasing probability

For $\hat A>0$, the objective stops increasing above $1+\epsilon$.

For $\rho<1-\epsilon$, decreasing probability still hurts.

<p class="caption">Clipping is directional because PPO takes a minimum.</p>

Note:
With A=2 and epsilon=.2, the objective is min(2rho,2.4). Below .8 it still equals 2rho. Students often incorrectly draw a flat segment at both ends; the next slide compares the signs.

---

<!-- .slide: id="clip-curves" -->

## The advantage sign changes the clipped side

<div class="plot" data-plotly="assets/ppo-clipping.json" role="img" aria-label="PPO surrogate versus probability ratio with epsilon 0.2. Advantage plus two flattens above 1.2; advantage minus two flattens below 0.8."></div>

<p class="caption">Analytical curves; maximize the vertical value.</p>

Note:
For negative A the left side flattens: with A=-2 the objective is min(-2rho,-1.6). Increasing an already overprobable bad action is still penalized. These are exact formula plots, not training measurements. The notebook checks all four cases in E02.

---

<!-- .slide: class="exercise" id="exercise-02" -->

<p class="exercise-meta">Exercise E02 · 7 minutes · Notebook E02</p>

## Which term does the minimum select?

Use $\epsilon=0.2$ and fill the clipped objective.

| Ratio | $\hat A=+2$ | $\hat A=-2$ |
| --- | --- | --- |
| 0.7 | ? | ? |
| 1.3 | ? | ? |

<div class="answer fragment"><p>At 0.7: $1.4, -1.6$. At 1.3: $2.4, -2.6$.</p></div>

Note:
Expected work compares raw and clipped products before taking the minimum. In particular min(-2.6,-2.4)=-2.6. State that these are objective values, not gradients; the active branch determines the derivative. At a kink the derivative needs an implementation convention.

---

<!-- .slide: id="clip-limits" -->

## Clipping is not a hard trust region

- Shared parameters can move many action probabilities.
- Several gradient steps can accumulate drift.
- A clipped sample can move through other samples' gradients.

Monitor KL, ratio tails and actual held-out behavior.

Note:
The KL monitor and stopping rule are separate implementation choices. Clip fraction reports how many ratios fall outside a band; it does not by itself show which clipped branch is active or whether the policy improved. Avoid claiming PPO has TRPO's theoretical constraint.

---

<!-- .slide: class="outline-slide" id="outline-loop" -->

## Lecture outline

<ol class="outline-topics"><li>From sampled rewards to gradients</li><li>Advantages and clipped policy updates</li><li aria-current="step">A complete PPO iteration</li></ol>

Note:
Connect the equations to a batch's lifetime. Work through GAE before presenting the full loop. The critic must respect true termination versus a time-limit truncation.

---

<!-- .slide: id="td-residual" -->

## A one-step prediction error

$$\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$$

At true termination, the next-state value is zero.

At an artificial cutoff, decide whether bootstrapping is appropriate.

Note:
The residual compares observed reward plus predicted next value against the current prediction. Do not bootstrap past a true EOS into padding. A truncation is not automatically a true terminal state. Values used to compute rollout targets are held fixed for the inner update batch.

---

<!-- .slide: id="gae" -->

## GAE combines residuals backward

$$\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$$

$$\hat V_t^{\mathrm{target}}=\hat A_t+V_{\mathrm{old}}(s_t)$$

$\lambda$ controls how far residuals propagate.

<p class="source">Schulman et al., GAE (2015/2016)</p>

Note:
Source: https://arxiv.org/abs/1506.02438. This is the finite-rollout recursion with the correct boundary treatment. Lambda zero uses one-step residuals; lambda one accumulates the full available return relative to V, with the chosen final bootstrap. Advantage normalization for policy updates should not silently redefine value targets.

---

<!-- .slide: class="exercise" id="exercise-03" -->

<p class="exercise-meta">Exercise E03 · 7 minutes · Notebook E03</p>

## Propagate the terminal reward

$r=[0,0,1]$, $V=[0.2,0.3,0.4]$; terminal value 0.

Use $\gamma=1$, $\lambda=0.9$.

<div class="answer fragment"><p>$\delta=[0.1,0.1,0.6]$<br>$\hat A=[0.676,0.64,0.6]$</p></div>

Note:
Expected backward calculation: .6; .1+.9*.6=.64; .1+.9*.64=.676. Value targets are [.876,.94,1]. For a follow-up set lambda=1 and recover advantages [.8,.7,.6] and value targets [1,1,1]. The notebook checks the specified .9 case.

---

<!-- .slide: id="reward-and-kl" -->

## Combine a task score with a reference penalty

$$\ell_t=\log\pi_{\mathrm{old}}(a_t\mid s_t)-\log\pi_{\mathrm{ref}}(a_t\mid s_t)$$

Use $-\beta\ell_t$ as sampled shaping feedback and add the terminal task score.

This sample can have either sign.

Note:
This slide describes a common rollout-reward construction; record the exact variant used. In expectation under the old policy at a fixed state, the log ratio is KL(old||reference). It is not a nonnegative divergence on every sampled token. Cache these rollout rewards while computing advantages. An implementation that instead adds a differentiable current-policy KL penalty is a distinct objective and must be specified.

---

<!-- .slide: id="ppo-loop" -->

## One PPO iteration has two phases

| Collect and freeze | Update and monitor |
| --- | --- |
| Prompts and sampled tokens | Current token log probabilities |
| Old and reference log probabilities | Clipped policy loss |
| Rewards and old values | Value prediction loss |
| Advantages and value targets | KL, entropy and ratio diagnostics |

Note:
Draw a boundary between rollout collection and inner updates. Freeze the old quantities even if actor and critic share layers. After a bounded number of minibatch epochs, collect new rollouts. Entropy bonuses, value clipping and KL controls vary across implementations; include their coefficients in the experimental record. Source: PPO Algorithm 1 and InstructGPT Section 3.3.

---

<!-- .slide: class="exercise" id="exercise-04" -->

<p class="exercise-meta">Exercise E04 · 5 minutes · Notebook E04</p>

## Find the overwritten denominator

Rollout probability: $0.20$. Current: $0.24$. Reference: $0.10$.

The code replaces the rollout probability with the current one.

<div class="answer fragment"><p>Correct ratio: $1.2$. Buggy ratio: $1$.</p></div>

Note:
Expected repair: retain detached rollout log probabilities, not a view that changes with model updates. The reference is not the PPO denominator; log(.24/.10)=log(2.4) describes a different comparison. Recompute new current probabilities in each update while keeping cached old probabilities fixed.

---

<!-- .slide: id="length-weighting" -->

## Reduction determines which responses get more weight

| Reduction | Consequence |
| --- | --- |
| Sum token losses, mean responses | Longer responses can contribute more |
| Mean tokens within each response | Equal response-level scale |
| Mean all valid tokens | Token-weighted batch objective |

Note:
Masks, EOS handling and reduction are part of the algorithm specification. None of these conventions is automatically interchangeable. In particular do not present a length-normalized surrogate as if it exactly equals an unnormalized sequence score-function estimator.

---

<!-- .slide: id="diagnostics" -->

## A reward curve is not an evaluation report

Track reward, reference KL, entropy, response length and clip statistics.

Compare held-out task behavior and failure slices.

Investigate reward gains that coincide with worse verified behavior.

Note:
Reward-model exploitation can increase the optimized proxy without improving the intended task. Lecture 10 explains prompt-family splits, independent evaluation, paired comparisons and uncertainty. These are requirements for a defensible claim, not another algorithm component.

---

<!-- .slide: id="core-summary" -->

## Four checks before trusting the update

1. The sampled distribution matches the claimed policy.
2. Advantages are fixed coefficients in the policy loss.
3. Old and reference policies have distinct roles.
4. Clipping and reduction match the written objective.

Note:
This ends the Week 9 taught core. Direct students to the lecture notes and E01–E04. The next slides prepare material not yet taught; omit them from a 135-minute reconstruction of the original class. No assessment coverage expands merely because these slides exist.

---

<!-- .slide: class="outline-slide" id="outline-supplement" -->

## Supplement: GRPO and DPO

- GRPO replaces a learned value baseline with group comparisons.
- DPO learns directly from offline preference pairs.
- Both require explicit data and evaluation assumptions.

<p class="caption">Prepared for later teaching; not taught in Week 9.</p>

Note:
Allow a separate 35–45 minute session for this supplement, including P01/P02. It is not an extension required for master's students and carries no bonus grading track. Sources: DeepSeekMath Section 4 and DPO Sections 3–4.

---

<!-- .slide: id="grpo-group" -->

## GRPO compares responses to the same prompt

Sample $G$ responses and score each one.

$$\hat A_i=\frac{r_i-\bar r}{\sigma_r+\varepsilon}$$

Reuse that relative score across response $i$'s tokens.

<p class="source">DeepSeekMath (2024), Section 4</p>

Note:
Source: https://arxiv.org/html/2402.03300v3. This is outcome-supervised group normalization; state whether std uses the population convention and how zero variance is handled. It removes a learned critic, not the need for a reward source. Original formulations and later implementations differ in normalization and loss weighting.

---

<!-- .slide: class="exercise" id="practice-01" -->

<p class="exercise-meta">Practice P01 · optional · Notebook P01</p>

## What does a group reveal?

Rewards: $[0,0,1,1]$. Use population standard deviation.

Then consider $[1,1,1,1]$.

<div class="answer fragment"><p>First group: $[-1,-1,1,1]$. Constant group: zero relative signal.</p></div>

Note:
Allow five minutes in the separate supplement. Mean=.5 and population std=.5. With an epsilon in the denominator, nonzero magnitudes are slightly below one; the notebook takes the exact nonzero-std calculation and explicitly returns zero for a constant group. Zero does not imply the responses are bad; it means there is no within-group distinction.

---

<!-- .slide: id="grpo-surrogate" -->

## GRPO retains a clipped token surrogate

$$s_{i,t}=\min(\rho_{i,t}\hat A_i,\bar\rho_{i,t}\hat A_i)$$

$$J=\frac1G\sum_i\frac1{T_i}\sum_t(s_{i,t}-\beta k_{i,t})$$

No learned value function appears in this version.

Note:
This shows per-response token averaging, matching the stated teaching version. Cache old probabilities and group advantages. k denotes a specified reference penalty; do not quietly substitute a different KL estimator or reduction. Responses must be generated under the declared rollout sampler. This is not a full-response importance ratio.

---

<!-- .slide: id="grpo-kl" -->

## A nonnegative sampled KL expression

Let $u=\pi_{\mathrm{ref}}(a\mid s)/\pi_\theta(a\mid s)$.

$$k=u-\log u-1\geq0$$

Under current-policy sampling, its expectation is $D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})$.

Note:
This identity assumes shared positive support at a fixed state: E_current[u]=1. A batch sampled from the old policy is not generally a current-policy sample. Differentiating the expression on a fixed batch does not automatically include the sampling-distribution derivative of an exact KL expectation. Present it as the algorithm's specified surrogate estimator, not an unconditional equality of gradients.

---

<!-- .slide: id="group-cautions" -->

## Group normalization changes the estimator

- A response contributes to its own group mean.
- Standardization changes the relative weight of prompts.
- Constant groups supply no relative learning signal.
- Length reduction changes response weighting.

Note:
For independent responses, subtracting the self-including mean scales the unnormalized expected score gradient by (G-1)/G. Dividing by a random group std is an additional change. A leave-one-out baseline removes the self-inclusion term but does not make every normalization unbiased. Do not describe GRPO as simply an unbiased REINFORCE estimator with lower memory.

---

<!-- .slide: id="dpo-optimum" -->

## DPO starts from a KL-regularized optimum

$$\pi^*(y\mid x)\propto\pi_{\mathrm{ref}}(y\mid x)e^{r(x,y)/\beta}$$

$$r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+C(x)$$

<p class="source">Rafailov et al. (2023), Sections 3–4</p>

Note:
Source: https://arxiv.org/html/2305.18290v3. This is the ideal unrestricted optimum for beta>0 and reference support; it is not a guarantee that a finite neural model finds that optimum. Derive by a Lagrange multiplier or completing a KL. The prompt-only normalization term cancels when comparing responses for the same prompt.

---

<!-- .slide: id="dpo-margin" -->

## Compare policy gaps relative to a fixed reference

$$m_\theta=\log\pi_\theta(y_w\mid x)-\log\pi_\theta(y_l\mid x)$$

$$L_{\mathrm{DPO}}=-\log\sigma\!\left(\beta(m_\theta-m_{\mathrm{ref}})\right)$$

Use full response log-probability sums under the stated mask.

Note:
The Bradley–Terry model supplies the preference likelihood. Training uses offline chosen/rejected pairs; the basic DPO loop does not sample new online rollouts or fit a separate reward model. This trades one procedure for assumptions about the preference data and reference. Length-normalizing these sums changes this objective.

---

<!-- .slide: class="exercise" id="practice-02" -->

<p class="exercise-meta">Practice P02 · optional · Notebook P02</p>

## Calculate the DPO logit

Policy log probabilities: $(-2,-3)$ for chosen/rejected.

Reference: $(-2.5,-3)$; $\beta=0.2$.

<div class="answer fragment"><p>$z=0.1$, $L=-\log\sigma(0.1)$.</p></div>

Note:
Allow six minutes. The policy gap is 1 and reference gap .5, hence z=.2*(1-.5)=.1. The derivative with respect to the policy gap is -beta*sigmoid(-z), about -.0950. The notebook verifies this by finite differences. DPO differentiates the margin; do not detach it as if it were a precomputed REINFORCE advantage.

---

<!-- .slide: id="method-comparison" -->

## Match the method to its feedback

| Method | Feedback | Main extra object |
| --- | --- | --- |
| SFT | Demonstrated response | Target mask |
| REINFORCE | Sampled return | Optional baseline |
| PPO | Rollout advantage | Old probabilities, often critic |
| GRPO | Within-prompt reward group | Group statistics |
| DPO | Offline preference pair | Fixed reference likelihoods |

Note:
This is a teaching comparison, not a performance ranking. PPO can use advantages from different estimators; a learned critic is common rather than logically mandatory. GRPO can use learned or rule-based rewards. More advanced variants belong in subsequent reading. All methods need data provenance and evaluation of the intended behavior.

---

<!-- .slide: class="references" id="references" -->

## Read the objectives, then the implementation

- [REINFORCE: Williams (1992)](https://doi.org/10.1007/BF00992696)
- [PPO: Sections 3 and 5](https://arxiv.org/abs/1707.06347)
- [GAE: residuals and the advantage estimator](https://arxiv.org/abs/1506.02438)
- [GRPO: DeepSeekMath, Section 4](https://arxiv.org/abs/2402.03300)
- [DPO: Sections 3–4](https://arxiv.org/abs/2305.18290)

Note:
The instructor's supplied November 4, 24-page policy-optimization PDF guided the derivation order. The companion course notes state assumptions and worked calculations. No benchmark ranking or hardware allocation follows from citing a paper. Source review: November 10, 2026.
