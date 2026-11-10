# Lecture 09: Reinforcement learning and policy optimization

Week 9, November 4, 2026. Sections 1–8 consolidate REINFORCE and PPO.
Sections 9–11 prepare GRPO and DPO as supplementary material: they were not
covered in the Week 9 class and are not prerequisites for Quiz 4.

These notes adapt the instructor's November 4 reference, *Policy optimization
for LLMs: REINFORCE, PPO, GRPO and DPO*, with consistent notation and small
checked examples. We maximize $J$ and minimize $L=-J$. Logarithms are natural.

## 1. An autoregressive policy as a decision process

A prompt $x$ is sampled from a specified distribution. At step $t$, the state
is $s_t=(x,y_{<t})$, the action is the next token $y_t$, and the transition
appends it to the prefix. The response ends at EOS or a declared horizon.
Padding is not an action. The model is a policy $\pi_\theta(y_t\mid s_t)$.

For a trajectory $\tau$ of $T$ actions indexed from zero, define

$$R(\tau)=\sum_{t=0}^{T-1}\gamma^t r_t,\qquad G_t=\sum_{u=t}^{T-1}\gamma^{u-t}r_u,\qquad J(\theta)=\mathbb E_{\tau\sim p_\theta}[R(\tau)].$$

The initial state distribution, environment transitions, and reward function
are fixed in this derivation. For a language-model example with one terminal
score and $\gamma=1$, every response token has the same return $G_t=R(x,y)$.
Equivalently, the entire response can be viewed as one contextual-bandit action.
This equivalence does not imply that PPO should use a full-response ratio.

The value $V^\pi(s)=\mathbb E[G_t\mid s_t=s]$ averages future outcomes under
the policy. The action value $Q^\pi(s,a)$ additionally conditions on the next
action. The advantage is $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$.
The terminal state's value is zero when it is truly terminal.

## 2. The score-function identity

Differentiating an observed reward number does not capture how the policy
changes the probability of observing that outcome. Instead, use

$$\nabla_\theta p_\theta(\tau)=p_\theta(\tau)\nabla_\theta\log p_\theta(\tau).$$

For differentiable positive probabilities on fixed support, this gives

$$\nabla_\theta J=\sum_\tau R(\tau)\nabla_\theta p_\theta(\tau)=\mathbb E_{p_\theta}[R(\tau)\nabla_\theta\log p_\theta(\tau)].$$

Only the policy factors depend on $\theta$ under the stated assumptions, so

$$\nabla_\theta\log p_\theta(\tau)=\sum_t\nabla_\theta\log\pi_\theta(y_t\mid s_t).$$

The probability already determines the sampling distribution. Do not multiply
each Monte Carlo sample by its policy probability a second time.

Past rewards have zero expected contribution to a later action's score.
Removing them gives the return-to-go form

$$\nabla_\theta J=\mathbb E\left[\sum_t\gamma^tG_t\nabla_\theta\log\pi_\theta(y_t\mid s_t)\right].$$

The outer $\gamma^t$ belongs to this discounted trajectory-sum convention.
For our language-model examples we set $\gamma=1$ explicitly. This avoids
silently dropping a factor while claiming to optimize a discounted objective.

## 3. Baselines and REINFORCE

A baseline $b(s_t)$ that does not depend on the current sampled action has
zero expected score contribution:

$$\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}[b(s)\nabla_\theta\log\pi_\theta(a\mid s)]=b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)=0.$$

Thus $G_t-b(s_t)$ can replace $G_t$ without changing the expected gradient.
A well-chosen baseline can reduce variance; an arbitrary one need not.
Bootstrapping with an approximate critic can introduce a separate estimation bias.

For trajectories generated at $\theta_0$, construct the local loss

$$L_{\mathrm{RF}}(\theta)=-\frac{1}{N}\sum_{i,t}\gamma^t\operatorname{sg}[G_{i,t}-b(s_{i,t})]\log\pi_\theta(y_{i,t}\mid s_{i,t}).$$

Here $\operatorname{sg}$ means stop-gradient. At $\theta=\theta_0$, the
negative gradient estimates the desired policy gradient. Reusing the same
unweighted batch for many updates does not keep it on-policy as $\theta$ changes.

### Binary-action check

Let $p=\sigma(\theta)$ be the chance of action 1, with rewards 1 and 0 for
actions 1 and 0. Then $J=p$ and $dJ/d\theta=p(1-p)$.
The scores of the two actions are $1-p$ and $-p$ respectively. At $p=1/4$,

$$\mathbb E[R\,\nabla\log\pi]=\frac14\left(1\cdot\frac34\right)+\frac34\left(0\cdot-\frac14\right)=\frac{3}{16}.$$

Subtracting a fixed baseline leaves this mean unchanged. The sample-level
gradient contributions can change substantially, which is how a baseline
can change variance while preserving the expected update.

Reading: [Williams (1992)](https://doi.org/10.1007/BF00992696).

## 4. The old policy and the reference policy

The **old policy** generated the current rollout batch. Its cached log
probabilities stay fixed during inner updates. The **reference policy** is a
persistent anchor, often the SFT model, used to penalize unwanted drift.
Refreshing the old policy does not mean replacing the reference.

At a fixed sampled state and action, define

$$\rho_t(\theta)=\frac{\pi_\theta(y_t\mid s_t)}{\pi_{\mathrm{old}}(y_t\mid s_t)}=\exp\left(\ell_t(\theta)-\ell_t^{\mathrm{old}}\right).$$

This corrects an action expectation at a fixed state. Correcting an entire
trajectory distribution would generally require a product of ratios. PPO uses
a local surrogate based on old-policy state visitation and token-level ratios;
it does not make a full new-policy trajectory expectation exact.

The denominator must describe the distribution that actually generated the
tokens. Temperature changes or top-p filtering alter that distribution.
For the derivations here, sample directly from the stated policy without such
transformations. Otherwise account for them explicitly and check support.

## 5. PPO's clipped surrogate

For a fixed estimated advantage $\hat A_t$ and $0<\epsilon<1$, PPO maximizes

$$J_{\mathrm{clip}}(\theta)=\widehat{\mathbb E}_{\mathrm{old}}\left[\min\left(\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right].$$

The minimum picks the less optimistic contribution. The clipping behavior
depends on the **sign** of the advantage:

| Advantage | Region where this sample's policy term is flat | Region still receiving a gradient |
| --- | --- | --- |
| Positive | Ratio above $1+\epsilon$ | Ratios below that upper threshold |
| Negative | Ratio below $1-\epsilon$ | Ratios above that lower threshold |
| Zero | Everywhere | None from this sample's reward advantage |

With $\epsilon=0.2$, consider four synthetic cases:

| Advantage | Ratio | Unclipped value | Clipped objective contribution |
| ---: | ---: | ---: | ---: |
| $+2$ | 1.3 | 2.6 | 2.4 |
| $+2$ | 0.7 | 1.4 | 1.4 |
| $-2$ | 0.7 | -1.4 | -1.6 |
| $-2$ | 1.3 | -2.6 | -2.6 |

The first and third cases are flat with respect to their ratio away from the
kink. Other samples or auxiliary losses can still move shared parameters.
PPO does not force all probability ratios into the clipping interval and does
not impose a hard KL bound. Monitor actual policy changes separately.

Reading: [Schulman et al. (2017), Section 3 and Algorithm 1](https://arxiv.org/abs/1707.06347).

## 6. Returns, the critic, and GAE

A fixed rollout-time critic produces $V_t$. The temporal-difference residual is

$$\delta_t=r_t+\gamma V_{t+1}-V_t.$$

Generalized advantage estimation uses

$$\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1},\qquad\hat A_T=0.$$

For a true terminal episode, set $V_T=0$. A segment truncated from a continuing
episode needs an appropriate bootstrap; do not treat every batch boundary as
a terminal event. At $\lambda=0$, the estimate is one residual. At $\lambda=1$
with a true terminal state, it is the Monte Carlo return minus the value.

For rewards $[0,0,1]$, values $[0.2,0.3,0.4]$, $\gamma=1$, and $\lambda=0.9$,
the residuals are $[0.1,0.1,0.6]$. Backward recursion gives advantages
$[0.676,0.64,0.6]$ and value targets $\hat A_t+V_t=[0.876,0.94,1]$.
At $\lambda=1$, the advantages become $[0.8,0.7,0.6]$.

Reading: [Generalized Advantage Estimation, Section 3](https://arxiv.org/abs/1506.02438).

## 7. A PPO iteration for a language model

1. Freeze the rollout snapshot and collect responses. Cache old-policy and
   reference log probabilities, rollout-time critic values, and completion masks.
2. Score responses with a fixed reward model or verifier. In one common RLHF
   construction, add token rewards
   $-\beta(\ell_t^{\mathrm{old}}-\ell_t^{\mathrm{ref}})$ and place the response
   score at the final action.
3. Compute detached advantages and value targets, with explicit EOS and
   truncation handling. Declare whether the loss averages tokens or responses.
4. Recompute current-policy log probabilities on the batch. Minimize the
   negative clipped surrogate and a critic regression loss, with any entropy
   term documented separately. Keep cached denominators and targets fixed.
5. After the chosen inner epochs, collect a new rollout batch. Keep the
   reference anchor fixed in this standard setup.

At the first update, the actor equals the rollout snapshot and ratios are one.
Later updates on that same batch activate the ratio and clipping behavior.
Detach old log probabilities and actor advantages, but keep gradients through
the current log probabilities and the critic predictions used by its own loss.

Check reward, held-out task performance, length, reference KL, entropy, clip
fraction, and critic error. Reward growth alone does not establish improvement.

## 8. Three useful implementation distinctions

**Objective versus gradient contribution.** A detached weighted log-probability
loss can reproduce a first-order gradient at a particular point. It is not
generally identical to the original objective over later updates. In ordinary
PPO, differentiate the minimum and clipping operations themselves.

**Sequence sum versus token mean.** For terminal reward and $\gamma=1$, a
sequence score is a sum of token scores. Dividing each response by its length
changes its relative weight when response lengths differ. Padding should never
change either declared reduction.

**Value estimate versus reward model.** The critic predicts expected remaining
return under a policy and reward convention. The reward model predicts a score
for a candidate response. Replacing one with the other changes the algorithm.

## 9. Supplement: GRPO and within-prompt groups

This material was not taught in Week 9. Outcome-based GRPO draws $G\ge2$
responses for one prompt and standardizes their rewards:

$$\bar R=\frac1G\sum_iR_i,\quad s_R=\sqrt{\frac1G\sum_i(R_i-\bar R)^2},\quad\hat A_i=\frac{R_i-\bar R}{s_R+\varepsilon_{\mathrm{num}}}.$$

We use the population standard deviation in this example. Other conventions
change the scale. Every response token receives its response's group advantage,
but each token retains its own policy ratio. The method removes the learned
value critic; it does not remove reward evaluation or the rollout cost.

Rewards $[0,0,1,1]$ give advantages approximately $[-1,-1,1,1]$. If all rewards
agree, the relative reward signal is zero. A separate regularizer can still
contribute a gradient. This makes group size and reward difficulty important.

The original outcome-based GRPO surrogate uses a mean across responses and a
token mean within each response, plus a sampled reference penalty. Define
$u=\pi_{\mathrm{ref}}(y_t\mid s_t)/\pi_\theta(y_t\mid s_t)$ and
$k=u-\log u-1$. Then a version matching that convention is

$$J_{\mathrm{GRPO}}=\mathbb E_{\mathrm{old}}\left[\frac1G\sum_i\frac1{T_i}\sum_t\left(C(\rho_{i,t},\hat A_i)-\beta k_{i,t}\right)\right],$$

where $C$ is the PPO clipped contribution. Response lengths and group statistics
are fixed during an inner update. Name the precise variant when implementing
GRPO; later implementations can change normalization, clipping, and KL treatment.

The group mean includes the current sample. It is therefore correlated with
that sample's score. Even without standardization, the centered group estimator
has expectation $(G-1)/G$ times the raw reward gradient under independent draws.
A leave-one-out mean avoids that particular correlation. Standardization,
clipping, and length weighting introduce further differences: do not claim that
the complete GRPO update is an unbiased REINFORCE estimator.

Also distinguish the KL estimator's value from its gradient. Under actions
drawn from the **current** policy, $\mathbb E[k]=D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})$.
Differentiating $k$ on a **fixed old-policy batch** gives a surrogate gradient;
it does not include the score term from changing the sampling distribution.

Reading: [DeepSeekMath, Section 4.1 and Appendix A](https://arxiv.org/html/2402.03300v3).

## 10. Supplement: direct preference optimization

DPO fits fixed preference pairs directly. It does not require an explicit
reward-model fit or fresh policy rollouts during standard offline training.
Its motivation starts with a KL-regularized response distribution at a fixed
prompt. For a known reward and positive reference support,

$$\pi^*(y\mid x)=\frac{\pi_{\mathrm{ref}}(y\mid x)\exp(r(x,y)/\beta)}{Z(x)}.$$

Rearrange to obtain
$r(x,y)=\beta\log(\pi^*(y\mid x)/\pi_{\mathrm{ref}}(y\mid x))+\beta\log Z(x)$.
The prompt-only partition term cancels in a same-prompt reward difference.
Substituting into the Bradley–Terry likelihood motivates the margin

$$\Delta_\theta=\beta\left[\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right],\qquad L_{\mathrm{DPO}}=-\mathbb E_D\log\sigma(\Delta_\theta).$$

Use sums of response-token log probabilities, with a consistent EOS convention
and no prompt or padding targets. Length-normalizing each sequence produces a
different objective. Keep the reference fixed and retain gradients through the
policy margin. Compute the log-sigmoid stably.

For a margin of $\log 3$, the predicted preference is $3/4$ and the corrective
coefficient for ascent is $\beta/4$. At a margin of $-\log 3$, that coefficient
is $3\beta/4$. Incorrectly ranked pairs receive stronger corrective pressure.

The ideal optimum derivation does not say that finite-data DPO and PPO have
identical trajectories or outcomes. DPO's gradient is a preference-likelihood
gradient over a fixed dataset. A log-probability derivative alone does not
identify an algorithm as reinforcement learning: SFT also has one.

Reading: [Rafailov et al., Section 4](https://arxiv.org/html/2305.18290v3).

## 11. Comparing the learning problems

| Method | Data used for an update | Feedback | Main qualification |
| --- | --- | --- | --- |
| SFT | Fixed demonstrations | Target response tokens | Learns demonstrated behavior |
| REINFORCE | Current-policy samples | Return, optional baseline | Monte Carlo policy-gradient estimator |
| PPO | Fixed batch from a rollout snapshot | Detached advantages | Local clipped surrogate; monitor policy drift |
| GRPO | Groups from a rollout snapshot | Relative group scores | Variant and normalization matter |
| DPO | Fixed preference pairs | Winner/loser labels | Direct preference likelihood |

The companion notebook checks REINFORCE arithmetic, PPO's sign-dependent
clipping, GAE recursion, and old/reference roles. Optional P01–P02 check GRPO
group statistics and DPO margins. All examples run offline on CPU and are
ungraded. The [post-training reading guide](rl-posttraining-and-agents.md)
offers longer courses and implementation references.
