# Lecture 09 teaching plan

Reinforcement learning and policy optimization · November 4, 2026.
Reconstructed November 10 to match the instructor's account; not a transcript.

The core ends at `core-summary`. GRPO and DPO were **not taught in Week 9**.
The material after `outline-supplement` prepares a later 35–45 minute session;
it is not a prerequisite for Quiz 4. Timing estimates have not been rehearsed.

| Teaching minutes | Route | Student check |
| --- | --- | --- |
| 0–45 | `title`–`exercise-01` | REINFORCE identity and baseline, E01 (6 min). |
| 45–90 | `outline-ppo`–`clip-limits` | Values, old/reference distinction and clipping, E02 (7 min). |
| 90–135 | `outline-loop`–`core-summary` | GAE and PPO batch lifetime, E03 (7 min), E04 (5 min). |
| Separate supplement | `outline-supplement`–`references` | GRPO group calculation P01 (5 min); DPO margin P02 (6 min). |

Expected answers: E01 3/16 with either baseline; E02 objective values 1.4/-1.6
at ratio .7 and 2.4/-2.6 at ratio 1.3; E03 advantages [.676,.64,.6],
value targets [.876,.94,1]; E04 correct ratio 1.2, overwritten ratio 1.
P01 gives [-1,-1,1,1] with exact nonzero population std; a constant group has
zero relative signal. P02 gives logit .1 and derivative -0.2 sigmoid(-.1).
All examples are synthetic, ungraded and offline standard-library Python (0 GPU-hours).

Use [the full notes](../../docs/lecture-09-policy-optimization.html) for discount
conventions, the group-baseline caveat, and the distinction between a fixed-batch
KL surrogate and a current-policy expectation. The supplied 24-page November 4
policy-optimization PDF is the instructor reference, not a student download dependency.

Run `npm --prefix slides run check -- lecture-09` and execute every notebook cell.
Export a reviewed PDF with `npm --prefix slides run pdf -- lecture-09`.
