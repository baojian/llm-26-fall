# Lecture 08 teaching plan

InstructGPT, SFT and reward models · October 28, 2026.
Reconstructed on November 10 to match the instructor's account of the class.
The notes consolidate the taught content; they are not a transcript.

Central question: how do demonstrations and preferences become different training signals?

| Minutes | Route | Purpose |
| --- | --- | --- |
| 0–45 | `title` through `sft-limit` | Workflow, response factorization, shifted masks, reduction; E01 (6 min). |
| 45–90 | `outline-reward` through `reward-evaluation` | Pairwise preferences, reward loss, shortcut risks; E02 (5 min), E03 (6 min). |
| 90–135 | `outline-policy` through `references` | KL anchoring, model roles, PPO-ptx, evidence; E04 (5 min). |

Breaks are outside these 135 teaching minutes. Timing is a planning estimate.
The four ungraded exercises have worked answers in slide notes and the notebook.
All use offline standard-library Python on CPU (0 GPU-hours).
E01: token mean 5/3 log(2), response mean 7/4 log(2).
E02: probability 3/4, loss log(4/3), common offsets cancel.
E03: keep paraphrases from a source family in one split; a single test family is insufficient evidence.
E04: current/old = 1.2; current/reference log ratio = log(2.4).

Use [the notes](../../docs/lecture-08-instructgpt.html) for derivations and
[the reading guide](optional-reading.md) for primary sources.
Do not interpret higher reward-model score as sufficient evidence of improved human utility.

Verification: run `npm --prefix slides run check -- lecture-08` and execute every
notebook code cell from a fresh working directory. Use `npm --prefix slides run pdf -- lecture-08`
for a locally reviewed PDF. Generated review artifacts belong in `slides/.checks/`.
