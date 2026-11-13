# Assignment 03: SFT, Reward Modeling, and PPO

**Release version: November 13, 2026.** Due:
**Friday, December 4, 2026, 23:59 Asia/Shanghai (UTC+08:00)**, privately through
eLearning. Weight: **15% of the course**. Total: **100 points**, with identical
work and scoring for all students.

Build and investigate this pipeline:

**pretrained GPT-2 → supervised fine-tuning (SFT) → reward model → PPO policy**.

The central question is: **does optimizing a learned reward improve the behavior
we independently measure?** Higher reward is not sufficient evidence. Correct,
well-documented negative results can receive full credit. No minimum accuracy
or improvement threshold is required.

## 1. Setup and supplied materials

Use Python 3.11 and the included lockfile:

```sh
uv sync --locked
uv run python -m pytest -q
```

The initial public tests fail until you implement the functions in `a3.py`.
Do not replace the objectives with a library Trainer or PPO implementation.
The supplied `run.py` handles loading, batches, sampling, scalar heads,
optimization, saving, and evaluation. Your work is the eight functions in
`a3.py`, the experiment, and your analysis. You do not need your A2 model.

The baseline is [GPT-2](https://huggingface.co/openai-community/gpt2), pinned to
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. The same checkpoint and tokenizer
initialize every stage. Remote model code is disabled. The first run downloads
model files; later runs reuse the local cache. Keep weights and outputs out of
Git. The model card supplies its MIT license and intended-use limitations.

Run the main recipe on a GPU compatible with the supplied PyTorch/CUDA
environment. The main recipe fine-tunes
all 124,439,808 GPT-2 backbone parameters, including the tied embedding/output
matrix. Reward and value models add scalar heads; the value head shares the
policy backbone. Parameters and objectives use float32, with TF32 matrix
multiplication enabled on CUDA. The runner batches computation and caches
generation keys/values. Dropout is disabled during both collection and updates;
`eval()` leaves gradient computation available.

The separate CPU smoke configuration trains only the last Transformer block
and final normalization for quick wiring checks. It is not an alternative
required experiment or a separate grading track. Everyone uses the same main
recipe and rubric.

## 2. Data and evaluation boundary

The supplied **grounded-lookup-v3** dataset asks questions about fictional
records of four to six items. Each item lists a subset of its color, shape,
material, and size. Return exactly one attribute value, or `unknown` when the
item or requested attribute is absent. For example, a record stating
`Item 4821: color=blue, material=wood.` supports `blue` for its color and
`unknown` for its shape. No external knowledge is needed.

| Split | Independent source families | Prompt/preference examples |
| --- | ---: | ---: |
| Train | 1,024 | 2,048 |
| Development | 128 | 256 |
| Test | 128 | 256 |

Each family contains two question phrasings and shuffled renderings of the same
record, kept in one split. Query attributes are balanced. 75% of queries have
an answer; 12.5% ask about a missing item and 12.5% about a missing attribute.
The 32 attribute values and `unknown` are listed in the data manifest.

Preferred responses are independently derived from the record. Rejected
responses include wrong item values, unnecessary abstention, and unsupported
answers. Both sides use the same answer format; `unknown` occurs on each side
in exactly 25% of pairs. Item identifiers share the same random range in all
conditions. These controls reduce simple shortcuts; they do not establish
that a learned reward model reasons correctly.

All data is course-authored, deterministic, and CC0-1.0. The preferences are
**synthetic correctness judgments, not human feedback**. Facts and IDs are
fictional. Read the supplied data README and checksum manifest.

Use only training families for SFT, reward fitting, and PPO prompts. PPO sees
the prompt and learned reward; it must not use the target checker as its
training reward. Development examples support debugging. Freeze the recipe
before final test evaluation, and never choose seeds/checkpoints from test
performance. Public test labels rely on this protocol boundary; they are not
a secret benchmark. This evaluates new records within a shared task and
vocabulary, not broad conversational ability or out-of-domain reasoning.

An always-unknown predictor obtains 25% accuracy on each complete split.
Use the present-query slice to assess evidence of successful lookup. Valid-label
rate checks membership in the 33-answer vocabulary; a valid but wrong attribute
value is still incorrect. Response-token counts include a generated EOS.

## 3. Required work and points

| Part | Work | Automatic | TA | Total |
| --- | --- | ---: | ---: | ---: |
| 1 | Response-only SFT | 20 | 5 | 25 |
| 2 | Reward model and paired evaluation | 20 | 5 | 25 |
| 3 | PPO and policy bookkeeping | 20 | 10 | 30 |
| 4 | Controlled experiment and interpretation | 0 | 20 | 20 |
| **Total** | | **60** | **40** | **100** |

### Part 1: SFT (25)

Implement `response_labels` and `sft_loss`. Labels align with the unshifted
input IDs; prompt targets and padding are `-100`. Include the response EOS.
Shift logits and labels exactly once inside the loss. Average over valid
response targets across the batch, not over padded length or sequence means.
If no response targets remain, raise `ValueError`.

Explain which input position predicts the first response token and why prompt
context remains available when prompt targets are ignored. GPT-2 uses EOS as
the padding ID here: use the attention mask to distinguish the two roles.

Run the provided SFT stage. Inspect the curve and compare the baseline and SFT
outputs under the same greedy decoding and token cap.

### Part 2: reward model and evaluation (25)

Implement `terminal_scores`, `preference_loss`, and `paired_summary`.

The reward model starts from the SFT backbone, with a new scalar head. Pool
at the final real token, including the appended EOS. Use the pairwise loss

`mean(-log(sigmoid(score_chosen - score_rejected)))`.

Implement it stably for large margins. A common shift to both scores should
not change the pair loss. Compare held-out pair accuracy with 50% chance;
report ties separately. The reward model is frozen before PPO.

The paired evaluator joins records by ID, checks family/target agreement,
and compares normalized whole answers (case and surrounding whitespace only).
Do not count a correct word embedded in a wrong sentence as exact match.
Bootstrap source-family mean accuracy differences with the supplied seed and
1,000 resamples. Follow the function's percentile convention. These intervals
reflect the small sampled test-family set, not training-seed uncertainty.

Explain one way reward scores could improve while exact correctness does not.
Inspect at least one reward-model error, or state explicitly if the evaluated
set contains none and discuss a remaining limitation.

### Part 3: PPO (30)

Implement `shaped_rewards`, `gae`, and `ppo_loss` using their docstrings.

Keep these four roles distinct:

| Object | Initialization | During PPO |
| --- | --- | --- |
| Current policy | SFT checkpoint | Updated |
| Old/behavior policy | Policy that sampled the current rollout batch | Store its detached log probabilities once per batch |
| Reference policy | Completed SFT checkpoint | Fixed for the whole PPO stage |
| Reward model | Trained preference model | Fixed for the whole PPO stage |

The value head is trained jointly with the current policy. It estimates value
at the state **before** each response action. At a sampled token, use reward
`-beta * (old_logp - reference_logp)`; add the reward-model sequence score only
at the final valid response token. This is a fixed-rollout sampled estimate:
an individual log ratio need not be positive.

Use gamma 1 and lambda 0.95 for GAE. Both EOS and reaching the declared response
cap end the episode in this exercise, with zero next-state value. This finite
horizon is deliberate; do not bootstrap beyond it. Exclude padded actions
from rewards, advantages, policy loss, and value loss. Detach old probabilities,
advantages, and value targets. No advantage normalization is used.

Minimize the negative clipped policy objective with epsilon 0.2, plus 0.5 times
the value loss; the value loss itself is `0.5 * mean((V - return)**2)`.
Both positive and negative advantages must be handled correctly. No value
clipping, entropy bonus, or pretraining-loss mixture is required.

The runner samples the **full** categorical distribution at temperature 1;
there is no top-k/top-p transform that would make the stored behavior
probabilities inconsistent. It performs two updates on each rollout batch
before collecting the next. Verify that old probabilities are unchanged
across those two updates and that the reference/reward parameters stay fixed.
The two updates are not two independent rollout batches.

### Part 4: experiment and interpretation (20)

Run the supplied fixed configuration once (seed 17), then evaluate baseline,
SFT, and PPO on the same dev/test prompts. Do not search for the best seed.
Include these results in a concise report (at most 1,500 words, excluding
references and tables):

1. One table with baseline/SFT/PPO exact accuracy, valid-label rate, mean reward
   score, and mean generated length. Include the present/absent slices.
2. Reward-model pair accuracy and at least one inspected failure or limitation.
3. Baseline→SFT and SFT→PPO paired differences and family bootstrap intervals.
   Acknowledge the small sample and single training seed.
4. Inspect two example IDs: an improvement and a regression when present;
   otherwise show two informative unchanged/failing cases without inventing
   a success or failure. Quote their prompt, target, and stage outputs.
5. Explain model roles and clipping; use one recorded rollout to connect reward,
   advantage, and update. Distinguish reward optimization from correctness.
6. State one alternative explanation and a small follow-up experiment. Running
   the proposed follow-up is optional and carries no bonus points.

## 4. Run order and resource budget

Install the pinned environment, then validate the complete dataset. The first
command downloads only the tokenizer; model weights are downloaded on first
training use. Installation and downloads are outside the training-time target.

```sh
uv run python run.py validate-data --out outputs/data-check
uv run python run.py train --smoke --device cpu --out outputs/smoke
uv run python run.py evaluate --smoke --device cpu --out outputs/smoke --split dev
```

The smoke run tests wiring only. After implementing and checking all eight
functions, run the fixed main recipe on one CUDA GPU:

```sh
uv run python run.py train --device cuda --out outputs/main
uv run python run.py evaluate --device cuda --out outputs/main --split dev
# Freeze the configuration before this final command. Do not tune from test.
uv run python run.py evaluate --device cuda --out outputs/main --split test
```

The default `configs/main.json` fixes the workload:

| Stage | Updates/batches | Effective batch | Example presentations |
| --- | ---: | ---: | ---: |
| SFT | 256 updates | 16 demonstrations | 4,096 (two shuffled passes) |
| Reward model | 256 updates | 16 preference pairs | 4,096 pairs (two passes) |
| PPO collection | 64 rollout batches | 32 responses | 2,048 newly sampled responses |
| PPO optimization | Two epochs per rollout batch | 32 responses | 128 optimizer updates |

Optimization uses microbatches of 8, with token-weighted gradient accumulation
for SFT/PPO and pair-weighted accumulation for reward fitting. SFT/RM/PPO
learning rates are 0.00005 / 0.00001 / 0.000001; AdamW weight decay is 0.01,
gradient norm cap is 1, and beta is 0.05. Inputs reserve at most 256 tokens,
including generation/scoring space; responses are capped at 8 generated tokens.
No targets or prompts are silently truncated.

Complete the fixed workload and both full evaluation splits. Record actual
time and resource use; elapsed time and GPU model do not affect grading.
The runner's configurable per-command timeout stops at a batch boundary and
retains partial records. Read `runtime.md` for environment checks and recovery.

For a quick development check, add `--quick` to development evaluation: it uses
64 prompts and writes a separate `evaluation-dev-quick.json`. The required
report still uses all 256 dev and all 256 test prompts.

The runner saves a checkpoint after each stage. If interrupted, repeat the
same training command with `--resume`: completed stages are reused, while an
unfinished stage restarts from its beginning with its recorded seed. This is
stage-boundary recovery, not resumption from an arbitrary optimizer step.
Keep failed-attempt logs and include their cost in your report.

Allow 16 GB system RAM and at least 10 GB free disk for the environment, cache,
and one set of full-model checkpoints; reserve more disk for repeated runs.
Arrange access to a compatible GPU before the main run. CPU smoke and unit
tests remain useful during implementation. Contact the teaching team if GPU
access, drivers, memory, or downloads block the main run. Retain errors and
report missing evidence honestly. This package does not allocate cluster time.
No paid API, DPO, GRPO, larger model, or hyperparameter search is required.

## 5. Submission, reproducibility, and AI use

Copy `report-template.md` to `report.md` and complete it. Create the submission:

```sh
uv run python make_submission.py YOUR_STUDENT_ID --run outputs/main
```

This writes `results.json` and `a3-YOUR_STUDENT_ID.zip`. Its root contains
`a3.py`, `report.md`, and `results.json`; no enclosing folder. It includes the
training records and both evaluation files in `results.json`, not model weights.
Keep a private copy of checkpoints and raw logs. Upload the ZIP to eLearning;
do not submit a public pull request.

You may include optional figures, tables, notes, or small analysis scripts under
`evidence/`, including nested folders (20 MiB total uncompressed). Supported
extensions are `.png`, `.jpg`, `.jpeg`, `.svg`, `.pdf`, `.csv`, `.tsv`, `.json`,
`.txt`, `.md`, `.py`, and `.ipynb`. Link evidence from the report. The packager
rejects symlinks and unsupported files instead of silently omitting them.
Evidence adds no required work or bonus points; the original three-file ZIP
remains valid. Do not put weights, caches, or unrelated personal files there.

If some evidence is unavailable, use `--partial-reason "specific reason"`.
This records missing files honestly and does not claim their results. Available
implementation and evidence receive credit under the same rubric.

The runner records command, code revision/dirty state, source/data hashes,
configuration, model/tokenizer revision, stage seeds, versions, token counts,
stage timing, peak GPU memory, and fixed-reference/reward checks.
Add hardware, any failed attempts, and artifact locations to your report.
AI assistance for concepts, debugging, and coding is permitted; verify its
outputs and disclose material use. Write `AI-use disclosure: None` if unused.
A missing disclosure costs 5 points. Attribute reused code and sources.

### Late submissions

The policy is the same as Assignment 02. The penalty multiplies your earned
assignment score after ordinary grading, including any disclosure deduction.
Count each started 24-hour period after December 4, 2026, 23:59
(Asia/Shanghai) as one late day, using the eLearning submission timestamp.

| Late days | Final score |
| --- | ---: |
| 0 | 100% of the earned score |
| 1 | 90% |
| 2 | 80% |
| 3 | 70% |
| 4 | 60% |
| 5 | 50% |
| 6 or more | 0 |

## Readings

Use the course's Lecture 08 and Lecture 09 notes for conventions.
[InstructGPT](https://arxiv.org/abs/2203.02155) describes the overall sequence;
[PPO](https://arxiv.org/abs/1707.06347) Section 3 explains the clipped objective;
[GAE](https://arxiv.org/abs/1506.02438) motivates the advantage estimator.
The synthetic task and fixed training budget are teaching simplifications.
The required main recipe tunes all GPT-2 parameters; only the CPU smoke run
uses partial tuning. This exercise does not reproduce those papers' empirical
results.
