# Assignment 03: SFT, Reward Modeling, and PPO — report

Student ID:

At most 1,500 words, excluding references and tables. Fill the tables from full
main-run evaluation records; write `unavailable` with a reason for missing
evidence. Do not invent results or substitute smoke/quick/calibration outputs.

## Configuration and provenance

Command(s), revision/dirty state, source/data hashes, GPT-2 revision, software,
hardware/driver, base and stage seeds, token budget, peak allocated/reserved VRAM,
and local artifact locations. Report command `wall_seconds` (including loading
and cold-cache downloads), completed-stage `stage_seconds`, cache state, and
total GPU-hours across all commands and attempts, including failures. Deduplicate
by `attempt_id`; do not add an aggregate time to the attempts it already covers.
Report the stable training `run_id` and use each attempt's full `provenance`
for its environment: training's top-level provenance describes only the initial
attempt. Process RSS is a process-lifetime high-water mark; GPU allocation and
reservation peaks are per command. Do not sum memory peaks. Identify deviations
from the fixed configuration and any missing or incompatible evidence. If older
completed records were archived under partial filenames, identify their original
source/run and preserve their original bytes, provenance, and status; describe
them as historical evidence, not a completed run of the submitted code.

## Pipeline checks

Response mask and first-response predictor; EOS/padding; reward pooling and
loss; current/old/reference/reward/value roles; one PPO rollout worked through.

## Results

Use `outputs/main/evaluation-dev.json` and `evaluation-test.json` (also embedded
in `results.json`). Each full split has 256 prompts. State whether accuracy
and valid-label values are fractions or percentages. For overall values, copy
`metrics[stage]`: `accuracy`, `valid_label_rate`, `mean_reward`, and
`mean_response_tokens`. Response length includes generated EOS.

| Split | Stage | Exact accuracy | Valid-label rate | Mean reward | Mean response tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Dev | Baseline | | | | |
| Dev | SFT | | | | |
| Dev | PPO | | | | |
| Test | Baseline | | | | |
| Test | SFT | | | | |
| Test | PPO | | | | |

Required slices use `slice_metrics["slice:present"][stage]` and
`slice_metrics["slice:absent"][stage]`, with the same four metric keys.
Present/absent contain 192/64 prompts per full split, respectively.

| Split | Slice | Stage | Exact accuracy | Valid-label rate | Mean reward | Mean response tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Dev | Present | Baseline | | | | |
| Dev | Present | SFT | | | | |
| Dev | Present | PPO | | | | |
| Dev | Absent | Baseline | | | | |
| Dev | Absent | SFT | | | | |
| Dev | Absent | PPO | | | | |
| Test | Present | Baseline | | | | |
| Test | Present | SFT | | | | |
| Test | Present | PPO | | | | |
| Test | Absent | Baseline | | | | |
| Test | Absent | SFT | | | | |
| Test | Absent | PPO | | | | |

Copy `reward_model.accuracy` and `reward_model.ties`. To inspect an error,
align the chosen/rejected score pair in `reward_model.pairs` with
`reward_model.pair_ids`, then find that ID in the split's data JSONL.

| Split | Reward pair accuracy | Tie count (of 256 pairs) |
| --- | ---: | ---: |
| Dev | | |
| Test | | |

Copy `delta` and `ci95` from `baseline_to_sft` and `sft_to_ppo`. Label fractions
or multiply by 100 for percentage points. Explain family resampling, small
sample uncertainty, and the limitation of one training seed.

| Split | Comparison | Accuracy change | 95% family-bootstrap interval |
| --- | --- | ---: | --- |
| Dev | Baseline → SFT | | |
| Dev | SFT → PPO | | |
| Test | Baseline → SFT | | |
| Test | SFT → PPO | | |

Missing-object/missing-attribute and query-attribute breakdowns, and slice
paired intervals, are **optional** and carry no extra points. The evaluator
provides condition/attribute entries in `slice_metrics` and SFT→PPO slice
comparisons in `slices` if useful to your analysis.

## Examples, interpretation, and limitations

Give two example IDs with prompt, target, and baseline/SFT/PPO outputs: an
improvement and regression when present, otherwise informative unchanged or
failing cases. Join prediction IDs to the split's JSONL to recover prompts.
Inspect a reward-model error, or state if none was found and discuss a remaining
limitation. Give an alternative explanation and one proposed follow-up (running
it is optional). Explain what the evidence supports and cannot establish.
Preserve negative results and disclose missing evidence.

## AI-use disclosure and attribution

AI-use disclosure: [None, or tool/version, material assistance, and your checks.]

Sources and reused code:
