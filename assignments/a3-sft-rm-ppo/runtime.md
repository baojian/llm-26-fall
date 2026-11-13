# GPU setup and run records

## Fixed configuration

Use one GPU compatible with the supplied PyTorch/CUDA environment. A tentative
planning target is **16 GB VRAM**, not a measured minimum or guarantee: no
full-pipeline GPU timing or peak-memory measurement is available for this
release. Check access and memory early, and contact the teaching team if the
fixed recipe is blocked. This package does not allocate cluster resources.
Complete the fixed configuration and report actual time and peak memory;
GPU model and elapsed time are not graded.

The required configuration is `configs/main.json`: full GPT-2 tuning,
256 SFT updates, 256 preference updates (effective batch 16), and 64 PPO
batches of 32 sampled responses with two update epochs. Microbatch size is 8;
generation/evaluation are batched with a KV cache. Context reserve is 256 and
response cap is 8. All model/objective tensors are float32; CUDA matmuls allow
TF32. The value head shares the current policy backbone. The fixed SFT reference
and learned reward backbone remain frozen during PPO.

A main run presents each of 2,048 training rows twice to SFT and twice to the
reward model. PPO samples one response per row in a separate shuffled pass:
2,048 newly sampled sequences, not multiplied by PPO optimization epochs.
The same counts apply to every student.

## Environment preflight

Work from your private copy of the whole assignment directory, as described
in the handout; all commands below use that directory as their working directory.
Use Python 3.11 and `uv sync --locked`. The retained Torch 2.14.0 lock currently
resolves CUDA 13 dependencies on Linux. NVIDIA documents a minimum driver
family of 580 for CUDA 13.x; newer features may require a newer driver.
The GPU's age alone does not determine driver compatibility. Check the actual
machine before allocating a main run:

```sh
nvidia-smi
uv run python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)"
```

[Driver compatibility reference](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
If a shared machine has an older driver, the teaching team must validate an
appropriate pinned environment or arrange a driver update with its administrator.
Students should not improvise dependency upgrades on the cluster.

Allow 16 GB host RAM and at least 10 GB disk for environment/cache/checkpoints;
additional runs need additional storage. Installation runs separately, but
cold-cache tokenizer/model downloads and model loading occur inside the
training/evaluation command timer. Record the cache state instead of assuming
that command time equals compute time.

## Debugging and recovery

Implement all eight functions and pass `uv run python -m pytest -q` before
smoke training or GPU work. This includes `paired_summary` and JSON diagnostic
contracts. Unit tests require no downloaded weights. The source fingerprint
covers the whole `a3.py`; changing even the evaluator after training makes
existing checkpoints incompatible with that code. Keep source versions and
their records together. Do not edit provenance to bypass this check; if a new
run is unavailable, use the honest partial-submission route.

The separate smoke recipe runs on CPU and tests all stages with two SFT
updates, two preference updates, and two sampled PPO responses. It trains only
the last block and final normalization. It establishes wiring, not learning,
GPU compatibility, or full-run time.

`--quick` evaluates 64 development prompts into a separate file. The submission
requires the complete development/test evaluations. Smoke, quick, and calibration
results must never be described as the fixed main experiment.

The runner records synchronized CUDA command wall times, stage-loop times, input
and sampled-response counts, memory, hardware, source/data hashes, config,
model revision, and failed attempts. Interpret these records as follows:

- `wall_seconds`: elapsed time for that command, including tokenizer/model
  loading and any cold-cache downloads. Sum completed command and failed-attempt
  times for the total execution cost. Deduplicate by `attempt_id` across
  `attempts`, partial/final records, and `failed-attempt-*.json`: several files
  can describe the same attempt. Do not add an aggregate such as
  `total_attempt_wall_seconds` again to its constituent attempts. For legacy
  records without attempt IDs, compare command/timestamp and explain uncertainty.
- `stage_seconds`: training-loop time for each completed stage, excluding
  tokenizer/model loading. It is not the end-to-end pipeline time and omits
  interrupted work not represented by a completed stage.
- The default **45-minute per-command timeout** uses the command clock and
  checks at batch boundaries. It is a configurable recovery safeguard, not a
  benchmark, promised completion time, or grading requirement. A download or
  operation in progress may finish before the next check observes the timeout.
- Each `attempts` entry carries its own full `provenance`, including software,
  hardware, command, and source/configuration identity. Training's top-level
  provenance remains the initial attempt's snapshot; use the individual entries
  as the authoritative environment record for resumed commands. Do not attribute
  later hardware or package versions to the first attempt.
- `peak_process_rss_bytes` is the process-lifetime resident-memory high-water
  mark, including setup/loading; it is not a stage-only incremental measurement.
  `peak_gpu_allocated_bytes` and `peak_gpu_reserved_bytes` are peaks for each
  command, not sums across attempts. Report per-attempt values or the maximum
  with its attempt ID. Preserve missing measurements and errors as unavailable.
- The stable training `run_id` identifies one training history and binds its
  checkpoints and evaluations. Matching configuration/source fingerprints alone
  do not make records from separate training runs interchangeable. The packager
  rejects completed records with different run IDs.

For an interrupted training command, retain its partial and failed-attempt
records. Choose the retry command from the files that actually exist:

| Run-directory state | Retry |
| --- | --- |
| `training-partial.json` exists | Repeat the original training command with `--resume`. |
| Early tokenizer/download failure before `training-partial.json` exists | Repeat the original command **without `--resume`**; retain `failed-attempt-*.json`. |

With `--resume`, completed stages are reused and the unfinished stage restarts
from its beginning with the recorded seed. No intermediate optimizer step is
restored. If an individual stage
cannot finish under the same timeout, choose a larger `--max-wall-minutes`
within your available allocation; repeated restarts with the same inadequate
limit cannot make progress through that stage. For example, **when
`outputs/main/training-partial.json` exists**:

```sh
uv run python run.py train --device cuda --out outputs/main --resume --max-wall-minutes 90
```

Here 90 is an example limit, not an estimated runtime or an allocation. If a
cold-cache download exhausted the first budget, cached files may reduce the
next command's loading cost. Evaluation has no mid-split resume: repeat the
same evaluation command, increasing its timeout if needed. Retain partial
evaluation records and failures even when a later full evaluation succeeds.

The course assumes access to at least one compatible GPU. Required work is
defined by the fixed update counts and evaluations above. Record actual
GPU-hours across all attempts, including interrupted work. Additional GPU
availability does not add required runs or improve the grading criteria.
For active student work, allow an unmeasured 15–20 hours with A2/PPO preparation,
or 20–30+ hours with substantial setup or RL debugging. This is separate from
GPU execution, downloads, installation, and queue waits; no measured GPU
completion time is claimed.

`make_submission.py` includes available valid final records, explicitly labeled
`partial_records`, and `failed_attempts`. Use `--partial-reason` when required
evidence is unavailable and explain it in the report. Preserve error messages
and actual costs; neither a missing result nor a negative result justifies
inventing an outcome.

## Archiving incompatible completed records

`--partial-reason` does not override checks on completed records. If you changed
`a3.py` and cannot supply a compatible new run, archive earlier completed
records explicitly as partial submission evidence. Leave the original run
directory untouched; do not edit JSON status, hashes, provenance, or results.

1. Create a **new, empty** directory. This command deliberately fails if it
   already exists; choose a new directory name instead of overwriting it:

   ```sh
   mkdir outputs/partial-submission
   ```

2. Copy each available completed record from the original run using the
   destination name below. Copy only files that exist. These are byte-for-byte
   copies, including an original `status: "completed"`; the partial filename
   tells the packager to retain them as historical evidence rather than current
   completed results.

   | Original under `outputs/main/` | Copy under `outputs/partial-submission/` |
   | --- | --- |
   | `training.json` | `training-partial.json` |
   | `evaluation-dev.json` | `evaluation-dev-partial.json` |
   | `evaluation-test.json` | `evaluation-test-partial.json` |

   For example, if the training record exists:

   ```sh
   cp outputs/main/training.json outputs/partial-submission/training-partial.json
   ```

3. Retain existing interrupted and failure evidence. Copy an existing
   `*-partial.json` to the same basename in the new directory **only when that
   destination is absent**. If both an interrupted and completed version exist,
   preserve the interrupted version separately under a fresh
   `evidence/partial-submission-originals/` directory and link it from the
   report; never overwrite either version. Optional evidence retains its
   20 MiB limit. Copy all `failed-attempt-*.json` files unchanged into the new
   run directory so the packager includes them automatically.
4. Package the fresh directory and describe why current compatible evidence is
   unavailable, which original run the archive came from, and any separately
   retained interrupted files:

   ```sh
   uv run python make_submission.py YOUR_STUDENT_ID --run outputs/partial-submission --partial-reason "Source changed after the recorded run; earlier records are retained unchanged as historical evidence"
   ```

Inspect `results.json`: archived records belong under `partial_records`, and
missing current results and provenance differences must remain visible. This
route preserves available evidence; it does not certify an old run against
new code or create missing experiment results.
