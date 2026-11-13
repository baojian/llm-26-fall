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

The runner records synchronized CUDA command wall times, stage-loop times, input and
sampled-response counts, peak allocated/reserved VRAM, hardware, source/data
hashes, config, model revision, and failed attempts. Interpret the clocks as follows:

- `wall_seconds`: elapsed time for that command, including tokenizer/model
  loading and any cold-cache downloads. Sum completed command and failed-attempt
  times for the total execution cost; do not count the same attempt twice.
- `stage_seconds`: training-loop time for each completed stage, excluding
  tokenizer/model loading. It is not the end-to-end pipeline time and omits
  interrupted work not represented by a completed stage.
- The default **45-minute per-command timeout** uses the command clock and
  checks at batch boundaries. It is a configurable recovery safeguard, not a
  benchmark, promised completion time, or grading requirement. A download or
  operation in progress may finish before the next check observes the timeout.

For an interrupted training command, retain its partial and failed-attempt
records. Repeat the original command with `--resume`: completed stages are
reused and the unfinished stage restarts from its beginning with the recorded
seed. No intermediate optimizer step is restored. If an individual stage
cannot finish under the same timeout, choose a larger `--max-wall-minutes`
within your available allocation; repeated restarts with the same inadequate
limit cannot make progress through that stage. For example:

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
