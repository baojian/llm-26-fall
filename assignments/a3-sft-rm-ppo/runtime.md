# GPU setup and run records

## Fixed configuration

Use a GPU compatible with the
supplied PyTorch/CUDA environment and with sufficient memory for the recipe.
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

Allow 16 GB host RAM and at least 10 GB disk for environment/cache/checkpoints.
Keep downloads and installation outside training timing. Full-model stage
checkpoints use much more disk than the historical last-block-only prototype.
Additional runs need additional storage.

## Debugging and recovery

The separate smoke recipe runs on CPU and tests all stages with two SFT
updates, two preference updates, and two sampled PPO responses. It trains only
the last block and final normalization. It establishes wiring, not learning,
GPU compatibility, or full-run time. Unit tests require no downloaded weights.

`--quick` evaluates 64 development prompts into a separate file. The submission
requires the complete development/test evaluations. Smoke, quick, and calibration
results must never be described as the fixed main experiment.

The runner records synchronized CUDA wall times, stage-loop times, input and
sampled-response counts, peak allocated/reserved VRAM, hardware, source/data
hashes, config, model revision, and failed attempts. The default 45-minute
per-command recovery timeout is configurable with `--max-wall-minutes`.
It stops at the next batch boundary, retaining partial records. `--resume`
skips completed stages and restarts an interrupted stage with its recorded seed.
It does not restore an interrupted stage's optimizer step. Count repeated work.

The course assumes access to at least one compatible GPU. Required work is
defined by the fixed update counts and evaluations above. Record actual
GPU-hours across all attempts, including interrupted work. Additional GPU
availability does not add required runs or improve the grading criteria.
