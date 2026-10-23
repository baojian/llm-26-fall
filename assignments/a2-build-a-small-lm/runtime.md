# Runtime and workload

The common configuration is `configs/cpu.json`: float32, two CPU threads,
935,168 parameters, 512 updates per condition, batch 16, context 128, and
1,048,576 processed training tokens per condition. The required training suite
contains baseline pre-norm, post-norm, and deduplicated-data runs. It also includes
small diagnostic and restart checks. No GPU is required.

The first instructor pilot ran on an **Apple M3 Pro, 18 GiB RAM**, macOS 26.4.1,
Python 3.11.14, PyTorch 2.14.0, on October 9, 2026 (Asia/Shanghai).

| Condition | Measured condition wall time | Sampled peak process RSS |
| --- | ---: | ---: |
| Baseline | 18.72 s | 0.497 GiB |
| Post-norm | 18.20 s | 0.517 GiB |
| Deduplicated data | 17.95 s | 0.523 GiB |

Each interval includes data/tokenizer loading, model creation, training,
development evaluation, and checkpoint writing. These are three condition
intervals in a single process; they exclude Python startup, dependency installation,
the separate diagnosis/restart work, plotting, and generation/final evaluation.
Their sum is not a measured end-to-end student workflow time. RSS was sampled
between updates and around setup/evaluation; peaks between samples may be missed.
These figures describe one machine and one seed, not a performance promise.

Reserve at least 30 minutes for initial setup and command checks. Run the smoke
configuration first; if your laptop is much slower, contact the teaching team
with your run metadata rather than silently reducing the required token budget.
Do not train additional configurations to chase a grade. Budget approximately
12–15 hours for implementation, debugging, and interpretation across the three
weeks. This is an estimate, not a measured student/TA workload result; contact
the teaching team early if setup or runtime prevents progress.

Commands in `README.md` use the bundled environment and lockfile. Report your own
measured runtime, hardware, threads, versions, and memory method. The full suite
is the CPU reference path; accelerator execution and cross-device equality are
not part of the required checks.
