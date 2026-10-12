"""Compare 40 continuous updates with 20 + fresh-process restart + 20.

Run: uv run --extra tokenization python slides/lecture-06/assets/resume_demo.py
By default intermediate checkpoints live in an automatically removed directory.
Use --out PATH to retain them in a NEW ignored local directory.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import torch
from training import RunConfig, ToyRun


def worker(destination, stop, resume=None, reset_optimizer=False):
    started = time.perf_counter()
    run = ToyRun()
    if resume:
        run.restore(resume, reset_optimizer=reset_optimizer)
    trace = []
    while run.completed < stop:
        trace.append(run.advance())
    run.save(destination.with_suffix(".pt"))
    if sys.platform == "win32":
        import psutil
        peak_mib = psutil.Process().memory_info().peak_wset / 1024 ** 2
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mib = peak / (1024 ** 2 if sys.platform == "darwin" else 1024)
    destination.with_suffix(".json").write_text(json.dumps({
        "trace": trace, "run_seconds": time.perf_counter() - started,
        "process_peak_rss_mib": peak_mib,
    }, indent=2) + "\n")


def compare_in(directory):
    started = time.perf_counter()
    script = str(Path(__file__).resolve())
    for name, stop, source, reset in [
        ("continuous", 40, None, False), ("first", 20, None, False),
        ("resumed", 40, "first", False), ("weights_only", 40, "first", True),
    ]:
        command = [sys.executable, script, "--worker", str(directory / name),
                   "--stop", str(stop)]
        if source:
            command += ["--resume", str(directory / f"{source}.pt")]
        if reset:
            command.append("--reset-optimizer")
        subprocess.run(command, check=True, capture_output=True, text=True)
    names = ["continuous", "first", "resumed", "weights_only"]
    records = {name: json.loads((directory / f"{name}.json").read_text()) for name in names}
    states = {name: torch.load(directory / f"{name}.pt", weights_only=True)["model"]
              for name in ["continuous", "resumed", "weights_only"]}
    reference = records["continuous"]["trace"][20:]
    report = {"config": asdict(RunConfig()), "python": platform.python_version(),
              "torch": str(torch.__version__), "platform": platform.platform(),
              "machine": platform.machine(), "atol": 1e-7, "rtol": 0,
              "worker_records": records, "comparisons": {}}
    for name in ["resumed", "weights_only"]:
        candidate = records[name]["trace"]
        difference = max((states["continuous"][k] - states[name][k]).abs().max().item()
                         for k in states["continuous"])
        report["comparisons"][name] = {
            "same_batches": [x["batch_id"] for x in reference] == [x["batch_id"] for x in candidate],
            "same_lrs": [x["lr"] for x in reference] == [x["lr"] for x in candidate],
            "next_loss_difference": abs(reference[0]["nll"] - candidate[0]["nll"]),
            "second_loss_difference": abs(reference[1]["nll"] - candidate[1]["nll"]),
            "max_parameter_difference": difference,
            "within_tolerance": difference <= 1e-7,
        }
    report["wall_seconds"] = time.perf_counter() - started
    (directory / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    result = report["comparisons"]["resumed"]
    if not all(result[k] for k in ["same_batches", "same_lrs", "within_tolerance"]):
        raise AssertionError("The resumed run failed the declared comparison.")
    return report


def run_comparison(destination=None):
    if destination is not None:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        return compare_in(destination)
    with tempfile.TemporaryDirectory(prefix="l06-resume-") as directory:
        return compare_in(Path(directory))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stop", type=int, default=40, help=argparse.SUPPRESS)
    parser.add_argument("--resume", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--reset-optimizer", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.stop, args.resume, args.reset_optimizer)
    else:
        result = run_comparison(args.out)
        print(json.dumps({"comparisons": result["comparisons"],
                          "wall_seconds": result["wall_seconds"]}, indent=2))
