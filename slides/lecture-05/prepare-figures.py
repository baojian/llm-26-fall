"""Recompute the teaching loss figure and save its full local execution record.

Run from the repository root with:
    uv run --locked --no-sync python slides/lecture-05/prepare-figures.py
"""

import contextlib
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import time
from zoneinfo import ZoneInfo

import torch


def main():
    lecture = Path(__file__).resolve().parent
    root = lecture.parents[1]
    notebook = lecture / "lecture-05-exercise.ipynb"
    artifact = root / "slides/.checks/lecture-05/figure-run.json"
    chart = lecture / "assets/norm-comparison.json"
    command = "uv run --locked --no-sync python slides/lecture-05/prepare-figures.py"

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    provenance = {
        "command": command,
        "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "timezone": "Asia/Shanghai",
        "git_revision": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "notebook_sha256": hashlib.sha256(notebook.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": "float32 (training), float64 (reference checks)",
        "threads": 1,
        "seed": 7,
        "updates_per_variant": 200,
        "variants": ["pre-LN", "post-LN"],
        "data_version": "noa-two-continuations-v1",
        "data": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11],
                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]],
        "vocabulary": ["Noa", "can", "be", "annoying", "but", "she", "is", "a", "great", "cat", "friend", "EOS"],
        "model": {"batch": 2, "length": 10, "width": 16, "heads": 4,
                  "ffn_width": 32, "blocks": 2, "vocabulary_size": 12,
                  "positions": "learned absolute", "ffn_activation": "ReLU",
                  "linear_biases": False, "final_layer_norm": True,
                  "layer_norm_epsilon": 1e-5, "tied_readout": True,
                  "parameters": 4608, "initialization_std": 0.02},
        "optimizer": {"name": "AdamW", "learning_rate": 0.02,
                      "betas": [0.9, 0.999], "epsilon": 1e-8,
                      "weight_decay": 0.0},
        "dropout": 0.0,
        "compute_budget": "200 full-batch CPU updates per variant; loss and gradient norms at updates 0–200",
        "previous_reference_archive": "slides/.checks/lecture-05/pre-noa-20261010T020843+0800/manifest.json",
        "configuration": "slides/lecture-05/teaching-plan.md#exact-teaching-model",
        "raw_artifact": str(artifact.relative_to(root)),
    }
    namespace = {}
    output = io.StringIO()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(output):
            for cell in json.loads(notebook.read_text())["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
    finally:
        torch.set_num_threads(previous_threads)
    provenance["elapsed_seconds"] = round(time.perf_counter() - start, 4)
    provenance["data_sha256"] = hashlib.sha256(
        json.dumps({"vocabulary": provenance["vocabulary"], "documents": provenance["data"]},
                   separators=(",", ":")).encode()
    ).hexdigest()

    figure = json.loads(chart.read_text())
    records = {}
    for trace, prefix in zip(figure["data"][:2], ["pre", "post"]):
        losses = namespace[f"{prefix}_losses"]
        trace["x"] = list(range(len(losses)))
        trace["y"] = losses
        records[prefix] = {
            "loss": losses,
            "gradient_norm": namespace[f"{prefix}_gradient_norms"],
        }
    figure["data"][2]["y"] = [namespace["loss_floor"]] * 2
    # Keep the complete observed range, including any unfavorable spikes.
    maximum = max(max(namespace["pre_losses"]), max(namespace["post_losses"]))
    figure["layout"]["yaxis"]["range"] = [0, max(2.7, maximum * 1.05)]
    figure["layout"]["meta"] = provenance
    chart.write_text(json.dumps(figure, indent=2) + "\n")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps({
        "provenance": provenance,
        "results": records,
        "stdout": output.getvalue(),
    }, indent=2) + "\n")
    print(f"Wrote {chart.relative_to(root)} and {artifact.relative_to(root)}")
    print(f"Final losses: pre-LN {records['pre']['loss'][-1]:.6f}; "
          f"post-LN {records['post']['loss'][-1]:.6f}")


if __name__ == "__main__":
    main()
