"""Supplied plotting command: uv run python plot_results.py --out outputs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for name, color in (("baseline", "#20578c"), ("post", "#a85a29"), ("dedup", "#28673f")):
        run = json.loads((args.out / name / "run.json").read_text())
        tokens_per_step = run["config"]["batch_size"] * run["config"]["model"]["context"]
        axes[0].plot([r["step"] * tokens_per_step for r in run["dev"]], [r["loss"] for r in run["dev"]], label=name, color=color)
        axes[1].plot([r["step"] for r in run["history"]], [r["grad_norm"] for r in run["history"]], label=name, color=color, alpha=0.8)
        axes[2].plot([r["step"] for r in run["history"]], [r["lr"] for r in run["history"]], label=name, color=color)
    for axis, x, y in zip(axes, ["Training tokens processed", "Update", "Update"], ["Development loss (nats/token)", "Gradient norm before clipping", "Learning rate"]):
        axis.set_xlabel(x)
        axis.set_ylabel(y)
        axis.grid(alpha=0.2)
    axes[0].legend()
    figure.savefig(args.out / "comparison.png", dpi=160)
    figure.savefig(args.out / "comparison.svg")
    plt.close(figure)
    print(args.out / "comparison.png")


if __name__ == "__main__":
    main()
