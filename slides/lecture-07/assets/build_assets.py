"""Rebuild explicitly synthetic chart data; no training or downloads."""

import csv
import json
from pathlib import Path

from scaling import FLOOR, REFERENCE_COMPUTE, optimal_allocation, toy_loss

ROOT = Path(__file__).resolve().parent


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def layout(x_title, y_title, log_x=True, log_y=False):
    return {
        "xaxis": {"title": {"text": x_title}, "type": "log" if log_x else "linear"},
        "yaxis": {"title": {"text": y_title}, "type": "log" if log_y else "linear"},
        "showlegend": False,
        "margin": {"l": 110, "r": 40, "t": 25, "b": 90},
    }


def trace(xs, ys, color="#20578c", mode="lines", name="Synthetic example"):
    return {"x": xs, "y": ys, "type": "scatter", "mode": mode,
            "name": name, "line": {"color": color, "width": 4},
            "marker": {"color": color, "size": 12}}


def main():
    sizes = [10 ** (7 + i / 40) for i in range(81)]
    states = []
    for multiple in [1, 4, 16]:
        loss = [toy_loss(n, REFERENCE_COMPUTE * multiple / (6 * n)) for n in sizes]
        best_n, best_d = optimal_allocation(multiple)
        best_loss = toy_loss(best_n, best_d)
        states.append({
            "multiple": multiple, "parameters": best_n, "tokens": best_d,
            "loss": best_loss,
            "data": [trace([n / 1e6 for n in sizes], loss),
                     trace([best_n / 1e6], [best_loss], "#a51d2d", "markers", "Toy optimum")],
        })
    chart_layout = layout("Model parameters (millions, log scale)", "Synthetic NLL (nats/token)")
    chart_layout["xaxis"].update({"tickmode": "array", "tickvals": [10, 25, 50, 100, 200, 400, 1000],
                                 "ticktext": ["10", "25", "50", "100", "200", "400", "1000"]})
    chart_layout["yaxis"]["range"] = [1.8, 4.6]
    save("budget.json", {"data": states[0]["data"], "layout": chart_layout})
    save("budget-states.json", states)
    marked = [25e6, 50e6, 100e6, 200e6, 400e6]
    save("isoflop.json", {
        "data": [states[0]["data"][0], trace([n / 1e6 for n in marked],
                  [toy_loss(n, REFERENCE_COMPUTE / (6 * n)) for n in marked],
                  "#a51d2d", "markers", "Equal-compute candidates")],
        "layout": chart_layout,
    })
    grid_layout = layout("Model parameters (millions, log scale)", "Training tokens (billions)",
                         log_y=True)
    grid_layout["xaxis"].update({"tickmode": "array", "tickvals": [25, 50, 100, 200],
                                "ticktext": ["25", "50", "100", "200"], "range": [1.3, 2.4]})
    grid_layout["yaxis"].update({"tickmode": "array", "tickvals": [0.5, 1, 2, 4],
                                "ticktext": ["0.5", "1", "2", "4"], "range": [-0.4, 0.7]})
    grid_layout["showlegend"] = True
    grid_layout["legend"] = {"orientation": "h", "x": 0, "y": 1.2}
    grid_layout["margin"]["t"] = 65
    save("grid.json", {
        "data": [trace([n for n in [25, 50, 100] for _ in range(3)],
                       [d for _ in range(3) for d in [0.5, 1, 2]],
                       mode="markers", name="Proposed fit grid"),
                 trace([200], [4], "#a51d2d", "markers", "Reserved larger condition")],
        "layout": grid_layout,
    })
    multiples = [0.25, 1, 4, 16, 64]
    means = [FLOOR + 1.6 * x ** -0.25 for x in multiples]
    with (ROOT / "synthetic-runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["kind", "split", "compute_multiple", "replicate", "loss_nats"])
        for x, mean in zip(multiples, means):
            for replicate, offset in enumerate([-0.01, 0, 0.01], 1):
                loss = mean + offset + (0.16 if x == 64 else 0)
                writer.writerow(["synthetic", "holdout" if x == 64 else "fit", x, replicate, f"{loss:.12f}"])
    front_layout = layout("Training compute / C₀ (log scale)", "L − 1.5 (nats/token)", log_y=True)
    front_layout["xaxis"].update({"tickmode": "array", "tickvals": multiples[:4],
                                 "ticktext": [str(x) for x in multiples[:4]]})
    front_layout["yaxis"].update({"tickmode": "array", "tickvals": [0.5, 0.8, 1.2, 1.6, 2.4],
                                 "ticktext": ["0.5", "0.8", "1.2", "1.6", "2.4"]})
    save("frontier.json", {
        "data": [trace(multiples[:4], [loss - FLOOR for loss in means[:4]], mode="lines+markers")],
        "layout": front_layout,
    })
    holdout_layout = layout("Training compute / C₀ (log scale)", "Synthetic NLL (nats/token)")
    holdout_layout["xaxis"].update({"tickmode": "array", "tickvals": multiples,
                                   "ticktext": [str(x) for x in multiples]})
    holdout_layout["showlegend"] = True
    holdout_layout["legend"] = {"orientation": "h", "x": 0, "y": 1.2}
    holdout_layout["margin"]["t"] = 65
    extrapolation = trace(multiples[3:], means[3:], name="Extrapolation")
    extrapolation["line"]["dash"] = "dash"
    save("holdout.json", {
        "data": [trace(multiples[:4], means[:4], mode="lines+markers", name="Fit means"),
                 extrapolation,
                 trace([64], [means[-1] + 0.16], "#a51d2d", "markers", "Holdout mean")],
        "layout": holdout_layout,
    })


if __name__ == "__main__":
    main()
