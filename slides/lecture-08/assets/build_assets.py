"""Rebuild editable charts from cited historical values and synthetic fixtures."""

import json
from pathlib import Path

from evaluation import accuracy, comparison_predictions, paired_bootstrap, pass_at_k

ROOT = Path(__file__).resolve().parent
COLORS = ["#285b8e", "#b45a32"]


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def bars(names, values, *, ylabel, ceiling=100):
    return {
        "data": [{"type": "bar", "name": name, "x": names, "y": y,
                  "marker": {"color": COLORS[i]},
                  "text": [f"{v:.1f}" for v in y], "textposition": "outside",
                  "cliponaxis": False}
                 for i, (name, y) in enumerate(values)],
        "layout": {"barmode": "group", "yaxis": {"title": {"text": ylabel}, "range": [0, ceiling]},
                   "xaxis": {"automargin": True}, "showlegend": True,
                   "legend": {"orientation": "h", "y": 1.16, "x": 0.05},
                   "margin": {"l": 100, "r": 35, "t": 55, "b": 100}},
        "config": {"displayModeBar": False, "responsive": True},
    }


def main():
    # HellaSwag paper, Figure 4. Also supplied in Spring PPTX slide 17.
    historical = bars(["Default", "Ending only", "Shuffled", "Shuffled +<br>ending only"],
                      [("SWAG", [86.7, 74.8, 77.0, 60.4]),
                       ("HellaSwag", [46.7, 41.4, 36.2, 31.6])],
                      ylabel="Accuracy (%)")
    save("historical-shortcuts.json", historical)

    baseline, candidate = comparison_predictions()
    mixed_a = baseline + [1] * 19 + [0]
    mixed_b = candidate + [1] * 11 + [0] * 9
    mixed = [100 * accuracy(mixed_a), 100 * accuracy(mixed_b)]
    clean = [100 * accuracy(baseline), 100 * accuracy(candidate)]
    states = [
        {"label": "Mixed evaluation", "size": 120, "scores": mixed,
         "description": "Synthetic: 100 clean items plus 20 exposed items. Raw 65.8%, deduplicated 62.5%."},
        {"label": "Clean evaluation", "size": 100, "scores": clean,
         "description": "Synthetic: the 100 clean items only. Raw 60.0%, deduplicated 64.0%."},
    ]
    save("evaluation-states.json", states)
    plot = bars(["Raw data", "Deduplicated data"], [("Accuracy", mixed)], ylabel="Accuracy (%)")
    plot["data"][0]["marker"]["color"] = COLORS
    plot["layout"]["showlegend"] = False
    save("contamination.json", plot)

    result = paired_bootstrap(baseline, candidate)
    frequencies = {}
    for sample in result["samples"]:
        point = round(100 * sample)
        frequencies[point] = frequencies.get(point, 0) + 1
    xs = sorted(frequencies)
    save("bootstrap.json", {
        "data": [{"type": "bar", "x": xs, "y": [frequencies[x] for x in xs],
                  "marker": {"color": COLORS[0]}}],
        "layout": {
            "xaxis": {"title": {"text": "Candidate minus baseline (percentage points)"}, "range": [-20, 25]},
            "yaxis": {"title": {"text": "Bootstrap samples"}},
            "margin": {"l": 100, "r": 30, "t": 20, "b": 100},
            "shapes": [{"type": "line", "x0": 0, "x1": 0, "y0": 0, "y1": 1,
                        "yref": "paper", "line": {"color": "#a51d2d", "dash": "dash"}}],
        },
        "config": {"displayModeBar": False, "responsive": True},
    })
    summary = {key: value for key, value in result.items() if key != "samples"}
    save("teaching-results.json", {
        "kind": "synthetic teaching fixtures; no language-model training or inference",
        "clean": {"n": 100, "baseline_correct": sum(baseline), "candidate_correct": sum(candidate)},
        "exposed": {"n": 20, "baseline_correct": 19, "candidate_correct": 11},
        "bootstrap": summary,
        "pass_at_k": {"n": 10, "c": 2, "pass1": pass_at_k(10, 2, 1), "pass5": pass_at_k(10, 2, 5)},
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
