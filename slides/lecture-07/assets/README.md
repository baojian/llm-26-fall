# Lecture 07 visual and data provenance

Every numeric chart and CSV in this folder is a **synthetic teaching example**.
No training runs, GPU measurements, speaker results, or reproduced paper
measurements are represented by these numbers.

The authored surface is
`L(N,D) = 1.5 + 0.8/sqrt(N/1e8) + 0.8/sqrt(D/2e9)`.
The dense compute approximation is `C = 6ND`, with reference
`C0 = 1.2e18 FLOPs`. At fixed compute, its exact minimum is
`N = 1e8 sqrt(C/C0)` and `D = 2e9 sqrt(C/C0)`. Thus the frontier is
`L = 1.5 + 1.6 (C/C0)^(-0.25)`.

| Asset | What it shows |
| --- | --- |
| `isoflop.json` | Loss at one fixed compute budget, with five candidate sizes |
| `budget.json`, `budget-states.json` | Same surface at 1, 4, and 16 times the budget |
| `grid.json` | Proposed experiment locations, not completed runs |
| `frontier.json` | Mean excess loss at four fitting budgets |
| `holdout.json` | Total loss and a deliberately shifted larger-scale holdout |
| `synthetic-runs.csv` | Four fitting budgets and one holdout, three invented replicates each |
| `training-matmul.svg`, `.excalidraw` | Forward, input-gradient, and weight-gradient products with dimensions |
| `warmup-fraction.svg`, `.excalidraw` | 200 updates occupy 20% or 2% of two normalized run bars |
| `prediction-check.svg`, `.excalidraw` | Fit, freeze, predict, then inspect the reserved larger run |

The three diagrams are original course drawings. `build_diagrams.py` writes
both editable scenes and SVGs from the same coordinates and text, following
Lecture 06's export convention. In the matrix diagram, `G = ∂L/∂Y`; each
product has approximately `2mdh` FLOPs. The warmup bars each represent a whole
run, so their absolute update durations differ.

The Gopher/Chinchilla table in the slides uses reported counts from Hoffmann
et al., Table 1, rather than this synthetic dataset. It is a native table,
with the source and limits of the compute comparison in the slide notes.

The fitting budgets are `C/C0 = 0.25, 1, 4, 16`; the holdout is `64`.
Replicate offsets are exactly `-0.01, 0, +0.01 nats/token`. At 64, add a further
`0.16` to all three replicates to create an extrapolation failure. This shift
does not diagnose a particular real-world mechanism. These offsets are not a
measured seed distribution or uncertainty interval. The exercise supplies the
floor 1.5 and a decision tolerance 0.05 in advance. Neither is a research
recommendation.

Rebuild with the repository environment:

```sh
uv run python slides/lecture-07/assets/build_assets.py
uv run python slides/lecture-07/assets/build_diagrams.py
```

`scaling.py` and the notebook use only Python's standard library. The browser
loads local Plotly JSON through the shared slide framework. The small demo
cycles prepared data and resets to the 1× budget. All axes, units, and
synthetic status are visible in the deck. No external image or guest material
is bundled.

The loss model and IsoFLOP design are motivated by
[Hoffmann et al., §3](https://arxiv.org/html/2203.15556v1).
The coefficient values, worked examples, and charts here are original teaching
constructions. Other primary-source links appear beside the relevant slides
or in their speaker notes.
