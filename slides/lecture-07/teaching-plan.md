# Lecture 07: Compute budgets and scaling

**Central question:** How can small runs guide a larger training budget?

The [course schedule](../../docs/schedule.md) places this lecture on
October 21, 2026. The instructor moved scaling ahead of data preparation to
support Jiaxuan Zou's guest session in period 8. Lecture 08 now covers data
preparation and quality.

## Teaching package

The deck has **34 slides**, with a guest welcome and final references included.
The instructor's material occupies 75 minutes after the existing 15-minute
Quiz 3 slot. The full final period belongs to the guest. Quiz questions, hints,
and keys are not part of this public package.

The first block retrieves Lecture 05's parameter counting and Lecture 06's
training records. It develops FLOPs, memory, and equal-compute comparisons.
The second block develops compute-optimal allocation and an extrapolation
check. It prepares vocabulary and questions for the guest without attempting
to reproduce his talk.

Four ungraded exercises, E01–E04, appear in matching notebook order. The
notebook needs only Python's standard library and the supplied `assets/`.
Students can do the arithmetic or discuss the tables without opening Jupyter.
All loss values and numeric charts are explicitly synthetic. The Chinchilla
case is a separate, cited comparison of reported model sizes and token counts.
Three editable diagrams explain matrix-product costs, warmup fractions, and
the order of fitting and validation. A [reading guide](optional-reading.md)
connects the four exercises to the core sources. No job launch, new
graded work, or GPU allocation is required.

## Timed route

Times are Asia/Shanghai. Exercise times include a short debrief.

| Time | Slides | Activity |
| --- | --- | --- |
| 13:30–13:45 | 1 as holding slide | Existing Quiz 3 slot; private materials handled separately |
| 13:45–13:49 | 2–4 | Route, objectives, and one run record |
| 13:49–13:56 | 5–6 | Dense compute approximation and E01 |
| 13:56–14:00 | 7–8 | Persistent memory and illustrative elapsed time |
| 14:00–14:06 | 9–10 | Equal-compute curve and parametric loss |
| 14:06–14:13 | 11–14 | Power laws, historical estimates, Chinchilla case, protocol effects |
| 14:13–14:15 | — | Questions, collection/transition buffer |
| 14:15–14:25 | — | Break |
| 14:25–14:29 | 15–16 | Allocation derivation |
| 14:29–14:35 | 17–18 | E02 and interactive budget demonstration |
| 14:35–14:43 | 19–23 | Grid (2 min), recipe/frontier (1), warmup (1), evaluation (2), freeze/predict/check (2) |
| 14:43–14:52 | 24–25 | Synthetic fit and E03, including holdout reveal |
| 14:52–14:56 | 26–27 | Failed extrapolation and diagnostic evidence |
| 14:56–15:00 | 28–29 | Data/serving constraints and loss versus capability |
| 15:00–15:03 | 30 | E04: a twelve-run proposal |
| 15:03–15:06 | 31–32 | Guest questions and existing coursework reminder |
| 15:06–15:10 | 34 as needed | Questions, reading, and transition buffer |
| 15:10–15:25 | 33 as holding slide | Break and speaker setup |
| 15:25–16:10 | 33, then guest's own material | Jiaxuan Zou guest session |

A suggested guest allocation is a 2-minute introduction, 33-minute talk, and
10-minute discussion, subject to the speaker's preference. The instructor
confirmed the session and article, not a minute-by-minute guest agenda or a
final speaker-approved English title. “How to Build a Scientific Scaling
Ladder” is the English rendering used for the announcement.

If time slips, shorten slides 27–29 and make E04 a spoken design prompt.
Retain E01's budget comparison, E02's allocation calculation, and E03's
independent prediction check. Finish the instructor block by 15:10 and
start the guest at 15:25. No live training is needed.

## Worked examples and limitations

| Exercise | Checked response |
| --- | --- |
| E01, 4 min | Both runs use `1.2e18` approximate FLOPs. Four FP32 arrays occupy 0.8/3.2 decimal GB, excluding activations and buffers. Assumed sustained 100 TFLOP/s gives 3.33 hours. |
| E02, 4 min | At four times the compute, the symmetric toy optimum is 200M parameters and 4B tokens, loss 2.631371. The 400M/2B alternative has loss 2.7. |
| E03, 7 min | Fit means at 0.25, 1, 4, 16 times the reference budget. Known floor 1.5 gives K=1.6 and gamma=0.25. At 64, prediction 2.065685 versus synthetic holdout 2.225685 gives residual 0.16, exceeding the predeclared classroom tolerance 0.05. |
| E04, 3 min | One design uses 8 smaller fitting runs and 4 larger validation runs. Its unequal budgets total 21 reference budgets, `2.52e19` approximate FLOPs. Count slots and compute separately. |

The toy surface, generation formula, and invented replicate offsets are in
[asset provenance](assets/README.md). Ordinary least squares in log excess
loss with a supplied floor is an instructional simplification. It does not
reproduce Chinchilla's robust nonlinear fitting method or establish empirical
coefficients for a real model family. The deliberate larger-scale shift has
no known physical cause.

The dense `6ND` rule is an approximation. Specify parameter conventions and
account separately for vocabulary projection, attention, recomputation, and
actual throughput. Total model parameters do not by themselves specify MoE
compute. Repeated data and inference demand change the allocation problem.
The historical exponents and “20 tokens per parameter” heuristic do not
constitute universal prescriptions.

The two fitting scales in E04 are a limited design exercise, insufficient to
identify a general five-parameter surface or justify a strong uncertainty claim.
A final scale used for selecting the recipe ceases to be an independent test.

## Guest handoff

[Jiaxuan Zou's website](https://jiaxuanzou0714.github.io/) identifies him as an
undergraduate in Mathematics and Statistics at Xi’an Jiaotong University and
a research intern in ByteDance Seed Pre-training. Check for any update directly
with the speaker before the introduction.

The [talk reading](https://jiaxuanzou0714.github.io/blog/2026/how-to-build-scientific-scaling-ladder/)
connects small training experiments with larger-scale predictions. Suggested
pre-reading: §§1–3 and 10. Our bridge emphasizes comparable measurements,
allocation, and a reserved larger-scale check. Detailed hyperparameter
transfer, MoE design, and deployment-specific methodology remain opportunities
for the guest's examples and discussion.

Do not present the lecture's synthetic values or decision tolerance as the
speaker's recommendations. No guest slides, recording, or promised public
release are included.

## Source map

| Material | Primary reading |
| --- | --- |
| Empirical power laws and dense accounting | [Kaplan et al.](https://arxiv.org/abs/2001.08361), §§2–3 and Appendix A |
| IsoFLOP design, allocation, historical estimates | [Hoffmann et al.](https://arxiv.org/html/2203.15556v1), §3, Tables 1–2, Appendix F |
| Output-layer cost, warmup, optimizer tuning | [Porian et al.](https://arxiv.org/html/2406.19146v4), §§3–4 |
| Optional hyperparameter transfer | [Yang et al.](https://arxiv.org/abs/2203.03466), introduction and experiments |
| Repeated data | [Muennighoff et al.](https://arxiv.org/abs/2305.16264) |
| Training plus serving objective | [Sardana et al.](https://arxiv.org/abs/2401.00448) |

The course revision map points to CS336 scaling lessons as additional
background. This package uses independently authored examples and no Stanford
API. Lecture 06's Spring source map places the earlier scaling discussion in
slides 77–88 of the Spring GPT deck; this lecture develops a new short route
to fit the guest-session format.

## Delivery and verification

```sh
uv sync
uv run python slides/lecture-07/assets/build_diagrams.py
uv run python scripts/slides.py serve
uv run python -m pytest -q tests/test_lecture_07.py
node slides/lecture-07/check-demo.mjs http://127.0.0.1:8000
npm --prefix slides run check -- lecture-07
npm --prefix slides run pdf -- lecture-07
```

Open `/slides/lecture-07/`. The shared **Notebook** launcher creates a personal
copy and preserves saved student work. **Budget** cycles 1×, 4×, 16×; **Reset**
restores 1×. Charts, fonts, and browser libraries are local course assets.
The PDF uses the initial demo state. Review images and PDF remain in ignored
`slides/.checks/lecture-07/`.

Numerical checks include an independent search around the allocation minimum,
recovery of the fit coefficients, exclusion of the larger holdout from fitting,
invalid-input cases, and notebook execution from a fresh copied directory.

Preparation check, October 20, 2026 (Asia/Shanghai): all 10 numerical tests
passed; all six notebook code cells ran in a fresh kernel. The dimensioned
backward products were also checked against autograd. The 34 slides passed
three viewport checks; every slide image and all 34 PDF pages were inspected.
Budget cycling/reset and the notebook launcher passed their browser checks.
Classroom timings remain planning estimates, not a live rehearsal.
