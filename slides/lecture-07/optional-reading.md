# Lecture 07 reading guide

**Question:** When should evidence from small training runs justify a larger one?

Use this guide alongside the [slides](index.html) and [practice notebook](practice.ipynb).
The readings are optional; they add no submission or reference-count requirement.
Jiaxuan Zou's guest session is October 21, 15:25–16:10 (Asia/Shanghai).

The [90-minute board notes](../../docs/lecture-07-scaling-laws/README.md) give
a standalone lesson on Kaplan, Chinchilla, and Porian, with the allocation
derivation written step by step. The LaTeX source includes worked examples,
discussion answers, vector figures, and PDF build instructions.

## A short reading route

| Reading | Focus | Bring one answer to class |
| --- | --- | --- |
| [Kaplan et al. (2020), Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), §§2–3 and Appendix A | Loss, parameter count, data, and compute accounting | Which quantities must remain comparable when fitting a scaling curve? |
| [Hoffmann et al. (2022), Training Compute-Optimal Large Language Models](https://arxiv.org/html/2203.15556v1), §3, Tables 1–2, and Appendix F | IsoFLOP experiments and allocation | Why can increasing the parameter count make a run worse at fixed compute? |
| [Porian et al. (2024), Resolving Discrepancies in Compute-Optimal Scaling of Language Models](https://arxiv.org/html/2406.19146v4), §§3–4 | Output-layer cost, warmup, and optimizer tuning | What would you audit before interpreting a changed exponent? |
| [Jiaxuan Zou, How to Build a Scientific Scaling Ladder](https://jiaxuanzou0714.github.io/blog/2026/how-to-build-scientific-scaling-ladder/), §§1–3 and 10 | The decision a ladder supports and how to validate it | Which result would make you stop before the expensive run? |

If time is short, start with Hoffmann §3.2 and the guest article's introduction
and checklist. Use the other readings to investigate a question from class.

## Keep the evidence distinct

The **reported comparison** is Gopher at 280B parameters / 300B tokens and
Chinchilla at 70B / 1.4T. Hoffmann et al. report comparable training compute
and better evaluated performance for Chinchilla. Rounded model counts in
`6ND` need not reproduce their detailed FLOP accounting; consult Appendix F.

The **course examples** use invented loss coefficients, a supplied floor,
and synthetic replicate offsets. E03's larger run deliberately misses its
prediction. This teaches a validation procedure; it supplies no empirical
estimate for a real model family.

## After the exercises

- **E01:** Explain each of the three matrix products behind `6ND`. Which
  memory costs does `16N` leave out?
- **E02:** Derive the allocation exponents before substituting the symmetric
  toy values. Equal compute alone does not identify the better run.
- **E03:** Separate held-out text within a training run from a held-out scale
  across runs. Explain why changing the fit after seeing the larger run
  requires another independent check.
- **E04:** Count total compute as well as run slots. Specify the comparison,
  fixed measurements, repeats, and decision rule before requesting resources.

The [teaching plan](teaching-plan.md#source-map) lists further readings on
hyperparameter transfer, repeated data, and serving costs. These are useful
extensions after the core allocation and validation ideas.
