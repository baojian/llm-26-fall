"""Arithmetic and synthetic examples for ungraded Lecture 07 practice."""

import math

REFERENCE_COMPUTE = 1.2e18
REFERENCE_PARAMETERS = 1e8
REFERENCE_TOKENS = 2e9
FLOOR = 1.5


def positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def training_flops(parameters, tokens):
    """Dense 6ND approximation, with both inputs expressed as counts."""
    return 6 * positive(parameters, "parameters") * positive(tokens, "tokens")


def persistent_bytes(parameters):
    """Four FP32 arrays: weights, gradients, and two Adam moments."""
    return 16 * positive(parameters, "parameters")


def toy_loss(parameters, tokens):
    """Invented loss surface in nats/token, not a measured scaling law."""
    n = positive(parameters, "parameters") / REFERENCE_PARAMETERS
    d = positive(tokens, "tokens") / REFERENCE_TOKENS
    return FLOOR + 0.8 / math.sqrt(n) + 0.8 / math.sqrt(d)


def optimal_allocation(compute_multiple):
    """Analytic minimum for this particular symmetric toy loss surface."""
    scale = math.sqrt(positive(compute_multiple, "compute_multiple"))
    return REFERENCE_PARAMETERS * scale, REFERENCE_TOKENS * scale


def fit_power_law(points, floor):
    """Fit mean losses as floor + K*x**(-gamma), with a supplied floor.

    `points` contains (positive compute multiple, mean loss) pairs.
    The caller chooses the fitting split. Holdout rows must stay separate.
    This ordinary least-squares fit in log excess loss is a teaching method,
    not the robust nonlinear fitting procedure used in Chinchilla.
    """
    if not math.isfinite(floor):
        raise ValueError("floor must be finite")
    pairs = list(points)
    if len(pairs) < 2:
        raise ValueError("at least two fitting points are required")
    xs, ys = [], []
    for compute, loss in pairs:
        xs.append(math.log(positive(compute, "compute")))
        ys.append(math.log(positive(loss - floor, "loss minus floor")))
    x_mean, y_mean = sum(xs) / len(xs), sum(ys) / len(ys)
    variance = sum((x - x_mean) ** 2 for x in xs)
    if variance == 0:
        raise ValueError("distinct compute levels are required")
    slope = sum((x - x_mean) * (y - y_mean)
                for x, y in zip(xs, ys)) / variance
    return math.exp(y_mean - slope * x_mean), -slope


def predict_loss(compute_multiple, coefficient, exponent, floor):
    positive(compute_multiple, "compute_multiple")
    positive(coefficient, "coefficient")
    if not all(math.isfinite(value) for value in (exponent, floor)):
        raise ValueError("exponent and floor must be finite")
    return floor + coefficient * compute_multiple ** (-exponent)
