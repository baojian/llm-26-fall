"""Small, offline calculations for the ungraded Lecture 08 exercises.

All examples are authored teaching fixtures, not predictions from a model.
The functions operate on values in memory and never remove files.
"""

import math
import random
import unicodedata


def normalize_document(text):
    """Use the course's exact-document convention; preserve internal spacing."""
    if not isinstance(text, str):
        raise TypeError("A document must be a string.")
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n").strip()


def exact_duplicate_groups(records):
    """Return repeated normalized texts as ordered lists of record IDs."""
    groups = {}
    seen_ids = set()
    for record in records:
        if record["id"] in seen_ids:
            raise ValueError("Record IDs must be unique.")
        seen_ids.add(record["id"])
        key = normalize_document(record["text"])
        groups.setdefault(key, []).append(record["id"])
    return [ids for ids in groups.values() if len(ids) > 1]


def word_shingles(text, width=2):
    """Whitespace word shingles for the English toy example, not BPE tokens."""
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("Shingle width must be a positive integer.")
    words = normalize_document(text).split()
    return {tuple(words[i:i + width]) for i in range(len(words) - width + 1)}


def jaccard(left, right):
    """Jaccard similarity of two nonempty sets."""
    union = set(left) | set(right)
    if not union:
        raise ValueError("Two empty sets do not identify document similarity.")
    return len(set(left) & set(right)) / len(union)


def completion_score(token_logprobs, *, per_token=False):
    """Score only completion tokens with natural logarithms."""
    values = list(token_logprobs)
    if not values or any(not math.isfinite(x) or x > 0 for x in values):
        raise ValueError("Supply nonempty, finite log probabilities <= 0.")
    total = math.fsum(values)
    return total / len(values) if per_token else total


def accuracy(correct):
    values = list(correct)
    if not values or any(value not in (0, 1) for value in values):
        raise ValueError("Supply nonempty binary correctness indicators.")
    return sum(values) / len(values)


def pass_at_k(n, c, k):
    """HumanEval estimator for success among k of n sampled completions."""
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (n, c, k)):
        raise ValueError("n, c, and k must be integers.")
    if not (1 <= k <= n and 0 <= c <= n):
        raise ValueError("Require 1 <= k <= n and 0 <= c <= n.")
    return 1 - math.comb(n - c, k) / math.comb(n, k) if n - c >= k else 1.0


def paired_bootstrap(baseline, candidate, *, draws=5000, seed=20261028):
    """Percentile interval for candidate minus baseline accuracy.

    Resample item indices jointly. This conditions on these two fitted models
    and assumes independent evaluation items. It does not estimate training
    seed variability, prompt selection bias, or uncertainty from related items.
    """
    left, right = list(baseline), list(candidate)
    accuracy(left)
    accuracy(right)
    if len(left) != len(right):
        raise ValueError("Paired predictions must cover the same items.")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 2:
        raise ValueError("At least two bootstrap draws are required.")
    differences = [b - a for a, b in zip(left, right)]
    rng = random.Random(seed)
    size = len(differences)
    samples = sorted(sum(differences[rng.randrange(size)] for _ in range(size)) / size
                     for _ in range(draws))

    def percentile(fraction):
        index = (len(samples) - 1) * fraction
        lower, upper = math.floor(index), math.ceil(index)
        return samples[lower] + (samples[upper] - samples[lower]) * (index - lower)

    return {"difference": math.fsum(differences) / size,
            "low": percentile(0.025), "high": percentile(0.975),
            "draws": draws, "seed": seed, "samples": samples}


def comparison_predictions():
    """Explicit synthetic paired outcomes: 50 both, 10 A only, 14 B only."""
    baseline = [1] * 50 + [1] * 10 + [0] * 14 + [0] * 26
    candidate = [1] * 50 + [0] * 10 + [1] * 14 + [0] * 26
    return baseline, candidate
