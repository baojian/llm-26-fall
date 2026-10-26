"""Transparent CPU examples of data curation; no corpus downloads or training."""

import math


def minhash_signature(items, permutations):
    """Minimum-ranked item under each explicit permutation of a tiny universe.

    This teaching implementation enumerates the universe. Production MinHash
    uses hash families instead. It is not a web-scale deduplication tool.
    """
    values = set(items)
    if not values:
        raise ValueError("A signature needs at least one shingle.")
    orders = [list(order) for order in permutations]
    if not orders:
        raise ValueError("Supply at least one permutation.")
    universe = set(orders[0])
    if not values <= universe:
        raise ValueError("The permutation universe must contain every shingle.")
    signature = []
    for order in orders:
        if len(order) != len(universe) or set(order) != universe:
            raise ValueError("Each order must permute the same universe exactly once.")
        signature.append(next(item for item in order if item in values))
    return tuple(signature)


def candidate_probability(similarity, *, bands, rows):
    """Idealized LSH candidate probability with independent MinHash rows."""
    if not math.isfinite(similarity) or not 0 <= similarity <= 1:
        raise ValueError("Similarity must be between zero and one.")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in (bands, rows)):
        raise ValueError("Band and row counts must be positive integers.")
    return 1 - (1 - similarity ** rows) ** bands


def duplicate_components(ids, edges):
    """Connected components of declared match edges, ordered by input IDs.

    Components define possible split groups. Connectivity does not imply all
    members are mutually similar or that all but one should be deleted.
    """
    identifiers = list(ids)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Record IDs must be unique.")
    parent = {item: item for item in identifiers}

    def root(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for left, right in edges:
        if left not in parent or right not in parent:
            raise ValueError("Every edge endpoint must identify a supplied record.")
        parent[root(right)] = root(left)
    groups = {}
    for item in identifiers:
        groups.setdefault(root(item), []).append(item)
    return list(groups.values())


def mixture_exposure(corpus_tokens, processed_tokens, *, exponent=1.0):
    """Expected token allocation for p_i proportional to q_i ** exponent.

    Source probabilities are token weights. Expected passes assume sampling
    uniformly within each source with replacement, ignoring packing overhead.
    """
    counts = list(corpus_tokens)
    if not counts or any(not math.isfinite(x) or x <= 0 for x in counts):
        raise ValueError("Each source must have a positive finite token count.")
    if not math.isfinite(processed_tokens) or processed_tokens <= 0:
        raise ValueError("The processed-token budget must be positive and finite.")
    if not math.isfinite(exponent) or exponent < 0:
        raise ValueError("The sampling exponent must be nonnegative and finite.")
    # A common scale cancels in normalization; the largest weight stays at one.
    # This avoids every weight underflowing when the exponent is large.
    scale = max(counts)
    weights = [(count / scale) ** exponent for count in counts]
    denominator = math.fsum(weights)
    probabilities = [weight / denominator for weight in weights]
    allocated = [processed_tokens * p for p in probabilities]
    return {"probabilities": probabilities, "processed_tokens": allocated,
            "expected_passes": [budget / count for budget, count in zip(allocated, counts)]}
