"""Copy this file to submissions/<your-lowercase-github-username>.py.

Fill the three predictions before coding. Implement the five helper functions;
the supplied solve() validates inputs and assembles their results.
"""

import hashlib
import json
import math


PREDICTIONS = {
    "same_models": None,
    "ratio_beats_likelihood": None,
    "sampling_is_not_topk": None,
}
MY_CASES = []
NOTES = "REPLACE with 3–5 sentences using your experiment's evidence."


def bucket(ngram, num_buckets):
    """Supplied: stable, shared bucket space for unigrams and bigrams."""
    payload = json.dumps(
        [len(ngram), *ngram], ensure_ascii=False, separators=(",", ":")
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % num_buckets


def feature_counts(tokens, num_buckets):
    """Return M counts; include repeated unigrams and adjacent bigrams."""
    # YOUR CODE HERE
    raise NotImplementedError


def fit_distribution(documents, num_buckets, alpha):
    """Pool each document's feature counts; add alpha to every bucket."""
    # YOUR CODE HERE
    raise NotImplementedError


def log_importance(counts, raw_probs, target_probs):
    """Return the sum of count * (log target - log raw), without averaging."""
    # YOUR CODE HERE
    raise NotImplementedError


def normalized_weights(log_weights):
    """Return an ID-to-weight dict using stable exponentiation; handle {}."""
    # YOUR CODE HERE
    raise NotImplementedError


def sample_ids(log_weights, uniforms, k):
    """Choose k IDs by decreasing log weight + Gumbel noise, then ID."""
    # YOUR CODE HERE
    raise NotImplementedError


def solve(case):
    """Supplied validation and orchestration; see instruction.md for the contract."""
    m, alpha, k = case["num_buckets"], case["alpha"], case["k"]
    pool, uniforms = case["pool"], case["uniforms"]
    ids = [record["id"] for record in pool]
    if m <= 0 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("num_buckets and alpha must be positive and finite")
    if not any(case["raw_fit"]) or not any(case["target_fit"]):
        raise ValueError("each fitting collection must contain features")
    if len(set(ids)) != len(ids) or not 0 <= k <= len(pool):
        raise ValueError("pool IDs must be unique and k must be in range")
    if set(uniforms) != set(ids) or any(
        not math.isfinite(value) or not 0 < value < 1
        for value in uniforms.values()
    ):
        raise ValueError("supply one uniform in (0, 1) for each pool ID")
    raw = fit_distribution(case["raw_fit"], m, alpha)
    target = fit_distribution(case["target_fit"], m, alpha)
    logs = {
        record["id"]: log_importance(feature_counts(record["tokens"], m), raw, target)
        for record in pool
    }
    return {
        "raw_probs": raw,
        "target_probs": target,
        "log_weights": logs,
        "weights": normalized_weights(logs),
        "selected_ids": sample_ids(logs, uniforms, k),
    }
