"""Copy this file to submissions/<your-lowercase-github-username>.py.

Fill the three predictions before coding. Implement the five helper functions;
the supplied solve() validates inputs and assembles their results.
"""

import hashlib
import json
import math


# AI assistance: Codex help to wrote the predictions.
PREDICTIONS = {
    "same_models": ['b'],
    "ratio_beats_likelihood": ['b'],
    "sampling_is_not_topk": ['r'],
}
# Case 1 checks repeated counts; case 2 checks sampling against ratio top-k.
MY_CASES = [
    (
        {
            "raw_fit": [["red"], ["red"], ["red"]],
            "target_fit": [["red", "red", "red"]],
            "pool": [
                {"id": "a-single", "tokens": ["red"]},
                {"id": "z-repeat", "tokens": ["red", "red", "red"]},
            ],
            "num_buckets": 8,
            "alpha": 1.0,
            "k": 1,
            "uniforms": {"a-single": 0.8, "z-repeat": 0.5},
        },
        ["z-repeat"],
    ),
    (
        {
            "raw_fit": [["red"], ["red"], ["red"]],
            "target_fit": [["blue"], ["blue"]],
            "pool": [
                {"id": "a-blue", "tokens": ["blue"]},
                {"id": "z-red", "tokens": ["red"]},
            ],
            "num_buckets": 2,
            "alpha": 1.0,
            "k": 2,
            "uniforms": {"a-blue": 0.01, "z-red": 0.99},
        },
        ["z-red", "a-blue"],
    ),
]
NOTES = (
    "Dividing by raw probabilities discounts common source features, and "
    "Gumbel sampling can select a lower-ratio document. "
    "Development mean DSIR JS was 0.228 at M=16 and 0.209 at M=256, so the "
    "runner froze M=256 before final evaluation. "
    "Final mean JS was 0.060 for uniform, 0.351 for target-only, 0.237 for "
    "ratio top-k and 0.232 for DSIR. "
    "Uniform won here: the largest DSIR weight was almost 1, and 62 of the "
    "64 seed-0 chunks were focused. "
    "Pool-083 contains several topics despite its ecology label, so label "
    "counts miss some detail; these feature scores do not establish better "
    "language-model training."
)


def bucket(ngram, num_buckets):
    """Supplied: stable, shared bucket space for unigrams and bigrams."""
    payload = json.dumps(
        [len(ngram), *ngram], ensure_ascii=False, separators=(",", ":")
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % num_buckets


def feature_counts(tokens, num_buckets):
    """Return M counts; include repeated unigrams and adjacent bigrams."""
    counts = [0] * num_buckets
    for token in tokens:
        counts[bucket((token,), num_buckets)] += 1
    for left, right in zip(tokens, tokens[1:]):
        counts[bucket((left, right), num_buckets)] += 1
    return counts


def fit_distribution(documents, num_buckets, alpha):
    """Pool each document's feature counts; add alpha to every bucket."""
    counts = [0] * num_buckets
    for tokens in documents:
        for index, count in enumerate(feature_counts(tokens, num_buckets)):
            counts[index] += count
    total = sum(counts)
    # Scale both parts, including the smoothing, before multiplying by M.
    scale = max(total, alpha)
    smoothing = alpha / scale
    denominator = total / scale + smoothing * num_buckets
    return [(count / scale + smoothing) / denominator for count in counts]


def log_importance(counts, raw_probs, target_probs):
    """Return the sum of count * (log target - log raw), without averaging."""
    return math.fsum(
        count * (math.log(target) - math.log(raw))
        for count, raw, target in zip(counts, raw_probs, target_probs)
        if count
    )


def normalized_weights(log_weights):
    """Return an ID-to-weight dict using stable exponentiation; handle {}."""
    if not log_weights:
        return {}
    maximum = max(log_weights.values())
    scaled = {
        doc_id: math.exp(log_weights[doc_id] - maximum)
        for doc_id in sorted(log_weights)
    }
    total = math.fsum(scaled.values())
    return {doc_id: weight / total for doc_id, weight in scaled.items()}


def sample_ids(log_weights, uniforms, k):
    """Choose k IDs by decreasing log weight + Gumbel noise, then ID."""
    priorities = {
        doc_id: value - math.log(-math.log(uniforms[doc_id]))
        for doc_id, value in log_weights.items()
    }
    return sorted(priorities, key=lambda doc_id: (-priorities[doc_id], doc_id))[:k]


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
