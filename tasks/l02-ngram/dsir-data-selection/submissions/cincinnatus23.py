"""Copy this file to submissions/<your-lowercase-github-username>.py.

Fill the three predictions before coding. Implement the five helper functions;
the supplied solve() validates inputs and assembles their results.
"""

import hashlib
import json
import math


PREDICTIONS = {
    "same_models": ["b"],
    "ratio_beats_likelihood": ["b"],
    "sampling_is_not_topk": ["r"],
}
MY_CASES = [
    (
        {
            "num_buckets": 4,
            "alpha": 1.0,
            "raw_fit": [
                ["red"],
                ["red", "red"],
            ],
            "target_fit": [
                ["red"],
                ["red"],
                ["red"],
            ],
            "pool": [
                {"id": "a", "tokens": ["red", "red"]},
                {"id": "b", "tokens": ["red"]},
            ],
            "uniforms": {
                "a": 0.5,
                "b": 0.5,
            },
            "k": 1,
        },
        ["b"],
    ),
    (
        {
            "num_buckets": 4,
            "alpha": 1.0,
            "raw_fit": [
                ["red"],
                ["red"],
                ["red"],
                ["blue"],
            ],
            "target_fit": [
                ["red"],
                ["blue"],
                ["blue"],
                ["blue"],
            ],
            "pool": [
                {"id": "a", "tokens": ["red"]},
                {"id": "b", "tokens": ["blue"]},
            ],
            "uniforms": {
                "a": 0.99,
                "b": 0.01,
            },
            "k": 1,
        },
        ["a"],
    ),
]

NOTES = (
    "raw分母的作用是降低原始数据中已经很常见的特征的重要性，因此一个特征即使在目标数据中出现频率较高，"
    "如果它在raw数据中同样常见，也不会得到很高的重要性比率。在本次开发实验中，M=16时DSIR的平均JS散度为"
    "0.228352，而M=256时下降到0.208543，因此最终选择M=256。Gumbel采样使DSIR不同于直接进行ratio top-k，"
    "因为较低权重的文档仍然有机会被选中，同时不同随机数会使多次采样的结果产生变化。不过本次实验中uniform sampling"
    "的平均JS散度只有0.076473，明显优于DSIR，说明基于特征匹配得到的重要性权重并不一定能够带来更好的数据选择效果，"
    "共享的boilerplate或候选数据规模也可能限制DSIR的表现。"
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
    for i, token in enumerate(tokens):
        counts[bucket((token,), num_buckets)] += 1
        if i + 1 < len(tokens):
            bigram = (token, tokens[i + 1])
            counts[bucket(bigram, num_buckets)] += 1
    return counts


def fit_distribution(documents, num_buckets, alpha):
    """Pool each document's feature counts; add alpha to every bucket."""
    counts = [0] * num_buckets

    for tokens in documents:
        doc_counts = feature_counts(tokens, num_buckets)

        for i in range(num_buckets):
            counts[i] += doc_counts[i]

    total = sum(counts)

    scale = max(1.0, total / num_buckets, alpha)

    scaled_alpha = alpha / scale
    scaled_total = total / scale

    denominator = scaled_total + scaled_alpha * num_buckets

    return [
        (count / scale + scaled_alpha) / denominator
        for count in counts
    ]

def log_importance(counts, raw_probs, target_probs):
    """Return the sum of count * (log target - log raw), without averaging."""
    return sum(
        count * (math.log(target_prob) - math.log(raw_prob))
        for count, raw_prob, target_prob
        in zip(counts, raw_probs, target_probs)
    )

def normalized_weights(log_weights):
    """Return an ID-to-weight dict using stable exponentiation; handle {}."""
    if not log_weights:
        return {}
    max_log_weight = max(log_weights.values())
    exp_weights = {
        doc_id: math.exp(log_weight - max_log_weight)
        for doc_id, log_weight in log_weights.items()
    }
    total = sum(exp_weights.values())
    return {
        doc_id: weight / total
        for doc_id, weight in exp_weights.items()
    }

def sample_ids(log_weights, uniforms, k):
    """Choose k IDs by decreasing log weight + Gumbel noise, then ID."""
    if k == 0:
        return []
    priorities = {
        doc_id: log_weight - math.log(-math.log(uniforms[doc_id]))
        for doc_id, log_weight in log_weights.items()
    }
    return [
        doc_id
        for doc_id, _ in sorted(
            priorities.items(),
            key=lambda item: (-item[1], item[0]),
        )[:k]
    ]

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
