"""Copy to submissions/<your-lowercase-github-username>.py.

Implement the five helpers. Input validation and output assembly are supplied.
"""

import hashlib
import json
import math
from itertools import combinations


# AI assistance: Codex wrote the predictions.
PREDICTIONS = {
    "exact_copy": [['train-copy', 'eval-copy']],
    "short_documents": [],
    "contained_passage": [['train-long', 'eval-short']],
}
# Repeated bigrams count once; short texts have no bigrams.
# A contained passage can still have low whole-document similarity.
MY_CASES = [
    (
        {
            "documents": [
                {"id": "z-repeat-train", "split": "train", "tokens": ["tap", "tap", "tap", "tap"]},
                {"id": "a-repeat-eval", "split": "eval", "tokens": ["tap", "tap", "tap"]},
                {"id": "short-train", "split": "train", "tokens": ["tap"]},
                {"id": "empty-eval", "split": "eval", "tokens": []},
            ],
            "n": 2,
            "num_hashes": 6,
            "num_bands": 3,
            "jaccard_threshold": 1.0,
            "containment_threshold": 1.0,
        },
        [["z-repeat-train", "a-repeat-eval"]],
    ),
    (
        {
            "documents": [
                {
                    "id": "archive", "split": "train",
                    "tokens": ["cedar", "moss", "stone", "gap", "moss", "stone", "river", "tail"],
                },
                {
                    "id": "passage", "split": "eval",
                    "tokens": ["cedar", "moss", "stone", "river"],
                },
            ],
            "n": 2,
            "num_hashes": 6,
            "num_bands": 2,
            "jaccard_threshold": 0.8,
            "containment_threshold": 1.0,
        },
        [["archive", "passage"]],
    ),
]
NOTES = (
    "With four bands, the 38 candidates had 95% recall and 100% precision; "
    "with eight bands, the 62 candidates had 100% recall and 64.5% precision. "
    "The eight-band candidate eval-02/train-02 has Jaccard 0.52 but eval "
    "containment 1; all 1504 containment pairs stay unchanged because that "
    "check covers every train/eval pair. "
    "Eval-16 is shared notice text, showing why overlapping shingles are not "
    "proof that a model saw the evaluation passage during training. "
    "The supplied audit gives full/clean perplexities of 5.66/22.63 for A "
    "and 8.00/11.31 for B, so A's full-set lead reverses on the shared clean "
    "subset. "
    "These invented scores use 2**(total loss bits / total scored tokens), "
    "including EOS and excluding BOS; no models were trained for this audit."
)


def rank(shingle, seed):
    """Supplied: deterministic approximation to a random shingle ordering."""
    payload = json.dumps(
        ["minhash", seed, len(shingle), *shingle],
        ensure_ascii=False, separators=(",", ":"),
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest(), "big")


def shingles(tokens, n):
    """Return distinct contiguous n-tuples; short documents give an empty set."""
    return {tuple(tokens[start:start + n]) for start in range(len(tokens) - n + 1)}


def signature(shingle_set, num_hashes):
    """Return the seeded minima for a nonempty set, in seed order."""
    return [min(rank(shingle, seed) for shingle in shingle_set) for seed in range(num_hashes)]


def lsh_candidates(signatures, num_bands):
    """Return a set of sorted ID tuples matching at least one same-index band."""
    if not signatures:
        return set()
    rows = len(next(iter(signatures.values()))) // num_bands
    buckets = {}
    for doc_id in sorted(signatures):
        values = signatures[doc_id]
        for band in range(num_bands):
            start = band * rows
            key = (band, tuple(values[start:start + rows]))
            buckets.setdefault(key, []).append(doc_id)
    return {pair for members in buckets.values() for pair in combinations(members, 2)}


def verify_pairs(candidates, sets, threshold):
    """Return candidate pairs with exact Jaccard >= threshold; sets are nonempty."""
    return {
        tuple(sorted((left, right)))
        for left, right in candidates
        if len(sets[left] & sets[right]) / len(sets[left] | sets[right]) >= threshold
    }


def containment_pairs(sets, split_by_id, threshold):
    """Check every train/eval pair; return (train_id, eval_id) tuples."""
    train_ids = [doc_id for doc_id in sets if split_by_id[doc_id] == "train"]
    eval_ids = [doc_id for doc_id in sets if split_by_id[doc_id] == "eval"]
    return {
        (train_id, eval_id)
        for train_id in train_ids
        for eval_id in eval_ids
        if len(sets[train_id] & sets[eval_id]) / len(sets[eval_id]) >= threshold
    }


def solve(case):
    """Supplied validation and orchestration."""
    documents = case["documents"]
    n, h, b = case["n"], case["num_hashes"], case["num_bands"]
    j, c = case["jaccard_threshold"], case["containment_threshold"]
    ids = [record["id"] for record in documents]
    if n <= 0 or h <= 0 or b <= 0 or h % b:
        raise ValueError("positive dimensions required; num_bands must divide num_hashes")
    if len(set(ids)) != len(ids):
        raise ValueError("document IDs must be unique")
    if any(not math.isfinite(value) or not 0 < value <= 1 for value in (j, c)):
        raise ValueError("thresholds must be finite and in (0, 1]")
    all_sets = {record["id"]: shingles(record["tokens"], n) for record in documents}
    sets = {key: value for key, value in all_sets.items() if value}
    signatures = {key: signature(value, h) for key, value in sets.items()}
    candidates = lsh_candidates(signatures, b)
    verified = verify_pairs(candidates, sets, j)
    split_by_id = {record["id"]: record["split"] for record in documents}
    contamination = containment_pairs(sets, split_by_id, c)
    return {
        "candidate_pairs": [list(pair) for pair in sorted(candidates)],
        "verified_pairs": [list(pair) for pair in sorted(verified)],
        "contamination_pairs": [list(pair) for pair in sorted(contamination)],
        "unscorable_ids": sorted(key for key, value in all_sets.items() if not value),
    }
