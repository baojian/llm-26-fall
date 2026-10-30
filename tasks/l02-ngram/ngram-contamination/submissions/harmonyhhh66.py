"""Copy to submissions/<your-lowercase-github-username>.py.

Implement the five helpers. Input validation and output assembly are supplied.
"""

import hashlib
import json
import math
from itertools import combinations


PREDICTIONS = {
    "exact_copy": [["train-copy", "eval-copy"]],
    "short_documents": [],
    "contained_passage": [["train-long", "eval-short"]],
}
MY_CASES = [
    (
        {
            "documents":[
                {
                    "id": "train-repeat",
                    "split": "train",
                    "tokens": ["cat", "dog", "cat", "dog", "cat"]

                },
                {
                    "id": "eval-repeat",
                    "split": "eval",
                    "tokens": ["cat", "dog", "cat"]
                },
                {
                    "id": "eval-tiny",
                    "split": "eval",
                    "tokens": ["bird"]
                }

            ],
            "n": 2,
            "num_hashes": 4,
            "num_bands": 2,
            "jaccard_threshold": 0.8,
            "containment_threshold": 1.0
        },
        [["train-repeat", "eval-repeat"]]
    ),
    (
        {
            "documents":[
                {
                    "id": "train-expanded",
                    "split": "train",
                    "tokens": ["u", "v", "w", "a", "b", "c", "d", "x", "y", "z"]
                },
                {
                    "id": "eval-fragment",
                    "split": "eval",
                    "tokens": ["a", "b", "c", "d"]
                }
            ],
            "n": 2,
            "num_hashes": 4,
            "num_bands": 2,
            "jaccard_threshold": 0.8,
            "containment_threshold": 1.0
        },
        [["train-expanded", "eval-fragment"]]
    )
]
# Teaching-team edit, October 20: English translation of the original NOTES
# and comments; round the full-set Run A perplexity to 5.657.
NOTES = """
Increasing LSH bands from four to eight raises the candidate count from 38 to 62 and recall from 0.95 to 1.00, while precision falls from 1.00 to 0.645; four bands miss eval-00/train-00, and eight bands add 22 false candidates.
Jaccard measures overall similarity, while containment asks how much evaluation text is covered: a long training document can contain every shingle of a short evaluation passage despite low Jaccard, so overlap checks cannot be restricted to LSH candidates.
Removing flagged evaluation text reverses the ranking in the constructed audit: Run A/B have full-set perplexities 5.657/8.000 and clean-subset perplexities 22.627/11.314.
Text overlap does not establish leakage or training exposure, because different documents can share common sentences or templates.
"""


def rank(shingle, seed):
    """Supplied: deterministic approximation to a random shingle ordering."""
    payload = json.dumps(
        ["minhash", seed, len(shingle), *shingle],
        ensure_ascii=False, separators=(",", ":"),
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest(), "big")


def shingles(tokens, n):
    """Return distinct contiguous n-tuples; short documents give an empty set."""
    result = set()

    for i in range(len(tokens) - n + 1): # If len(tokens) < n, the result stays empty.
        gram = tuple(tokens[i: i + n])
        result.add(gram)

    return result


def signature(shingle_set, num_hashes):
    """Return the seeded minima for a nonempty set, in seed order."""

    # MinHash represents the n-gram set with a short signature.
    # Matching minima estimate Jaccard similarity.
    # Multiple seeds reduce the variability of the Jaccard estimate.
    result = []

    for seed in range(num_hashes):
        minimum = min(rank(g, seed) for g in shingle_set)
        result.append(minimum)

    return result


def lsh_candidates(signatures, num_bands):
    """Return a set of sorted ID tuples matching at least one same-index band."""
    # LSH finds candidate pairs through shared buckets; exact verification follows.
    # Include the band index to avoid matching different signature positions.
    # Compare documents within each shared band bucket.

    # Handle an empty signature collection.
    if not signatures:
        return set()

    num_hashes = len(next(iter(signatures.values()))) # Read the signature length from the first entry.
    rows_per_band = num_hashes // num_bands

    buckets = {}

    for doc_id, sig in signatures.items():
        for band_index in range(num_bands):
            start = band_index * rows_per_band
            end = start + rows_per_band

            band = tuple(sig[start: end]) # Convert the band to a hashable tuple.
            key = (band_index, band)

            buckets.setdefault(key, []).append(doc_id) # Add the document ID to its bucket.

    candidates = set()

    for ids in buckets.values():
        for a, b in combinations(sorted(ids), 2): # Every pair in a shared bucket is a candidate.
            candidates.add((a, b))

    return candidates



def verify_pairs(candidates, sets, threshold):
    """Return candidate pairs with exact Jaccard >= threshold; sets are nonempty."""
    # Verify exact similarity for the LSH candidates.

    verified = set()

    for a, b in candidates:
        set_a = sets[a]
        set_b = sets[b]

        jaccard = len(set_a & set_b) / len(set_a | set_b)

        if jaccard >= threshold:
            verified.add(tuple(sorted((a, b)))) # Keep IDs in lexicographic order.

    return verified


def containment_pairs(sets, split_by_id, threshold):
    """Check every train/eval pair; return (train_id, eval_id) tuples."""
    # Check directional train/evaluation overlap.
    # A contained evaluation passage can still have low Jaccard similarity.
    # Containment divides by the evaluation shingle-set size.

    result = set() # A set deduplicates the pairs.

    # Separate training and evaluation documents.
    train_ids = [
        doc_id for doc_id in sets # Iterating over a dictionary yields its keys.
        if split_by_id[doc_id] == "train"
    ]
    eval_ids = [
        doc_id for doc_id in sets
        if split_by_id[doc_id] == "eval"
    ]

    # Compute containment.
    for train_id in train_ids:
        for eval_id in eval_ids:
            # Retrieve the two shingle sets.
            train_set = sets[train_id]
            eval_set = sets[eval_id]

            # Compute containment.
            contaimment = len(train_set & eval_set) / len(eval_set)

            # Keep pairs meeting the threshold.
            if contaimment >= threshold:
                result.add((train_id, eval_id))

    return result


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
