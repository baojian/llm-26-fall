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
    # Repeated / short text (n=1): repeated tokens collapse to one set element, and
    # the single-token evaluation document is fully contained in the training doc.
    (
        {
            "documents": [
                {"id": "t", "split": "train", "tokens": ["repeat", "repeat", "repeat", "other"]},
                {"id": "s", "split": "train", "tokens": ["tiny"]},
                {"id": "e", "split": "eval", "tokens": ["repeat"]},
                {"id": "f", "split": "eval", "tokens": ["other", "repeat"]},
            ],
            "n": 1,
            "num_hashes": 4,
            "num_bands": 2,
            "jaccard_threshold": 1.0,
            "containment_threshold": 1.0,
        },
        [["t", "e"], ["t", "f"]],
    ),
    # Directional containment: the short eval passage is fully inside the longer train
    # doc, so containment flags it even though whole-document Jaccard stays low.
    (
        {
            "documents": [
                {"id": "long", "split": "train", "tokens": ["p", "alpha", "beta", "gamma", "q", "r"]},
                {"id": "short", "split": "eval", "tokens": ["alpha", "beta", "gamma"]},
                {"id": "unrelated", "split": "eval", "tokens": ["zzz", "yyy", "xxx"]},
            ],
            "n": 1,
            "num_hashes": 8,
            "num_bands": 2,
            "jaccard_threshold": 0.9,
            "containment_threshold": 1.0,
        },
        [["long", "short"]],
    ),
]
NOTES = (
    "The four-band run misses two above-threshold pairs, eval-00/train-00 and "
    "eval-13/train-44, because those two documents never land in the same band bucket, "
    "which is exactly how LSH trades recall for speed; raising to eight bands recovers "
    "both (recall 0.950 -> 1.000) but pushes 22 false candidates in, so precision drops "
    "from 1.000 to 0.645 and every extra candidate still has to be checked by exact "
    "Jaccard before anything is called contamination. Jaccard and containment answer "
    "different questions: Jaccard |A n B|/|A u B| is symmetric and small when the sets "
    "differ in size, while containment |S_e n S_t|/|S_e| is directional and reaches 1.0 "
    "whenever a short evaluation passage sits wholly inside a long training document, "
    "which is why 1,488 containment flags fall outside the candidate set at four bands. "
    "On the constructed audit both models look close on the full set, but run A collapses "
    "from perplexity 5.657 to 22.627 once the flagged eval-copy is removed and run B only "
    "moves from 8.0 to 11.314, so the clean subset reverses the ranking and shows the "
    "flag was propping up the model that memorized it. A flag still does not prove a "
    "contiguous copy or training exposure, since shared boilerplate and scattered "
    "overlapping trigrams both trip containment without the whole passage ever appearing "
    "in order. I corrected one prediction: I had expected the contained_passage case to "
    "need a low Jaccard, but the run showed the two trigram sets already reach Jaccard 0.5 "
    "with containment 1.0, so a single contained passage can satisfy both at once."
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
    return [
        min(rank(shingle, seed) for shingle in shingle_set)
        for seed in range(num_hashes)
    ]


def lsh_candidates(signatures, num_bands):
    """Return a set of sorted ID tuples matching at least one same-index band."""
    ids = sorted(signatures)
    if not ids:
        return set()
    rows = len(next(iter(signatures.values()))) // num_bands
    candidates = set()
    for band in range(num_bands):
        buckets = {}
        for doc_id in ids:
            key = (band, tuple(signatures[doc_id][band * rows:(band + 1) * rows]))
            buckets.setdefault(key, []).append(doc_id)
        for members in buckets.values():
            for left, right in combinations(members, 2):
                candidates.add(tuple(sorted((left, right))))
    return candidates


def verify_pairs(candidates, sets, threshold):
    """Return candidate pairs with exact Jaccard >= threshold; sets are nonempty."""
    verified = set()
    for left, right in candidates:
        union = len(sets[left] | sets[right])
        if union and len(sets[left] & sets[right]) / union >= threshold:
            verified.add((left, right))
    return verified


def containment_pairs(sets, split_by_id, threshold):
    """Check every train/eval pair; return (train_id, eval_id) tuples."""
    train_ids = sorted(key for key, split in split_by_id.items() if split == "train" and key in sets)
    eval_ids = sorted(key for key, split in split_by_id.items() if split == "eval" and key in sets)
    pairs = set()
    for train_id in train_ids:
        train_set = sets[train_id]
        for eval_id in eval_ids:
            eval_set = sets[eval_id]
            if len(eval_set) and len(eval_set & train_set) / len(eval_set) >= threshold:
                pairs.add((train_id, eval_id))
    return pairs


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
