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
NOTES = """
LSH 的 Band 数量越多，找的越全，但是误报也越多：当 LSH 的 Band 数从 4 增到 8 时，找到的候选文档对从 38 对增到 62 对，召回率从 0.95 提高到 1.00，但是精确率从 1.00 降到了 0.645；例如，B=4 时漏掉了eval-00 和 train-00 这对相似文档，而 B=8 时虽然找全了相似文档，却多出了 22 对不够相似的候选文档。
Jaccard看整体相似度，Containment 看评测内容是否被覆盖：区别主要在于分母。一篇很长的训练文档与一篇较短的评测文档整体上不太相似（Jaccard 低），也可能包含评测文档的全部内容，因此不能只依靠 LSH 找到的候选对来检查数据重叠。
去掉可能重叠的数据后，模型A、B的排名可能反转：在困惑度实验中，使用全部评测数据时，Run A 的 PPL = 5.658，Run B 的 PPL = 8.000，A 的结果更好；但是去掉被标记为重叠的评测数据后，Run A 的 PPL = 22.627，Run B 的 PPL = 11.314，B 更好，说明数据重叠可能影响模型的评测结果和排名。
文本重叠不等于一定发生了数据泄露：不同文档可能使用相同的常见句子或固定模版，检测到文本相似也不能证明模型在训练过程中一定见过一些评测数据。
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

    for i in range(len(tokens) - n + 1): # 如果 len(tokens)<n，直接返回空集合
        gram = tuple(tokens[i: i + n])
        result.add(gram)

    return result


def signature(shingle_set, num_hashes):
    """Return the seeded minima for a nonempty set, in seed order."""
    
    # MinHash：把 N-gram 集合表示成可比较的短签名
    # 用 MinHash 估计 Jaccard （ P(MinHash相同) = J（A,B））
    # 单次 MinHash 随机性强，因此使用不同的 seed 来模拟多种排序，随着 Hash 次数的增加，签名相同位置的比例会更稳定地估计 Jaccard
    result = []

    for seed in range(num_hashes):
        minimum = min(rank(g, seed) for g in shingle_set)
        result.append(minimum)

    return result


def lsh_candidates(signatures, num_bands):
    """Return a set of sorted ID tuples matching at least one same-index band."""
    # LSH（局部敏感哈希）：快速查找相似数据的方法（避免两两比较）【只是找出候选，不是最终相似度判定方法】————让相似数据有较高概率被映射到同一个哈希桶
    # Band 的编号必须参与匹配（防止错位（不是同一个 seed 下）比对）
    # 把有相同 Band key 的文档放进同一个 bucket，只需遍历每个 bucket 内部的文档（不需要亮亮比较文档）

    # 处理 Shingle 为空的情况
    if not signatures:
        return set()

    num_hashes = len(next(iter(signatures.values()))) # next() 从迭代器 iter() 中逐个取出下一个元素
    rows_per_band = num_hashes // num_bands

    buckets = {}

    for doc_id, sig in signatures.items():
        for band_index in range(num_bands):
            start = band_index * rows_per_band
            end = start + rows_per_band

            band = tuple(sig[start: end]) # 字典的 key 必须可哈希（转换为元组），list 不能直接作为字典的 key
            key = (band_index, band)

            buckets.setdefault(key, []).append(doc_id) # 把文档 id 放到对应的 bucket 里

    candidates = set()

    for ids in buckets.values():
        for a, b in combinations(sorted(ids), 2): # 同一个 bucket 内的所有文档两两配对，都是可能候选
            candidates.add((a, b))

    return candidates



def verify_pairs(candidates, sets, threshold):
    """Return candidate pairs with exact Jaccard >= threshold; sets are nonempty."""
    # 验证用 LSH 选出的候选对的相似度
    
    verified = set()

    for a, b in candidates:
        set_a = sets[a]
        set_b = sets[b]

        jaccard = len(set_a & set_b) / len(set_a | set_b)

        if jaccard >= threshold:
            verified.add(tuple(sorted((a, b)))) # 保证文档对中的 id 按字典序排序

    return verified


def containment_pairs(sets, split_by_id, threshold):
    """Check every train/eval pair; return (train_id, eval_id) tuples."""
    # 以上是相似文档检测，这个用于评测污染检测
    # jaccard 局限：如果训练文档很长，评测文档很短，即使训练文档包含了整个评测频段，jaccard也可能很低
    # containment（有方向）：分母是评测文档的 Shingle 集合大小，与训练文档大小无关

    result = set() # 创建空集合（自动去重）

    # 区分训练与评测文档
    train_ids = [
        doc_id for doc_id in sets # 遍历字典时，默认遍历的是 key，而不是 value
        if split_by_id[doc_id] == "train"
    ]
    eval_ids = [
        doc_id for doc_id in sets
        if split_by_id[doc_id] == "eval"
    ]

    # 计算 containment
    for train_id in train_ids:
        for eval_id in eval_ids:
            # 取对应训练文档和评测文档
            train_set = sets[train_id]
            eval_set = sets[eval_id]

            # 计算 containment
            contaimment = len(train_set & eval_set) / len(eval_set)

            # 判断并保存
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
