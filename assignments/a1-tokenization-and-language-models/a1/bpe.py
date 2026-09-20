"""A1 Part 1: GPT-2 pre-tokenizer and byte-level BPE (train, encode, decode).

Standard library only. See README.md for the exact rules and the tie-break.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

# GPT-2's pattern in Python `re` syntax. Python has no \p{L} / \p{N}: a letter is [^\W\d_]
# and a digit is \d. The "other" branch must then be "non-space that is neither letter nor
# digit" written as a lookahead, because [^\s\w] would silently drop underscores (\w includes
# "_") and break the round trip. Every character lands in exactly one branch. Known difference
# from tiktoken: numerals that are not decimal digits (², ½, Ⅻ) count as \w but not \d in
# Python, so they join the letter branch here, while GPT-2's \p{N} puts them with digits.
GPT2_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"  # contraction suffixes
    r"| ?[^\W\d_]+"  # optional space + letters
    r"| ?\d+"  # optional space + decimal digits
    r"| ?(?:(?![^\W\d_]|\d)\S)+"  # optional space + run of other non-space characters (punctuation, _, symbols)
    r"|\s+(?!\S)"  # whitespace not followed by a non-space (the run keeps its last space for the next word)
    r"|\s+"  # any remaining whitespace
)


def pretokenize(text: str) -> list[str]:
    """Chunk text with the GPT-2 rules; chunks concatenate back to the input."""
    # YOUR CODE HERE
    raise NotImplementedError


def _pair_counts(chunks: dict[tuple[bytes, ...], int]) -> Counter:
    counts: Counter = Counter()
    for symbols, n in chunks.items():
        for a, b in zip(symbols, symbols[1:]):
            counts[(a, b)] += n
    return counts


def _merge_symbols(symbols: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    out: list[bytes] = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train_bpe(text: str, vocab_size: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Learn byte-level BPE merges inside pre-tokenized chunks.

    Start from the 256 single-byte tokens. Repeat until the vocabulary has
    ``vocab_size`` entries or no pair remains: maintain adjacent-pair counts
    weighted by chunk frequency, merge the most frequent pair, and break ties
    by the lexicographically greater pair of byte strings. Update counts only
    for chunks containing the selected pair; all other counts stay valid.
    Returns (vocab: id -> bytes, merges in learned order).
    """
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []
    # YOUR CODE HERE
    raise NotImplementedError
    return vocab, merges


def encode(text: str, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]) -> list[int]:
    """Apply the merges in learned order inside each chunk; return token ids."""
    # YOUR CODE HERE
    raise NotImplementedError


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """Concatenate the bytes of the ids and decode UTF-8 (errors="replace")."""
    # YOUR CODE HERE
    raise NotImplementedError
