"""Sunny-hs submission for the optional GPT-2 pre-tokenizer activity."""

import re


PREDICTIONS = {
    "don't stop": ["don", "'t", " stop"],
    "x2 + 3x = 0": ["x", "2", " +", " 3", "x", " =", " 0"],
    "It's 3.14": ["It", "'s", " 3", ".", "14"],
}

MY_CASES = [
    ("WE'LL\tgo🙂", ["WE", "'", "LL", "\t", "go", "🙂"]),
    ("A  ²_１２!\n B", ["A", " ", " ²", "_", "１２", "!", "\n", " B"]),
]

NOTES = """Attaching an ASCII space to the following word lets the model learn a reusable distinction between a word at a boundary and the same letters after whitespace. Frequent forms such as " world" can then be compressed without creating separate tokens for every possible preceding word. If spaces were attached to the preceding word instead, many left words would need both trailing-space and no-space variants, coupling their tokens to what follows. The leading-space convention therefore makes word starts explicit while leaving punctuation and non-ASCII whitespace governed by their own branches."""


_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[^\W\d_]+"
    r"| ?\d+"
    r"| ?(?:(?![^\W\d_]|\d)\S)+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def solve(text: str) -> list[str]:
    """Return lossless chunks produced by the simplified GPT-2 regex."""
    chunks = [match.group(0) for match in _PATTERN.finditer(text)]
    if "".join(chunks) != text:
        raise RuntimeError("pre-tokenizer failed to consume the complete input")
    return chunks
