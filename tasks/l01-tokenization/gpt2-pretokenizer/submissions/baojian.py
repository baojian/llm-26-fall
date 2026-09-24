"""Intentionally incorrect instructor demo for the public task feedback.

The predictions describe the required behavior; solve deliberately disagrees.
This example is for inspecting failed checks, not for merging as a solution.
"""

import re


PREDICTIONS = {
    "don't stop": ["don", "'t", " stop"],
    "x2 + 3x = 0": ["x", "2", " +", " 3", "x", " =", " 0"],
    "It's 3.14": ["It", "'s", " 3", ".", "14"],
}

MY_CASES = [
    ("Café_42", ["Café", "_", "42"]),
    ("we'll wait", ["we", "'ll", " wait"]),
]

NOTES = """Attaching a space to the next word lets BPE learn a common form such
as ' world', combining a word boundary with letters in a single token when that
merge is learned. This can reduce token counts without allowing merges across
arbitrary word boundaries. Attaching the space to the previous word would
instead create entries such as 'Hello ' and shift the boundary marker to the
word before it. Both choices create boundary-sensitive variants, but leading
spaces distinguish a word after a space from the same spelling at the beginning
of a text."""

# Intentional bug: the word, digit, and punctuation rules omit the optional
# leading ASCII space, so spaces become separate chunks instead of attaching.
_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"|[^\W\d_]+"
    r"|\d+"
    r"|(?:(?![^\W\d_]|\d)\S)+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def solve(text: str) -> list[str]:
    """Preserve all characters but deliberately use incorrect space boundaries."""
    return _PATTERN.findall(text)
