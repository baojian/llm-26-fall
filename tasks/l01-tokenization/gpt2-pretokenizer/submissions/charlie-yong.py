"""Submission for l01-tokenization/gpt2-pretokenizer. GitHub: Charlie-Yong."""

PREDICTIONS = {
    "don't stop": ["don", "'t", " stop"],
    "x2 + 3x = 0": ["x", "2", " +", " 3", "x", " =", " 0"],
    "It's 3.14": ["It", "'s", " 3", ".", "14"],
}

MY_CASES = [
    ("IT'S 3AM", ["IT", "'", "S", " 3", "AM"]),
    ("a_b c", ["a", "_", "b", " c"]),
]

# Teaching-team edit: translated or corrected NOTES during the authorized October 1 review.
NOTES = """Attaching a leading ASCII space to a word gives the model a reusable word-start cue in forms such as " world". BPE may learn this frequent form, but both "world" and " world" can occur in the vocabulary, and a pre-tokenization chunk may still become several tokens. Using trailing spaces would instead favor variants such as "world ", marking a boundary after the word. This changes the space-bearing forms the vocabulary learns; it neither requires a separate token for every following punctuation mark nor guarantees a smaller vocabulary or one token per word."""

import re

_PATTERN = re.compile(
    "'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[^\W\d_]+"
    r"| ?\d+"
    r"| ?(?:(?![^\W\d_]|\d)\S)+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def solve(text: str) -> list[str]:
    return _PATTERN.findall(text)
