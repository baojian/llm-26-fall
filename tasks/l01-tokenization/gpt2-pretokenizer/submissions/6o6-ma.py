import re


PREDICTIONS = {
    "don't stop": ["don", "'t", " stop"],
    "x2 + 3x = 0": ["x", "2", " +", " 3", "x", " =", " 0"],
    "It's 3.14": ["It", "'s", " 3", ".", "14"],
}

MY_CASES = [
    ("Hi,  Bob", ["Hi", ",", " ", " Bob"]),
    ("we'll_go", ["we", "'ll", "_", "go"]),
]

# Teaching-team edit: translated or corrected NOTES during the authorized October 1 review.
NOTES = """Attaching a space to the following word lets sentence-initial world and space-prefixed " world" form distinct, reusable patterns. The leading space provides a word-boundary cue, and common forms can be reused across different preceding words or punctuation. Attaching the space to the preceding word would instead favor trailing-space forms such as "hello " and "world ", changing the boundary variants learned by the vocabulary. The pre-tokenization boundaries constrain which pieces BPE can merge, but neither convention guarantees one token per word or a fixed vocabulary-size advantage."""

_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:(?![^\W\d_]|\d)\S)+|\s+(?!\S)|\s+"
)


def solve(text: str) -> list[str]:
    return [match.group(0) for match in _PATTERN.finditer(text)]
