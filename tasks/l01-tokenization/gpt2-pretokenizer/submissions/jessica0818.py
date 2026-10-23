import re


PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[^\W\d_]+"
    r"| ?\d+"
    r"| ?(?:(?![^\W\d_]|\d)\S)+"
    r"|\s+(?!\S)"
    r"|\s+"
)


PREDICTIONS = {
    "don't stop": ["don", "'t", " stop"],
    "x2 + 3x = 0": ["x", "2", " +", " 3", "x", " =", " 0"],
    "It's 3.14": ["It", "'s", " 3", ".", "14"],
}


MY_CASES = [
    ("I'll go", ["I", "'ll", " go"]),
    ("abc__XYZ", ["abc", "__", "XYZ"]),
]


NOTES = """
Attaching a leading space to the following word preserves information about where a
new word begins while avoiding a separate space token in many common cases. This can
improve compression because forms such as " world" can be learned as useful vocabulary
items. If spaces were attached to the preceding word instead, the vocabulary would
need many variants of words depending on whether and what kind of spacing followed them.
Using leading spaces therefore makes word-boundary patterns more reusable across contexts.
"""


def solve(text: str) -> list[str]:
    return PATTERN.findall(text)
