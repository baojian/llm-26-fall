"""Sample submission for tasks/example/word-count (octocat is GitHub's mascot)."""

import re

WORD = re.compile(r"[A-Za-z0-9']+")

PREDICTIONS = {
    "Don't stop.": ["don't", "stop"],
    "x2 + 3x = 0": ["x2", "3x", "0"],
    "naïve café": ["na", "ve", "caf"],
}
MY_CASES = [
    ("rock'n'roll", ["rock'n'roll"]),
    ("A--B__C", ["a", "b", "c"]),
]
NOTES = """
Keeping the leading space lets the model learn that " hard" begins a word while
"hard" continues one, so word boundaries survive without a separate space token,
and the sequence gets shorter because the space costs nothing extra. The price is
vocabulary: many words need two entries, one with and one without the space, so
a 50k vocabulary holds fewer distinct words than it could otherwise.
"""


def solve(text: str) -> list[str]:
    return [word.lower() for word in WORD.findall(text)]
