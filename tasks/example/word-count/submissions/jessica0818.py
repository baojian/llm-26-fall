import re

WORD = re.compile(r"[A-Za-z0-9']+")

PREDICTIONS = {
    "Don't stop.": ["don't", "stop"],
    "x2 + 3x = 0": ["x2", "3x", "0"],
    "naïve café": ["na", "ve", "caf"],
}

MY_CASES = [
    ("I'll go there ", ["i'll", "go", "there"]),
    ("My account name is je_ss-ica",
     ["my", "account", "name", "is", "je", "ss", "ica"]),
]

NOTES = """
Keeping the leading space helps the model know where a new word begins.
It preserves word-boundary information without using a separate space token.
The cost is that the same word may need two vocabulary entries, one with a
leading space and one without, so the vocabulary is used less efficiently.
"""

def solve(text: str) -> list[str]:
    return [word.lower() for word in WORD.findall(text)]