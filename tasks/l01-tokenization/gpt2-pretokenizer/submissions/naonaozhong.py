import re

PREDICTIONS = {
    "don't stop": ["don", "'t", " stop"],
    "x2 + 3x = 0": ["x", "2", " +", " 3", "x", " =", " 0"],
    "It's 3.14": ["It", "'s", " 3", ".", "14"],
}
MY_CASES = [
    ("it's 42", ["it", "'s", " 42"]),
    ("Hello, world!", ["Hello", ",", " world", "!"]),
]
NOTES = """GPT-2 usually attaches a space to the next word-like chunk, so " world" carries word-boundary information. BPE may learn both "world" and " world", and either can still split into smaller tokens. Attaching the space to the previous chunk would instead favor forms like "world ", shifting the boundary information to the preceding token."""

_PAT = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[^\W\d_]+"
    r"| ?\d+"
    r"| ?(?:(?![^\W\d_]|\d)\S)+"
    r"|\s+(?!\S)"
    r"|\s+"
)

def solve(text: str) -> list[str]:
    return re.findall(_PAT, text)
