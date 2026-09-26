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
NOTES = """GPT-2 puts the space with the next word, so we get " world" instead of "world ". 
Spaces almost always come at the start of a word, so the vocabulary only needs one entry per word (" world"), 
and it does not need separate entries like "world", "world.", or "world,". 
This keeps the vocabulary small and lets BPE store the common form of each word as one compact token. 
If the space were attached to the previous word instead, each word would need one entry for every character that could follow it, 
which makes the vocabulary much bigger. That is why my solve leaves the space with the next word."""

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
