import re

PREDICTIONS = {
    "don't stop": ["don", "'t", " stop"],
    "x2 + 3x = 0": ["x", "2", " +", " 3", "x", " =", " 0"],
    "It's 3.14": ["It", "'s", " 3", ".", "14"],
}

PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d"""
    r"""| ?[^\W\d_]+"""
    r"""| ?\d+"""
    r"""| ?(?:(?![^\W\d_]|\d)\S)+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)


def solve(text: str) -> list[str]:
    return PATTERN.findall(text)


MY_CASES = [
    ("I'M here   now", ["I", "'", "M", " here", "  ", " now"]),
    ("end.  \n", ["end", ".", "  \n"]),
]
NOTES = """Attaching a space to the following word gives a recurring form such as " world" that can be reused at different positions in a sentence. The leading space marks the start of a word, giving the model a useful boundary cue when predicting what comes next. With trailing spaces, "world " and "world" would instead distinguish a word followed by a space from one followed by punctuation or the end of the text. This would change which boundary variants occupy vocabulary entries; it does not imply a fixed doubling of vocabulary size, and pre-tokenization chunks may still split into multiple BPE tokens."""
