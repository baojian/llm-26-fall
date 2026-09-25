import re

PREDICTIONS = {
    "don't stop": ["don", "'t", " stop"],
    "x2 + 3x = 0": ["x", "2", " +", " 3", "x", " =", " 0"],
    "It's 3.14": ["It", "'s", " 3", ".", "14"],
}

MY_CASES = [
    ("A  B", ["A", " ", " B"]),
    ("we'll go", ["we", "'ll", " go"]),
]

NOTES = """
Attaching a space to the following word lets tokens such as " world" be reused
whenever that word appears after a normal word boundary. This can improve
compression because the tokenizer can learn a useful distinction between a word
at the beginning of text and the same word after a space. If spaces were instead
attached to the preceding word, many tokens would need separate variants depending
on what whitespace follows them, making vocabulary reuse less efficient.
"""


# The six rules are kept separately because their ORDER matters.
PATTERNS = [
    # 1. Lowercase contraction suffixes
    re.compile(r"'(?:s|t|re|ve|m|ll|d)"),

    # 2. Optional ASCII space + letter-like characters
    re.compile(r" ?[^\W\d_]+"),

    # 3. Optional ASCII space + decimal digits
    re.compile(r" ?\d+"),

    # 4. Optional ASCII space + punctuation / symbols / underscore
    re.compile(r" ?(?:(?![^\W\d_]|\d)\S)+"),

    # 5. Whitespace, possibly backtracking so that one space
    #    can belong to the following word
    re.compile(r"\s+(?!\S)"),

    # 6. Any remaining whitespace
    re.compile(r"\s+"),
]


def solve(text: str) -> list[str]:
    chunks = []
    pos = 0

    while pos < len(text):
        for pattern in PATTERNS:
            match = pattern.match(text, pos)

            if match is not None:
                chunks.append(match.group())
                pos = match.end()
                break
        else:
            # In principle the six rules should cover every character.
            # This guard prevents an infinite loop if something unexpected
            # slips through.
            chunks.append(text[pos])
            pos += 1

    return chunks