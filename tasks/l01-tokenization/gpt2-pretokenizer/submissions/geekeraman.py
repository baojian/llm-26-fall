import re

PREDICTIONS = {
    "don't stop": ["don", "'t", " stop"],
    "x2 + 3x = 0": ["x", "2", " +", " 3", "x", " =", " 0"],
    "It's 3.14": ["It", "'s", " 3", ".", "14"],
}

PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"          # A lowercase contraction suffix: 's, 't, 're, 've, 'm, 'll, 'd.
    r"| ?[^\W\d_]+"                     # An optional ASCII space followed by a letter-like run ([^\W\d_]+).
    r"| ?\d+"                           # An optional ASCII space followed by decimal digits (\d+).

    r"| ?(?:(?![^\W\d_]|\d)\S)+"        # An optional ASCII space followed by other non-whitespace characters: (?:(?![^\W\d_]|\d)\S)+. This includes punctuation, symbols, and _.
    r"|\s+(?!\S)"                       # Whitespace matching \s+(?!\S); backtracking may leave one space for the following word.
    r"|\s+"                             # Any remaining whitespace run (\s+).
)

def solve(text: str) -> list[str]:
    return PATTERN.findall(text)

MY_CASES = [
    ("I'd like 2 llm courses!", ["I", "'d", " like", " 2", " llm", " courses", "!"]),
    ("Fall is  beautiful, isn't it?", ["Fall", " is", " ", " beautiful", ",", " isn", "'t", " it", "?"]),
]

NOTES = """Attaching the space to the following word stops one word from being saved two times in the vocabulary. For example, "I like this llm course." becomes "I", " like", " this", " llm", " course", "." and "This course is fine." becomes "This", " course", " is", " fine", ".", so "course" is the same token " course" in both sentences. If the space were attached to the previous word instead, "course" at the end of the first sentence would be "course" but in the second sentence it would be "course ", so the same word would be recognized as 2 different tokens. Because a word is almost always preceded by a space but often followed by punctuation, attaching the space to the following word keeps one token per word, so the vocabulary is smaller and the model learns each word from all of its examples."""
