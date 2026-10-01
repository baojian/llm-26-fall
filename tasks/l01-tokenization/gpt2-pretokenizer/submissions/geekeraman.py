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

# Teaching-team edit: translated or corrected NOTES during the authorized October 1 review.
NOTES = """Attaching a leading space to the following word makes a form such as " course" reusable in both "I like this llm course" and "This course is fine". The space gives the model a word-start cue, but a sentence-initial unspaced "course" can still have a different representation. With trailing spaces, the vocabulary could instead learn "course " alongside "course", so the boundary information would be carried after the word. This changes which variants are learned rather than guaranteeing one vocabulary entry per word or a smaller vocabulary, and BPE may still split each pre-tokenization chunk into several tokens."""
