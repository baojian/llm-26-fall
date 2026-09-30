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

NOTES = """把空格接到后面的词上，可以让句首的 world 和句中的 " world" 成为不同但可复用的模式，模型能直接学习一个词在有前导空格时通常表示新词边界。这样常见词不用为每一种前面标点或前一个词组合都造词表项，只需要学习带空格和不带空格的常见形态。如果改成把空格接到前一个词，词表会更容易出现 "hello "、"world " 这种尾随空格版本，行尾、标点前和多空格情况会制造更多稀疏变体。前导空格也让 BPE 合并更自然地发生在一个词内部，而不是跨过词后的空白。"""

_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:(?![^\W\d_]|\d)\S)+|\s+(?!\S)|\s+"
)


def solve(text: str) -> list[str]:
    return [match.group(0) for match in _PATTERN.finditer(text)]
