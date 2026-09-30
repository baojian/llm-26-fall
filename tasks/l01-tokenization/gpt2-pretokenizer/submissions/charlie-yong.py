"""Submission for l01-tokenization/gpt2-pretokenizer. GitHub: Charlie-Yong."""

PREDICTIONS = {
    "don't stop": ["don", "'t", " stop"],
    "x2 + 3x = 0": ["x", "2", " +", " 3", "x", " =", " 0"],
    "It's 3.14": ["It", "'s", " 3", ".", "14"],
}

MY_CASES = [
    ("IT'S 3AM", ["IT", "'", "S", " 3", "AM"]),
    ("a_b c", ["a", "_", "b", " c"]),
]

NOTES = """把空格挂在后一个词的开头（例如把 “ world” 当作一个整体 token）而不是挂在前一个词的末尾，最大的好处是同一个词无论出现在句首还是句中都能复用同一条词表项。倘若反过来把空格附在前词末尾，“world” 与 “world ”、“world.”、“world,” 都得各自占一个词条，标点和空格的组合会让词表膨胀得相当厉害，几乎每个词都要为它后面可能跟随的字符各存一份。GPT-2 的做法本质上是把空格当作词的一部分而不是分隔符，BPE 在训练阶段就会自然把“前导空格 + 词”这种高频模式合并成单个 token，于是常见词的表示更稳定、序列也更短。换个角度看，这种挂法把位置信息编码进了 token 本身，模型不必再从上下文里推断某个词是不是位于句首，词表也能保持相对紧凑。"""

import re

_PATTERN = re.compile(
    "'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[^\W\d_]+"
    r"| ?\d+"
    r"| ?(?:(?![^\W\d_]|\d)\S)+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def solve(text: str) -> list[str]:
    return _PATTERN.findall(text)
