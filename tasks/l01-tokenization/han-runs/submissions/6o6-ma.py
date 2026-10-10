PREDICTIONS = {
    "Kimi K2 的 tokenizer": ["Kimi", " ", "K2", " ", "的", " ", "tokenizer"],
    "2026年9月16日": ["2026", "年", "9", "月", "16", "日"],
    "GPT-4o很强": ["GPT", "-", "4o", "很强"],
}

MY_CASES = [
    ("边界ABC中文é", ["边界", "ABC", "中文", "é"]),
    ("𠀀A中９", ["𠀀", "A", "中", "９"]),
]

# Teaching-team edit: translated or corrected NOTES during the authorized October 1 review.
NOTES = """Allowing Han and Latin characters in the same pre-tokenization chunk could produce dedicated tokens for mixed forms such as 用GPT and GPT用. Those combinations can shorten mixed-script sequences, but may be rare and reduce reuse of the GPT piece across Chinese contexts. Putting the Han branch first separates 用 from GPT when matching starts at the Han character in 用GPT. Excluding Han from other letter branches is also needed to stop a broader Latin-first run from consuming GPT用 across the boundary. These boundaries can increase the number of mixed-script tokens while improving reuse of Chinese and Latin pieces."""


def _category(char: str) -> int:
    if "\u4e00" <= char <= "\u9fff":
        return 1
    if ("A" <= char <= "Z") or ("a" <= char <= "z") or ("0" <= char <= "9"):
        return 2
    if char.isspace():
        return 3
    return 4


def solve(text: str) -> list[str]:
    if not text:
        return []

    chunks = []
    start = 0
    current = _category(text[0])
    for index, char in enumerate(text[1:], start=1):
        category = _category(char)
        if category != current:
            chunks.append(text[start:index])
            start = index
            current = category

    chunks.append(text[start:])
    return chunks
