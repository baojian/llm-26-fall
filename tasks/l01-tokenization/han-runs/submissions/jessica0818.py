PREDICTIONS = {
    "Kimi K2 的 tokenizer": ["Kimi", " ", "K2", " ", "的", " ", "tokenizer"],
    "2026年9月16日": ["2026", "年", "9", "月", "16", "日"],
    "GPT-4o很强": ["GPT", "-", "4o", "很强"],
}

MY_CASES = [
    ("用GPT用", ["用", "GPT", "用"]),
    ("Ａ1中🙂", ["Ａ", "1", "中", "🙂"]),
]

NOTES = """
Preventing merges across Han-Latin boundaries makes vocabulary items easier to reuse
across different mixed-script contexts, because a token such as GPT does not need
separate merged forms for 用GPT and GPT用. This may produce slightly more tokens in
mixed-script text, but avoids wasting vocabulary capacity on many language-specific
cross-boundary combinations. In 用GPT, putting Han first ensures 用 is captured as a
Han run before the Latin branch is considered. Excluding Han characters from other
letter branches also guarantees the same boundary in GPT用, so the two scripts cannot
accidentally merge into one pre-tokenization chunk.
"""


def category(ch: str) -> int:
    if "\u4e00" <= ch <= "\u9fff":
        return 1
    if ch.isascii() and ch.isalnum():
        return 2
    if ch.isspace():
        return 3
    return 4


def solve(text: str) -> list[str]:
    if not text:
        return []

    result = []
    start = 0
    current_category = category(text[0])

    for i in range(1, len(text)):
        new_category = category(text[i])

        if new_category != current_category:
            result.append(text[start:i])
            start = i
            current_category = new_category

    result.append(text[start:])
    return result