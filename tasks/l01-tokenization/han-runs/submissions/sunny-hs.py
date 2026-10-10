"""Sunny-hs submission for the optional Han-runs activity."""

PREDICTIONS = {
    "Kimi K2 的 tokenizer": ["Kimi", " ", "K2", " ", "的", " ", "tokenizer"],
    "2026年9月16日": ["2026", "年", "9", "月", "16", "日"],
    "GPT-4o很强": ["GPT", "-", "4o", "很强"],
}

MY_CASES = [
    ("A9中文\té２B", ["A9", "中文", "\t", "é２", "B"]),
    ("㐁🙂Ａ中\n9", ["㐁🙂Ａ", "中", "\n", "9"]),
]

NOTES = """Allowing merges across Han–Latin boundaries could create script-order-specific vocabulary items such as 用GPT and GPT用, reducing reuse and making mixed-script token lengths less predictable. Putting the Han branch first ensures that every U+4E00–U+9FFF character is claimed before broader alternatives can consume it. Excluding Han from the other letter branches makes both the Han-to-Latin boundary in 用GPT and the Latin-to-Han boundary in GPT用 explicit. Together these choices let the same GPT and Han pieces be reused in either order instead of memorizing each mixed-script combination."""


def _category(char: str) -> int:
    codepoint = ord(char)
    if 0x4E00 <= codepoint <= 0x9FFF:
        return 0
    if "A" <= char <= "Z" or "a" <= char <= "z" or "0" <= char <= "9":
        return 1
    if char.isspace():
        return 2
    return 3


def solve(text: str) -> list[str]:
    """Return maximal runs from the task's four disjoint categories."""
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    current = _category(text[0])
    for index in range(1, len(text)):
        category = _category(text[index])
        if category != current:
            chunks.append(text[start:index])
            start = index
            current = category
    chunks.append(text[start:])
    return chunks
