PREDICTIONS = {
    "Kimi K2 的 tokenizer": ["Kimi", " ", "K2", " ", "的", " ", "tokenizer"],
    "2026年9月16日": ["2026", "年", "9", "月", "16", "日"],
    "GPT-4o很强": ["GPT", "-", "4o", "很强"],
}

MY_CASES = [
    ("边界ABC中文é", ["边界", "ABC", "中文", "é"]),
    ("𠀀A中９", ["𠀀", "A", "中", "９"]),
]

NOTES = """如果允许 Han 和 Latin 在同一个预分词块里随意合并，像 用GPT 或 GPT用 这样的混合脚本片段可能学成专门 token，词表会被具体搭配占用，GPT 这个英文片段在不同中文上下文里的复用也会变差。先匹配 Han run 可以把 用 和 GPT 分开，GPT用 也会在 Latin run 后遇到 Han 边界而停止。Kimi 的模式还要把 Han 从其他字母分支排除，否则 Han 可能被更宽的 letter 类吞掉，重新产生跨脚本合并。这样的边界通常会让混合文本的 token 数略增，但换来更稳定的中文、英文子词复用。"""


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
