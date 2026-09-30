import re

PREDICTIONS = {
    "Kimi K2 的 tokenizer": ["Kimi", " ", "K2", " ", "的", " ", "tokenizer"],
    "2026年9月16日": ["2026", "年", "9", "月", "16", "日"],
    "GPT-4o很强": ["GPT", "-", "4o", "很强"],
}

PATTERN = re.compile(r"[一-鿿]+|[A-Za-z0-9]+|\s+|[^一-鿿A-Za-z0-9\s]+")


def solve(text: str) -> list[str]:
    return PATTERN.findall(text)


MY_CASES = [
    ("测试_用例_01", ["测试", "_", "用例", "_", "01"]),
    ("é中　x", ["é", "中", "　", "x"]),
]
NOTES = """If BPE merges across Han-Latin boundaries, forms such as "用GPT" and "GPT用" can each occupy a separate vocabulary entry and shorten mixed-script token sequences. These mixed forms may be rare, receive fewer training updates, and reduce reuse of the shared "GPT" piece. Putting the Han branch first ensures that a run beginning with Han characters is matched there before a general letter branch can consume it. Excluding Han from other letter branches also stops a Latin-first run such as "GPT用" at the script boundary, so both examples split into reusable "用" and "GPT" pieces."""

