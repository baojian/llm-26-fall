PREDICTIONS = {
    "3.14159": ["3", ".", "141", "59"],
    "Room 101": ["Room ", "101"],
    "2024-09-16": ["202", "4", "-", "09", "-", "16"],
}

MY_CASES = [
    ("ID００１2345 end", ["ID", "００１", "234", "5", " end"]),
    ("a12\n3456b", ["a", "12", "\n", "345", "6", "b"]),
]

NOTES = """如果每个长度 1 到 k 的 ASCII 数字串都单独进词表，并且允许前导零，k=1 需要 10 个，k=3 需要 10+100+1000=1110 个，k=4 则需要 11110 个。字符串 123456789012 在 k=1 时要 12 个数字 token，在 k=3 时可以是 4 个三位块，在 k=4 时可以是 3 个四位块。更大的 k 会缩短序列，但词表会快速膨胀，还会让模型为很多罕见数字组合付出容量。预分词最多三位只规定了 BPE 可以合并的边界，训练数据里没有学到的块仍可能被继续拆开，所以它不保证每个 chunk 都是一个 BPE token。"""


def solve(text: str) -> list[str]:
    chunks = []
    i = 0
    while i < len(text):
        is_digit = text[i].isdecimal()
        j = i + 1
        while j < len(text) and text[j].isdecimal() == is_digit:
            j += 1

        run = text[i:j]
        if is_digit:
            chunks.extend(run[k : k + 3] for k in range(0, len(run), 3))
        else:
            chunks.append(run)
        i = j

    return chunks
