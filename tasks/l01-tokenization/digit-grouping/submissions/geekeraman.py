import re

PREDICTIONS = {
    "3.14159": ["3", ".", "141", "59"],
    "Room 101": ["Room ", "101"],
    "2024-09-16": ["202", "4", "-", "09", "-", "16"],
}

MY_CASES = [
    ("2²=4", ["2", "²=", "4"]),
    ("12\n3456", ["12", "\n", "345", "6"]),
]


def solve(text: str) -> list[str]:
    chunks = []
    for run in re.findall(r"\d+|\D+", text):
        if run[0].isdecimal():
            for i in range(0, len(run), 3):
                chunks.append(run[i:i + 3])
        else:
            chunks.append(run)
    return chunks


NOTES = """If every digit string of length 1 to k has its own token, the vocabulary needs 10 tokens for k=1 (0-9), 1110 for k=3 (1000+100+10), and 11110 for k=4 (10000+1000+100+10), because each extra digit makes the count ten times bigger. The string "123456789012" has 12 digits, so it needs 12 tokens when k=1, 4 tokens when k=3, and 3 tokens when k=4. This is the tradeoff: the vocabulary grows very fast, but the sequence gets only a little shorter. Going from 3 digits to 4 digits costs 10000 more entries and saves only 1 token. A pre-tokenization limit just says a chunk can never be longer than 3 digits, but it does not promise that one chunk is one BPE token, because BPE must also learn that merge from the training text."""
