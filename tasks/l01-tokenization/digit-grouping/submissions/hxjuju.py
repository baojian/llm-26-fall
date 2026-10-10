import re

PREDICTIONS = {
    "3.14159": ["3", ".", "141", "59"],
    "Room 101": ["Room ", "101"],
    "2024-09-16": ["202", "4", "-", "09", "-", "16"],
}


def solve(text: str) -> list[str]:
    chunks = []
    for run in re.findall(r"\d+|\D+", text):
        if run[0].isdecimal():
            chunks.extend(run[i:i + 3] for i in range(0, len(run), 3))
        else:
            chunks.append(run)
    return chunks


MY_CASES = [
    ("１２３４", ["１２３", "４"]),
    ("007\n\n42", ["007", "\n\n", "42"]),
]
NOTES = """For k = 1, 3, and 4, the numeric vocabulary needs 10, 10 + 100 + 1000 = 1110, and 11110 entries respectively, including leading zeros. The string 123456789012 takes 12, 4, and 3 tokens respectively if every allowed numeric chunk has its own token. Larger groups shorten sequences, but each extra digit increases the vocabulary by roughly a factor of ten, and rare long groups receive little training. A pre-tokenization limit only bounds chunk length: if a merge was never learned, BPE can still split a chunk such as 739 into 7 and 39."""
