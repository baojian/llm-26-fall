PREDICTIONS = {
    "3.14159": ["3", ".", "141", "59"],
    "Room 101": ["Room ", "101"],
    "2024-09-16": ["202", "4", "-", "09", "-", "16"],
}

MY_CASES = [
    ("00skdl23l00k", ["00", "skdl", "23", "l", "00", "k"]),
    ("AA0- 3n_jk09", ["AA", "0", "- ", "3", "n_jk", "09"]),
]

# Teaching-team edit: translated or corrected NOTES during the authorized October 1 review.
NOTES = """Treating every complete number as one token would require a very large vocabulary, because there are too many possible numbers. Including every ASCII digit string of lengths 1 through k and leading zeros needs 10 entries for k=1, 10+100+1000=1110 for k=3, and 11110 for k=4. The string 123456789012 takes 12, 4, and 3 tokens in these three cases. Grouping digits into longer chunks shortens sequences at the cost of a larger vocabulary. A pre-tokenization limit only sets chunk boundaries, and BPE may split a chunk further if the required merges were not learned."""


def solve(text: str) -> list[str]:
    result = []
    i = 0

    while i < len(text):
        j = i

        if text[i].isdecimal():
            while j < len(text) and text[j].isdecimal():
                j += 1

            digits = text[i:j]

            for start in range(0, len(digits), 3):
                chunk = digits[start:start + 3]
                result.append(chunk)

        else:
            while j < len(text) and not text[j].isdecimal():
                j += 1

            result.append(text[i:j])

        i = j

    return result
