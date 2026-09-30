PREDICTIONS = {
    "3.14159": ["3", ".", "141", "59"],
    "Room 101": ["Room ", "101"],
    "2024-09-16": ["202", "4", "-", "09", "-", "16"],
}

MY_CASES = [
    ("00skdl23l00k", ["00", "skdl", "23", "l", "00", "k"]),
    ("AA0- 3n_jk09", ["AA", "0", "- ", "3", "n_jk", "09"]),
]

NOTES = """
Treating every complete number as one token would require a very large vocabulary,
because there are too many possible numbers. Splitting every digit separately keeps
the vocabulary small, but makes long numbers use many tokens. Grouping digits into
chunks of at most three is a compromise: it limits vocabulary growth while keeping
number sequences relatively short.
"""


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