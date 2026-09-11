"""Sample submission for tasks/example/word-count (octocat is GitHub's mascot)."""

import re

WORD = re.compile(r"[A-Za-z0-9']+")


def solve(text: str) -> list[str]:
    return [word.lower() for word in WORD.findall(text)]
