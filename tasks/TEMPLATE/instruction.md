# {{TITLE}}

**Lecture:** {{LECTURE}} · **Difficulty:** easy · **Time:** about 45 minutes ·
**Deadline:** Tuesday 23:59

## Goal

REPLACE: One sentence saying what the student will be able to do afterwards.

## What you submit

One file: `submissions/<your-github-username>.py` containing a function

```python
def solve(text: str) -> list[str]:
    ...
```

REPLACE: Describe the input and the output precisely (types, order, edge cases).

## Examples

| Input | Output |
| :--- | :--- |
| REPLACE `"..."` | REPLACE `[...]` |
| REPLACE `""` | REPLACE `[]` |

## How it is checked

`tests/test_task.py` calls your `solve` on the examples above and on a few
similar cases. Run it yourself before opening the PR:

```sh
uv run python scripts/tasks.py check tasks/{{LECTURE}}/{{SLUG}} <username>
```

## Rules

- Standard library only unless the instruction says otherwise.
- Use your own words and code; discussing the approach with classmates is fine.
- PR title `{{LECTURE}}/{{SLUG}}: <username>`, body `Related to #<issue>`.

## Why this matters

REPLACE: One or two sentences linking the task to the lecture.
