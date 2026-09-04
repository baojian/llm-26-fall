# Workspace

This folder is yours. Put your notes, experiments, exercise solutions, and
scratch code here rather than editing the course files elsewhere in the
repository.

Everything inside `workspace/` except this README is ignored by git, so:

- `git pull` never conflicts with your files when new course material lands.
- Your work stays out of any pull request you open against the course repo.
- `git status` stays clean while you experiment.

## Working on course files

Copy the file into `workspace/` first and edit the copy. If you find a bug or
want to improve the original, edit it in place and open a pull request.

## Running code

Run from the repository root so the shared uv environment is used:

```bash
uv run python workspace/my_script.py
```

## Keeping your own history

If you want version control for your work, start a separate repository inside
this folder:

```bash
cd workspace && git init
```

The course repository ignores everything here, so the two histories stay
independent.
