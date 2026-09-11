# Tasks

Small, self-checking exercises that anyone in the class can do. Each task is
one folder; each student submits **one file** into that folder's
`submissions/` directory through a pull request. A checker runs the same
tests on every submission, so you know you are done before you open the PR.

```text
tasks/
  README.md                      this page
  TEMPLATE/                      copy this to create a task
  example/word-count/            a complete, tiny example
  l02-ngram/<slug>/              real tasks, grouped by lecture label
    task.toml                    id, title, deadline, what to submit
    instruction.md               the task: goal, interface, examples, how it is checked
    tests/test_task.py           the checker, run on every file in submissions/
    submissions/<username>.py    one file per student
```

[PROGRESS.md](PROGRESS.md) is the class progress board: tasks passed,
correct rate, participation, merged PRs, and badges per student. It is
regenerated after each merge batch with `uv run python scripts/progress.py`.

Task issues on GitHub carry the same lecture label as the folder
(`l02-ngram`, `l03-embeddings`, …) plus the type label `task`. The full
rules, timeline, and labels are in
[docs/participation-workflow.md](../docs/participation-workflow.md).

## Students: do a task

1. Pick an open issue labelled `task` and comment "I'll take this".
2. Read the task's `instruction.md`. It says exactly which file to create.
3. Write your file as `submissions/<your-github-username>.py` (lowercase).
   Every task file has four parts: `PREDICTIONS` (your expected outputs for
   three given inputs, written before you code), `solve`, `MY_CASES` (two
   inputs of your own where a naive solution fails), and `NOTES` (3–5
   sentences answering the task's why-question). Then check it:

   ```sh
   uv run python scripts/tasks.py check tasks/l02-ngram/<slug> <username>
   ```

4. When the check passes, open a PR from your fork with the title
   `l02-ngram/<slug>: <username>` and `Related to #<issue>` in the body.
   Only your one file should be in the PR.
5. Deadline: Tuesday 23:59 of the same week. Late PRs are merged if they pass
   but are not counted for that week.

Try it first on the example: copy `tasks/example/word-count/submissions/octocat.py`
to `<username>.py` in the same folder and run the check.

## Teaching team: create a task

```sh
uv run python scripts/tasks.py new l02-ngram split-sentences "Split text into sentences"
```

This copies `TEMPLATE/` to `tasks/l02-ngram/split-sentences/`. Then:

1. Fill in `instruction.md`. Keep the six headings; a task should take a
   student 30–60 minutes.
2. Edit `tests/test_task.py`: set `FUNCTION`, `CASES`, and the three
   `PREDICTION_INPUTS` (repeat them in `instruction.md`). Put one why-question
   in the instruction's `NOTES` block. Add a hidden-input test only if the
   cases alone would be too easy to copy.
3. Set the deadline and time estimate in `task.toml`.
4. Run `uv run python scripts/tasks.py check tasks/l02-ngram/split-sentences`
   with a reference submission of your own, then delete it (solutions stay in
   the private instructors' repository).
5. Open the GitHub issue with the same title, labels `l02-ngram` + `task`,
   and a link to the folder.

`uv run python scripts/tasks.py list` shows every task with its deadline and
submission count; `progress` prints merged submissions per student.
