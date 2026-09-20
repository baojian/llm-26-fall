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
  l01-tokenization/<slug>/       Week 1 tasks: gpt2-pretokenizer, digit-grouping, han-runs
  l02-ngram/<slug>/              real tasks, grouped by lecture label
    task.toml                    id, title, deadline, what to submit
    instruction.md               the task: goal, interface, examples, how it is checked
    tests/test_task.py           the checker, run on every file in submissions/
    submissions/<username>.py    one file per student
```

[PROGRESS.md](PROGRESS.md) is the class progress board: tasks passed,
correct rate, participation, merged PRs, and badges per student. It is
regenerated after each merge batch with `uv run python scripts/progress.py`.

Start with the [Lecture 01 tokenization activities](l01-tokenization/README.md).

Task issues on GitHub carry the same lecture label as the folder
(`l02-ngram`, `l03-embeddings`, …) plus the type label `task`. The full
rules, timeline, and labels are in
[docs/participation-workflow.md](../docs/participation-workflow.md).

## Students: do a task

1. Pick an open issue labelled `task` and comment "I'll take this". Task issues
   accept submissions from many students; comments and assignments are not
   exclusive claims.
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
   Only your one file should be in the PR. Open **Checks → Public task
   feedback** for milestones and failed cases. Push fixes to the same branch
   to rerun it; see the [feedback guide](../docs/task-feedback.md).
5. Open your PR by the deadline in the task's `instruction.md` (normally
   Tuesday 23:59 of the same week). Solution PRs remain open until the
   teaching team merges one batch for the task **after its deadline**;
   they do not need to be merged by the submission cutoff. Late submissions
   and changes after the batch require an explicit instructor decision.

Public PRs and forks are visible even before merging. The delayed merge keeps
solutions off `main` during the exercise; it does not make submissions private.

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
3. Set the deadline and time estimate in `task.toml`. If the deadline depends
   on the release PR's merge date, record that rule in both `task.toml` and
   `instruction.md`; after merging, publish the matching calendar date in
   both files and the task issue.
4. Run `uv run python scripts/tasks.py check tasks/l02-ngram/split-sentences`
   with a reference submission of your own, then delete it (solutions stay in
   the private instructors' repository).
5. Open the GitHub issue with the same title, labels `l02-ngram` + `task`,
   and a link to the folder.

`uv run python scripts/tasks.py list` shows every task with its deadline and
submission count; `progress` prints merged submissions per student.

## Teaching team: merge solutions once, after the deadline

This is a mandatory rule for every student exercise solution and correction,
including a solution embedded in a PR with another title or purpose:

1. Review and run checks at any time, but never merge or enable auto-merge
   before the task's deadline. Verify the current time and that `task.toml`,
   `instruction.md`, and the release issue agree on the cutoff and time zone.
   If the deadline is unclear, do not merge.
2. After the deadline, finish reviewing all on-time PRs for that task. Resolve
   outstanding reviews or obtain the instructor's decision before starting
   the batch. Inspect the task's merge history and any `merge-batch.md` to
   avoid a second batch.
3. Publish `merge-batch.md` in the task folder through a PR, listing the
   deadline, accepted PR numbers, reviewed head commits, and batch status.
   Merge those student PRs in one coordinated batch, preserving attribution.
   An interrupted batch may resume using the same record.
4. Record completion and refresh the progress board through a PR. Do not
   start another batch or merge later solution fixes without an explicit
   instructor decision naming the task and exception.

A general request to merge student PRs never waives this policy. Surveys and
course-material fixes without exercise solutions follow their separate rules.
