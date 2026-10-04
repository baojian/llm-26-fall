# Lecture 02: N-gram activities

Both exercises are **optional** and have the same requirements for all students.
Each targets **90–120 minutes**, including a supplied CPU experiment. Complete
either or both; submit **one separate PR per exercise**.

**Submissions are open.** Follow the handout and task issue for each exercise.

**Deadline for both tasks:** October 14, 2026, 23:59 **Asia/Shanghai (UTC+08:00)**,
seven days after October 7. Each handout, `task.toml`, and task issue uses this date.

| Exercise | What you explore | Supplied corpus | Issue |
| :--- | :--- | :--- | :--- |
| [DSIR data selection](dsir-data-selection/instruction.md) | Hashed n-grams, importance weights, and sampling against baselines | 3,200 chunks of 128 tokens; 2,048 candidates | [#273](https://github.com/baojian/llm-26-fall/issues/273) |
| [Duplicates and contamination](ngram-contamination/instruction.md) | MinHash, LSH, exact overlap, and evaluation contamination | 320 documents, up to 1,024 tokens each | [#274](https://github.com/baojian/llm-26-fall/issues/274) |

Each task supplies hashing, validation, experiment code, and data. You implement
five helpers and complete `PREDICTIONS`, `MY_CASES`, and `NOTES` in one file.
No Assignment 01 implementation is needed.

To submit:

1. Copy the task's `starter.py` to `submissions/<your-lowercase-github-username>.py`.
2. Follow the five-section handout, run the comparison, and check your file locally.
3. Open a PR changing only that file. Use title `l02-ngram/<slug>: <username>`
   and body `Related to #<task-issue>`.
4. Read **Checks → Public task feedback** and push corrections to the same branch.

The action checks submission formatting, public cases, predictions, personal
tests, and explanation completeness. A human reviews your reasoning and
experiment evidence. See the [feedback guide](../../docs/task-feedback.md)
and [submission workflow](../README.md).

Task issues accept many students. Passing checks does not merge a PR: solutions
merge in one batch per task after its published deadline and human review.
Public PRs and forks are visible before merge. Graded assignments and project
deliverables continue to use private eLearning submissions.
