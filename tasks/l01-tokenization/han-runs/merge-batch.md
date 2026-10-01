# October 1, 2026 merge batch: han-runs

**Status:** prepared; student PR merges and teaching-team corrections pending.

## Deadline and instructor decisions

- Extended deadline: **September 30, 2026, 23:59 Asia/Shanghai (UTC+08:00)**,
  confirmed by the instructor on October 1. Earliest merge: October 1, 00:00.
- Checked `task.toml`, `instruction.md`, and [release issue #203](https://github.com/baojian/llm-26-fall/issues/203).
- On October 1, the instructor explicitly accepted late submissions for the
  three optional Lecture 01 exercises and authorized minor teaching-team fixes.
- This is the first and only recorded batch for this task. The main-branch
  submission history contained no student solutions when review began.
- Instructor demonstrations and the closed survey are outside this batch.

## Reviewed submissions

| PR | Student | Reviewed head commit | Deadline status | Teaching-team correction | Merge status |
| --- | --- | --- | --- | --- | --- |
| [#227](https://github.com/baojian/llm-26-fall/pull/227) | Sunny-hs | `bfc4c63bc0dcd075edb05d4db7d118a809b69052` | On time | None | Pending |
| [#233](https://github.com/baojian/llm-26-fall/pull/233) | porgin | `38422da7e5efec58d9c79f7752649b3dfe3ec910` | On time | English NOTES translation/correction | Pending |
| [#245](https://github.com/baojian/llm-26-fall/pull/245) | Jessica0818 | `99fa47b6b48b92e2819028912aea0775725bb424` | On time | Trailing whitespace/blank lines only | Pending |
| [#251](https://github.com/baojian/llm-26-fall/pull/251) | 6o6-Ma | `f71c6664744e1ebb73a95ad9d945e8a3ccfb70c1` | On time | English NOTES translation/correction | Pending |
| [#257](https://github.com/baojian/llm-26-fall/pull/257) | hxjuju | `a80774f0a4fba8c513495bf947862f84a463806f` | On time | Trailing whitespace/blank lines only | Pending |

## Review and validation

- Inspected each complete diff, implementation, predictions, personal cases,
  explanation, and previous review comments; each PR changes only its own
  correctly named submission file.
- All implementations pass the current public task checker and **1,010**
  additional cases covering Unicode, whitespace, boundaries, empty input,
  long input, and lossless reconstruction.
- Prepared explanation and whitespace corrections also pass the public
  checker. An AST comparison confirms that they do not change executable
  logic, predictions, or personal cases.
- The instructor authorized resolving the outstanding explanation comments
  through the teaching-team edits listed above. Corrected explanations
  remain clearly attributed to the teaching team.

## Batch sequence

1. Merge this deadline and batch record through a course-maintenance PR.
2. Merge each listed student PR at its reviewed head, retaining its author
   and PR identity. Verify current CI and pause if any reviewed head changes.
3. Apply the reviewed teaching-team corrections through a separate PR within
   this same batch. Some forks disable maintainer edits; this also keeps
   instructor-authored explanations distinct from the original submissions.
4. Verify every final submission, record merge commits and completion here,
   and regenerate the progress board from merged files and GitHub PR counts.

The progress board records the final reviewed submissions, including approved
teaching-team corrections; it is participation feedback, not a graded score.
