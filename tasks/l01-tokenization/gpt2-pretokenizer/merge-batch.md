# October 1, 2026 merge batch: gpt2-pretokenizer

**Status:** prepared; student PR merges and teaching-team corrections pending.

## Deadline and instructor decisions

- Extended deadline: **September 30, 2026, 23:59 Asia/Shanghai (UTC+08:00)**,
  confirmed by the instructor on October 1. Earliest merge: October 1, 00:00.
- Checked `task.toml`, `instruction.md`, and [release issue #202](https://github.com/baojian/llm-26-fall/issues/202).
- On October 1, the instructor explicitly accepted late submissions for the
  three optional Lecture 01 exercises and authorized minor teaching-team fixes.
- This is the first and only recorded batch for this task. The main-branch
  submission history contained no student solutions when review began.
- Instructor demonstrations and the closed survey are outside this batch.

## Reviewed submissions

| PR | Student | Reviewed head commit | Deadline status | Teaching-team correction | Merge status |
| --- | --- | --- | --- | --- | --- |
| [#226](https://github.com/baojian/llm-26-fall/pull/226) | Sunny-hs | `99aadfde6284de3d60cc34f8c9f2d46d2694f1b4` | On time | None | Pending |
| [#228](https://github.com/baojian/llm-26-fall/pull/228) | Cincinnatus23 | `e7e15d5dff18fea8d16fe5f6f8c2b195767e80f0` | On time | Trailing whitespace/blank lines only | Pending |
| [#230](https://github.com/baojian/llm-26-fall/pull/230) | naonaozhong | `534448f169d14a707c4e7db396ca6181a8aa7edb` | On time | None | Pending |
| [#232](https://github.com/baojian/llm-26-fall/pull/232) | porgin | `933b1e5dffc201c1185f2b35739f2c66def81894` | On time | English NOTES translation/correction | Pending |
| [#246](https://github.com/baojian/llm-26-fall/pull/246) | Jessica0818 | `00ad932ef60fc15e37c5c25ab225ad4e192bd23f` | On time | Trailing whitespace/blank lines only | Pending |
| [#247](https://github.com/baojian/llm-26-fall/pull/247) | Charlie-Yong | `c734823bfda7438bf25087fc05ae198926ccc252` | On time | English NOTES translation/correction | Pending |
| [#250](https://github.com/baojian/llm-26-fall/pull/250) | 6o6-Ma | `2d1b05993f4e581d68056de353cc2640138c960c` | On time | English NOTES translation/correction | Pending |
| [#254](https://github.com/baojian/llm-26-fall/pull/254) | geekeraman | `7cf842ca85715517c0367d615590523b56f9a5c7` | On time | English NOTES translation/correction | Pending |
| [#256](https://github.com/baojian/llm-26-fall/pull/256) | hxjuju | `8cab6825bdb9efbfbddf230928c58627b7636a2e` | On time | None | Pending |

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
