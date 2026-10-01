# October 1, 2026 merge batch: digit-grouping

**Status:** prepared; student PR merges and teaching-team corrections pending.

## Deadline and instructor decisions

- Extended deadline: **September 30, 2026, 23:59 Asia/Shanghai (UTC+08:00)**,
  confirmed by the instructor on October 1. Earliest merge: October 1, 00:00.
- Checked `task.toml`, `instruction.md`, and [release issue #201](https://github.com/baojian/llm-26-fall/issues/201).
- On October 1, the instructor explicitly accepted late submissions for the
  three optional Lecture 01 exercises and authorized minor teaching-team fixes.
- This is the first and only recorded batch for this task. The main-branch
  submission history contained no student solutions when review began.
- Instructor demonstrations and the closed survey are outside this batch.

## Reviewed submissions

| PR | Student | Reviewed head commit | Deadline status | Teaching-team correction | Merge status |
| --- | --- | --- | --- | --- | --- |
| [#225](https://github.com/baojian/llm-26-fall/pull/225) | Sunny-hs | `8068eac0ace1f28b6623041c1828797063c140c0` | On time | None | Pending |
| [#229](https://github.com/baojian/llm-26-fall/pull/229) | naonaozhong | `00b4b448bdff6862819e6009d5c694cc869624cf` | On time | Trailing whitespace/blank lines only | Pending |
| [#231](https://github.com/baojian/llm-26-fall/pull/231) | porgin | `8c4fa0497fe2fe1ae0d9360ca5fa639a6ef79104` | On time | English NOTES translation/correction | Pending |
| [#240](https://github.com/baojian/llm-26-fall/pull/240) | yzm896 | `d6d7533add1e9ca24b2eac46f1f150f59658f85e` | On time | Trailing whitespace/blank lines only | Pending |
| [#244](https://github.com/baojian/llm-26-fall/pull/244) | Jessica0818 | `570efd437ef5d68332a28d98e55856b3d42b94d8` | On time | English NOTES translation/correction | Pending |
| [#248](https://github.com/baojian/llm-26-fall/pull/248) | duduaena | `cc0cdc608e3598100b93c07b23bd1d52a36edb58` | On time | Trailing whitespace/blank lines only | Pending |
| [#249](https://github.com/baojian/llm-26-fall/pull/249) | 6o6-Ma | `881a5a71e6c7ebf344049bfb4fb6df0ed78fbf8c` | On time | English NOTES translation/correction | Pending |
| [#252](https://github.com/baojian/llm-26-fall/pull/252) | geekeraman | `edfd81ad723576123919a250b70ebedeb8a62c7a` | On time | None | Pending |
| [#255](https://github.com/baojian/llm-26-fall/pull/255) | hxjuju | `5067a831c92e83f562fefb3ffca0dfb6f2da90bb` | On time | None | Pending |
| [#258](https://github.com/baojian/llm-26-fall/pull/258) | Snowstalgia | `a1f4a9a5cdc0ba1cbd2bfcd4a5d97b280ff1ad6b` | Late; explicitly accepted | PR title corrected; source unchanged | Pending |

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
