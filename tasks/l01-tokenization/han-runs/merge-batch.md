# October 1, 2026 merge batch: han-runs

**Status:** complete. All listed student PRs merged on October 1, 2026.
The completion commit also contains the reviewed teaching-team corrections
and the regenerated progress board.

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
| [#227](https://github.com/baojian/llm-26-fall/pull/227) | Sunny-hs | `bfc4c63bc0dcd075edb05d4db7d118a809b69052` | On time | None | Merged [fdd92b8](https://github.com/baojian/llm-26-fall/commit/fdd92b8290d9823fbd02adbc0f29db3ce453993f) |
| [#233](https://github.com/baojian/llm-26-fall/pull/233) | porgin | `38422da7e5efec58d9c79f7752649b3dfe3ec910` | On time | English NOTES translation/correction | Merged [3e87e83](https://github.com/baojian/llm-26-fall/commit/3e87e83be94657759c07855f0deba668bc54a721) |
| [#245](https://github.com/baojian/llm-26-fall/pull/245) | Jessica0818 | `99fa47b6b48b92e2819028912aea0775725bb424` | On time | Trailing whitespace/blank lines only | Merged [35bb615](https://github.com/baojian/llm-26-fall/commit/35bb615f42ca5706a5525a67a511185ed881d882) |
| [#251](https://github.com/baojian/llm-26-fall/pull/251) | 6o6-Ma | `f71c6664744e1ebb73a95ad9d945e8a3ccfb70c1` | On time | English NOTES translation/correction | Merged [16e478d](https://github.com/baojian/llm-26-fall/commit/16e478d11f6a8175872501cc6518bf05918b5a4f) |
| [#257](https://github.com/baojian/llm-26-fall/pull/257) | hxjuju | `a80774f0a4fba8c513495bf947862f84a463806f` | On time | Trailing whitespace/blank lines only | Merged [765c300](https://github.com/baojian/llm-26-fall/commit/765c300bf8cb30473d7ada3a3dcc440fd53cdab3) |

## Review and validation

- Inspected each complete diff, implementation, predictions, personal cases,
  explanation, and previous review comments; each PR changes only its own
  correctly named submission file.
- All implementations pass the current public task checker and **1,010**
  additional cases covering Unicode, whitespace, boundaries, empty input,
  long input, and lossless reconstruction.
- The final explanation and whitespace corrections also pass the public
  checker. An AST comparison confirms that they do not change executable
  logic, predictions, or personal cases.
- The instructor authorized resolving the outstanding explanation comments
  through the teaching-team edits listed above. Corrected explanations
  remain clearly attributed to the teaching team.

## Completion record

- The deadline and exact-head batch records were merged first in
  [PR #262](https://github.com/baojian/llm-26-fall/pull/262).
- Each student PR was merged individually at the reviewed head above, with
  passing current public-feedback and regression checks. Original authorship
  and PR identities are preserved.
- Nine explanations across the three tasks were translated or corrected by
  the teaching team; seven other files received whitespace-only cleanup.
  These edits are separate from student commits, and changed explanations
  carry a teaching-team attribution comment.
- All final submissions pass their public task checks. The progress board was
  regenerated on October 1 from these files and the actual merged-PR counts.
- The archived survey, its responses, and its participation counts are retained.

The progress board records the final reviewed submissions, including approved
teaching-team corrections; it is participation feedback, not a graded score.
