# October 1, 2026 merge batch: gpt2-pretokenizer

**Status:** complete. All listed student PRs merged on October 1, 2026.
The completion commit also contains the reviewed teaching-team corrections
and the regenerated progress board.

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
| [#226](https://github.com/baojian/llm-26-fall/pull/226) | Sunny-hs | `99aadfde6284de3d60cc34f8c9f2d46d2694f1b4` | On time | None | Merged [4c1c140](https://github.com/baojian/llm-26-fall/commit/4c1c140c99a861435b1e6279a6a14120ea3144fb) |
| [#228](https://github.com/baojian/llm-26-fall/pull/228) | Cincinnatus23 | `e7e15d5dff18fea8d16fe5f6f8c2b195767e80f0` | On time | Trailing whitespace/blank lines only | Merged [e2975cb](https://github.com/baojian/llm-26-fall/commit/e2975cbc35021ad5e57ca608824e6d6d4321b281) |
| [#230](https://github.com/baojian/llm-26-fall/pull/230) | naonaozhong | `534448f169d14a707c4e7db396ca6181a8aa7edb` | On time | None | Merged [4c9e71d](https://github.com/baojian/llm-26-fall/commit/4c9e71d447e7b7372cad1349cecd84e40f2afdef) |
| [#232](https://github.com/baojian/llm-26-fall/pull/232) | porgin | `933b1e5dffc201c1185f2b35739f2c66def81894` | On time | English NOTES translation/correction | Merged [b9af6b5](https://github.com/baojian/llm-26-fall/commit/b9af6b50e7b513461fd7ae4b8ae92a7cf090f499) |
| [#246](https://github.com/baojian/llm-26-fall/pull/246) | Jessica0818 | `00ad932ef60fc15e37c5c25ab225ad4e192bd23f` | On time | Trailing whitespace/blank lines only | Merged [7d2dd47](https://github.com/baojian/llm-26-fall/commit/7d2dd479ca56719a5f5d173c5d4c799640a3e4c6) |
| [#247](https://github.com/baojian/llm-26-fall/pull/247) | Charlie-Yong | `c734823bfda7438bf25087fc05ae198926ccc252` | On time | English NOTES translation/correction | Merged [164d89b](https://github.com/baojian/llm-26-fall/commit/164d89b9cc70ede1a8cae526470e0ea6074f7cd8) |
| [#250](https://github.com/baojian/llm-26-fall/pull/250) | 6o6-Ma | `2d1b05993f4e581d68056de353cc2640138c960c` | On time | English NOTES translation/correction | Merged [31aa3a9](https://github.com/baojian/llm-26-fall/commit/31aa3a94a1693cab5ce1fd341dfd952cd52e502d) |
| [#254](https://github.com/baojian/llm-26-fall/pull/254) | geekeraman | `7cf842ca85715517c0367d615590523b56f9a5c7` | On time | English NOTES translation/correction | Merged [a1a2941](https://github.com/baojian/llm-26-fall/commit/a1a294171816cffed3787c79eb568e7d0f06af09) |
| [#256](https://github.com/baojian/llm-26-fall/pull/256) | hxjuju | `8cab6825bdb9efbfbddf230928c58627b7636a2e` | On time | None | Merged [ddadf07](https://github.com/baojian/llm-26-fall/commit/ddadf078a7660b0de9e2423e577a1abbfc864189) |

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

## October 10, 2026 instructor-demo exception

**Status:** complete. [PR #221](https://github.com/baojian/llm-26-fall/pull/221)
merged on October 10, 2026 at 09:06:35 Asia/Shanghai (UTC+08:00), as
[`c391124`](https://github.com/baojian/llm-26-fall/commit/c3911244e91cd21dd86c59802b5fb7d1e7d4fcdb).

- On October 10, 2026 (Asia/Shanghai), the instructor explicitly requested
  fixing and merging PR #221 now that the Lecture 01 task has finished.
  This authorizes this specific corrected instructor demonstration after the
  completed student batch; the nine-student batch above remains complete.
- Rechecked `task.toml`, `instruction.md`, and release issue #202: all specify
  September 30, 2026, 23:59 Asia/Shanghai (UTC+08:00). The current date is
  strictly after that cutoff. The release issue remains closed.
- Reviewed PR head: `05e25826315368b8845c002f1b193deb1ff89851`.
  The solution PR changed only `submissions/baojian.py`.
- Restored the optional leading ASCII space in the word, digit, and punctuation
  rules; replaced the intentional-failure comments. The original demo commit,
  predictions, two personal cases, and explanation are preserved.
- Validation: 22 selected task checks passed; all 23 isolated public-feedback
  checks passed; nine focused whitespace, Unicode, and punctuation cases passed.
  Supplied tests and data are unchanged. Both GitHub checks passed on the
  reviewed head before merge.
- The authorization and reviewed-head record were merged first through
  [PR #285](https://github.com/baojian/llm-26-fall/pull/285). This completion
  update regenerates the progress board from the merged submissions and actual
  merged-PR counts, without changing the completed student batch or survey.
