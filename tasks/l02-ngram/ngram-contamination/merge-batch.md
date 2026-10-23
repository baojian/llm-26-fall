# October 20, 2026 review: n-gram contamination

**Status:** complete. All three student PRs merged on October 20, 2026,
at the approved heads after the batch record reached `main` and checks passed.

**Instructor decision, October 20, 2026:** the instructor approved merging
#295, #301, and #303 with the prepared teaching-team translation and
clarifications, and crediting the accepted work on the participation board.
This authorization covers the listed PRs and does not reopen submissions.

## Deadline and merge history

- Published deadline: **October 14, 2026, 23:59 Asia/Shanghai (UTC+08:00)**,
  machine timestamp `2026-10-14T23:59:00+08:00`.
- Checked `task.toml`, `instruction.md`, and
  [release issue #274](https://github.com/baojian/llm-26-fall/issues/274);
  all three agree. This review takes place on October 20, after the cutoff.
- No prior student solution merge or `merge-batch.md` exists for this task.
  This is its first and only authorized batch.
- All three PRs were opened before the deadline. #303 has a later merge of
  upstream `main`, but its submission blob is identical to the pre-deadline
  parent `346211b55e697e1e7d096172ccf4278d47a13266`.
- No other open student solution PR for this task was found in the review.

## Reviewed submissions

| PR | Student | Reviewed original head | Original checks | Approved teaching-team correction |
| --- | --- | --- | --- | --- |
| [#295](https://github.com/baojian/llm-26-fall/pull/295) | `harmonyhhh66` | `e9cff4dba83bf52a8dbfe8befaa4415978d4d0cb` | 19 passed | Translate NOTES and comments into English; correct one rounded value; remove trailing whitespace |
| [#301](https://github.com/baojian/llm-26-fall/pull/301) | `OOOO1027` | `69520a16bbf9a9312ad985c80fe68b508ebefb05` | 19 passed | None |
| [#303](https://github.com/baojian/llm-26-fall/pull/303) | `cyk-1234` | `5fa75c95813774eee67eb2af23b0bc2a1c449b5e` | 19 passed locally; GitHub checks pass after title correction | Add attributed explanation clarifications |

The missing space in #303's PR title was corrected during review. That
metadata-only change reran GitHub feedback successfully without changing the
submission. The original PRs preserve student attribution.

## Final heads for this batch

| PR | Reviewed merge head | Correction commit |
| --- | --- | --- |
| #295 | `b4faaf4382d5ed683972b50239df40ab5803d0b6` | `b4faaf4382d5ed683972b50239df40ab5803d0b6` |
| #301 | `69520a16bbf9a9312ad985c80fe68b508ebefb05` | None |
| #303 | `6ad73d3dffd2759e777735ae5dd5b376c48f273d` | `6ad73d3dffd2759e777735ae5dd5b376c48f273d` |

## Review and validation

- Inspected all full diffs, histories, predictions, personal cases,
  reflections, and review threads. Each PR changes only its author's file.
- All three originals pass the current 19 public checks, as do the two
  corrected versions.
- Reproduced the supplied comparison with each original submission:
  four/eight bands yield 38/62 candidates, recall `0.95/1.00`, and precision
  `1.00/0.645161`. The 1,504 containment pairs remain unchanged.
- The constructed audit gives full/clean perplexities
  `5.656854/22.627417` for A and `8.000000/11.313708` for B.
- #295's translation retains the student's four explanatory points.
  An AST comparison excluding NOTES confirms no executable, prediction,
  or personal-case changes.
- #303's attributed clarification distinguishes duplicate verification from
  exhaustive containment, identifies the audit as invented scores rather
  than evidence of actual memorization, and corrects the `contained_passage`
  explanation: it uses bigrams, Jaccard `0.5` fails its `0.8` threshold,
  and containment `1.0` passes. Its complete AST remains unchanged.
- Experiment reports remain local. These are optional participation tasks;
  checker results are not course grades.

## Completion record

- The approved batch record merged first in
  [PR #308](https://github.com/baojian/llm-26-fall/pull/308), commit
  `b9f2d08b7beae8cbbc2d750f7400acca054616a2`, at 13:00:43 Asia/Shanghai.
- All three student PRs merged individually with their original authorship
  and separate teaching-team correction history preserved:

| PR | Merged October 20, Asia/Shanghai | Merge commit |
| --- | --- | --- |
| #295 | 13:01:01 | `6c469a52ac5e4b9dd6c0a91defbaf7400ecc18a6` |
| #301 | 13:01:06 | `cff817c7ce3f67072e9a4472c567533a74d4d8e8` |
| #303 | 13:01:09 | `4ec74bf8e3c8398082af33b9122f1f139adb3016` |

- Each final submission passes all 19 public checks. All three PRs had
  passing GitHub feedback and regression checks before merging.
- The completion PR regenerates `tasks/PROGRESS.md` and `tasks/progress.svg`
  from the merged submissions and actual merged-PR counts. `harmonyhhh66`,
  `oooo1027`, and `cyk-1234` each receive one passed-task entry for this task.
  Each receives Lecture 02 participation once, regardless of completing one
  or both tasks; their earlier survey participation is retained.
- These entries record participation and checker results. No course-grade
  allocation is changed.

Later submissions or fixes require a new explicit instructor decision under
the repository policy.
