# October 20, 2026 review: DSIR data selection

**Status:** complete. Both remaining student PRs merged on October 20, 2026,
at the approved heads after the batch record reached `main` and checks passed.

**Instructor decision, October 20, 2026:** after reviewing the specific
exception and corrections, the instructor approved completing the remaining
DSIR submissions despite the earlier #277 merge, applying the prepared
teaching-team repairs, and crediting the accepted work on the participation
board. This authorization covers #300 and #302 and does not reopen submissions.

## Deadline and merge history

- Published deadline: **October 14, 2026, 23:59 Asia/Shanghai (UTC+08:00)**,
  machine timestamp `2026-10-14T23:59:00+08:00`.
- Checked `task.toml`, `instruction.md`, and
  [release issue #273](https://github.com/baojian/llm-26-fall/issues/273);
  all three agree. This review takes place on October 20, after the cutoff.
- [PR #277](https://github.com/baojian/llm-26-fall/pull/277), by `Cincinnatus23`,
  was already merged on October 9 at 16:05:46 Asia/Shanghai, before the published
  deadline. Its reviewed head was
  `2a5b3d4a1eb77049cb2595a8f3284ece07ae89e8`; the merge commit is
  `ac7f86df2d05f6e796dbf099a710c3b46bc9a73a`.
- No earlier `merge-batch.md` exists for this task. The October 20 instructor
  decision authorizes completing the remaining submissions after that merge.
- Both remaining PRs were opened before the deadline. No other open student
  solution PR for this task was found in the October 20 review.

## Reviewed submissions

| PR | Student | Reviewed original head | Original checks | Approved teaching-team correction |
| --- | --- | --- | --- | --- |
| [#300](https://github.com/baojian/llm-26-fall/pull/300) | `OOOO1027` | `f14ae57cd0888df877ea6ff347b03fbf6cd890d8` | 19 passed | None |
| [#302](https://github.com/baojian/llm-26-fall/pull/302) | `cyk-1234` | `a86d03c825f66bdfd1c5815d829dae4760d37f8b` | Python syntax error | Restore damaged Python imports, the `PREDICTIONS` opening, and two `pool` keys; translate comments and add attributed explanation corrections |

The original PRs preserve student attribution. Approved edits are separate
teaching-team changes. The five helper implementations, prediction values,
and personal-case expected outputs in #302 are retained.

## Final heads for this batch

| PR | Reviewed merge head | Correction commit |
| --- | --- | --- |
| #300 | `f14ae57cd0888df877ea6ff347b03fbf6cd890d8` | None |
| #302 | `5daad98b801f9b0cf27fd16e4952360c713c743b` | `5daad98b801f9b0cf27fd16e4952360c713c743b` |

## Review and validation

- Inspected both full diffs, commit histories, predictions, personal cases,
  reflections, and review threads. Each PR changes only its author's file.
- #300 passes all 19 current public checks. The repaired #302 also
  passes all 19; the original #302 is not described as passing.
- Reproduced development and final comparisons with the supplied CPU runner.
  Development chooses `M=256`: mean DSIR JS is `0.228352` at `M=16` and
  `0.208543` at `M=256`. Final means are `0.059515` uniform, `0.351108`
  target-only, `0.236573` ratio top-k, and `0.231733` DSIR.
- #302's original reflection incorrectly orders development ratio-top-k
  `0.213147` and DSIR `0.208543`, and attributes `same_models` selection to
  the ID tie-break. The attributed teaching-team note states the correct
  comparison and the actual cause: different supplied uniforms.
- Experiment reports remain local. These are optional participation tasks;
  checker results are not course grades.

## Completion record

- The approved batch record merged first in
  [PR #308](https://github.com/baojian/llm-26-fall/pull/308), commit
  `b9f2d08b7beae8cbbc2d750f7400acca054616a2`, at 13:00:43 Asia/Shanghai.
- Both student PRs merged individually with their original authorship and
  separate teaching-team correction history preserved:

| PR | Merged October 20, Asia/Shanghai | Merge commit |
| --- | --- | --- |
| #300 | 13:00:53 | `5ebb0b513a9d5ec3f5f4cd4adc57288843a68e6f` |
| #302 | 13:00:57 | `ce6bb87fcda08c5336bed191220d9d410c915319` |

- Each final submission passes all 19 public checks. Both PRs had passing
  GitHub feedback and regression checks before merging.
- The completion PR regenerates `tasks/PROGRESS.md` and `tasks/progress.svg`
  from the merged submissions and actual merged-PR counts. `oooo1027` and
  `cyk-1234` each receive one passed-task entry for DSIR; their second task
  entry comes from the contamination batch. The existing `cincinnatus23`
  submission and credit are retained.
- These entries record participation and checker results. No course-grade
  allocation is changed.

Later submissions or fixes require a new explicit instructor decision under
the repository policy.
