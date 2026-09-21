# Task feedback: predict, build, test, explain

Every exercise solution PR gets automatic feedback from the task's public
checks. Push a fix to the same branch to try again. The report shows progress
through the exercise without assigning a grade or merging your PR.

## Read your report

Open your PR's **Checks** tab, select **Public task feedback**, and open the
workflow run's **Summary**. Expand a failed check to see its input, expected
output, actual output, or Python error. The run also saves a downloadable
`task-feedback-pr-N` artifact containing `feedback.md`, including failed
attempts. Reports identify the submission commit and public checker commit;
use the run for your latest push. Artifacts are retained for 14 days.

A first-time contributor's workflow
may need a teaching-team member to approve the run before feedback appears.

| Milestone | What it checks |
| :--- | :--- |
| Submission format | Your filename, PR title, issue reference, and one-file scope |
| Examples and edge cases | Exact results on the supplied public cases, including whitespace and Unicode |
| Your predictions | The three requested predictions agree with your implementation |
| Your own tests | Your new test inputs satisfy the task's requirements and agree with your implementation |
| Explanation completeness | `NOTES` meets the checker's text requirements; a human reviews its reasoning |

For example, a digit-grouping failure could show:

```text
Input: '2026'
Expected: ['202', '6']
Actual: ['2026']
```

Try that input locally and explain the difference before changing your code.
Then rerun all checks. Keep `MY_CASES` distinct from the supplied cases: use
them to explore a different boundary or challenge a tempting shortcut.

**All public checks passed** means you are ready for human review. It does
not prove correctness on every input, validate your explanation, or award
points. The teaching team reviews reasoning and additional cases. Solutions
still merge in [one batch per task after its deadline](../tasks/README.md#teaching-team-merge-solutions-once-after-the-deadline).
Public PRs and forks remain visible before merge.

## Get the same feedback locally

The original checker command still works:

```sh
uv run python scripts/tasks.py check tasks/l01-tokenization/digit-grouping <username>
```

For the milestone report and detailed failures:

```sh
uv run python scripts/task_feedback.py tasks/l01-tokenization/digit-grouping <username>
```

Replace the task folder and lowercase username for your submission. To save
the report, add `--summary workspace/feedback.md`. Keep generated reports in
`workspace/`; your PR should contain only your submission file.

The report runs only your file in a temporary copy of the public task. Syntax
errors, import errors, missing fields, and loops that exceed the 30-second
check limit receive feedback. Submission files may be at most 128 KiB. Empty
or skipped test runs do not count as a pass. This command executes your Python
code; it is not a security sandbox.

## Teaching team: maintain useful checks

Use the same public checker locally and in CI. Add cases that cover the
specification's normal behavior and meaningful boundaries, with input,
expected output, and actual output in failure messages. Validate fixture
files and reject an empty or obviously incorrect implementation. Verify a
reference implementation privately before release; keep reference solutions
and hidden tests in the private instructor repository.

The task template supplies the feedback milestones. Preserve its
`user.<username>` test IDs so the runner can verify that submission checks
actually ran. For a new task, adapt its cases and explanation prompt; do not
add requirements to tests that are absent from the instructions. Human
review still checks whether the task is clear, reasonable, and educational.

CI uses the checker and data from the PR's base branch and copies only the
student's permitted file from the exact PR commit. Changes to tests, data,
other submissions, or course files are rejected for solution PRs. Student
code runs on an ephemeral GitHub-hosted runner with no supplied secrets or
write token; checkout credentials are not retained. The workflow never
merges PRs or publishes solutions into `main`.

Run `uv run python -m pytest tests/test_task_feedback.py` to verify the
feedback service. Every PR also runs **Feedback service regression tests** in
a separate job against the proposed changes, covering the service, published
task fixtures, and task commands. This detects broken checker changes while
the student's feedback continues to use the trusted base branch.

**Actions → Task feedback → Run workflow** runs those service tests on GitHub
without submitting an exercise solution. A failed setup or fetch is reported
as unavailable feedback, not as a result about the student's code. This
workflow is separate from [lecture checks and previews](lecture-quality.md).
