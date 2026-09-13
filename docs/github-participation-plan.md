# Teaching with GitHub: semester participation plan

Review draft dated September 10, 2026. This plan describes how the course uses
GitHub so that students participate every week and leave the course with
skills they will use in research and industry: working from a fork, writing a
reviewable pull request, reading a failed check, reviewing a peer, resolving a
conflict, and publishing a reproducible project. It builds on the Week 1
survey ([issue #6](https://github.com/baojian/llm-26-fall/issues/6)), which
drew 24 student pull requests in one day.

Dates and topics follow [schedule.md](schedule.md) and
[course-revision.md](course-revision.md). Assessment weights are unchanged:
participation 5%, quizzes 15%, practicals 40%, project 40%.

## What worked in Week 1, and what to keep

1. **One file per student.** Every response is its own file, so PRs never
   conflict and merge in any order.
2. **A mechanical review rule.** A reviewer checks filename, at most two
   boxes, and one app name. No judgment, no delay.
3. **Nothing to leak.** Each answer is personal, so an early merge reveals
   nothing to later students.
4. **A closed loop.** The class sees its own results in the next lecture.

Every weekly activity below keeps these four properties.

## Three kinds of participation

| Tier | Who | What | Where it counts |
| --- | --- | --- | --- |
| A. Weekly report | Every student, every teaching week | One small file with the week's evidence (see the table). Opened as a PR from a fork, checked by CI, merged by the teaching team. | Participation 5%. Ten merged reports out of fourteen earn the full mark. |
| B. Claimable task | Volunteers, one student per task | A real improvement to the course repo: a notebook bug, a missing test, a Chinese translation, a figure, a paper summary. Labelled `help wanted`; the first student to comment "I'll take this" is assigned. | Named in `CONTRIBUTORS.md`. Two merged tasks replace one missed weekly report. No other grade effect. |
| C. Project in the open | Every student, from Week 7 | The individual project lives in the student's own GitHub repository created from a course template. Proposal, progress update, and final release are PRs and tags in that repository. | Project 40%, graded by the existing rubric. GitHub is the submission channel, not extra credit. |

Practical assignments A1–A3 are **not** submitted through public PRs. Their
solutions are private, so they use private submission (GitHub Classroom or the
existing channel). Public PRs are for evidence that is different for every
student.

## The weekly rhythm

| When | Action |
| --- | --- |
| Wednesday, end of lecture | Teaching team opens the pinned issue `Week NN report: <title>` with the template, the filename rule, the due time, and the exact CI checks. Opens 2–3 `help wanted` tasks for the week. |
| Wednesday to Tuesday | Students work in `workspace/`, then add one file under `reports/week-NN/<username>.md` and open a PR titled `week-NN: <username>` with `Related to #<issue>` in the body. |
| Any time | CI validates the file (path, required headings, numbers present, no other files changed). A red check is the student's signal to fix and push to the same branch. |
| Within 48 hours | A teaching-team member merges green PRs. Red PRs get one templated comment pointing to the failing check. |
| Tuesday 23:59 | Reports close for this week. Late PRs are merged but not counted. |
| Next Wednesday | The first five minutes of the lecture show the aggregated results (a script builds the chart from the merged files, as with the survey). |

The same rhythm every week is what makes participation a habit. Do not vary
the filenames, titles, or due day.

## Week-by-week plan

"Report" is the Tier A file for that week and matches the evidence column in
`course-revision.md`. "Git skill" is the one new GitHub skill taught in the
guided-practice period; each skill is used again in later weeks.

| Week | Date | Lecture | Weekly report (one file per student) | Git skill taught this week | Claimable tasks (examples) | Project milestone |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Sep 9 | Introduction and tokenization | Survey response (done) | Fork and create a file in the GitHub web editor; open a PR | — | Project format introduced |
| 2 | Sep 16 | Language models and perplexity | Bigram perplexity on a text you chose: vocabulary size, train and held-out perplexity, one sampled sentence | Clone your fork, branch, commit, push, open the PR from the command line; sync with `upstream` | Typos in Lecture 02; a Chinese-text example for the n-gram notebook | — |
| 3 | Sep 23 | Embeddings and PyTorch | Nearest neighbours of one word you chose, and a shape trace of the small LM | Read a failing CI check and fix it by pushing to the same branch | Add a test for the Lecture 02 notebook; translate one slide section | Quiz week |
| 4 | Sep 30 | Neural LMs and attention | Loss at three steps of a tiny run with your own seed, and one attention row checked by hand | Review a classmate's PR: request changes or approve with a comment | A "common errors" page for the training loop | A1 due (private) |
| 5 | Oct 7 | skip | none | none | Holiday pool stays open | — |
| 6 | Oct 14 | The Transformer | Shape table for one decoder block and a causal-mask check | Resolve a merge conflict (a prepared conflict exercise on a practice branch) | Diagram or Plotly figure for the decoder block | Quiz week |
| 7 | Oct 21 | Pretraining and decoding | A training curve, a resumed run, one diagnosed failure | Create your project repo from the template; README, `.gitignore` for weights, issues and a milestone | Paper summary in `papers/` | Proposal due as a PR in the student's repo; link posted in the Week 7 course issue |
| 8 | Oct 28 | Data preparation | Two data policies compared under the same budget | Add a GitHub Actions smoke test to your project repo | Deduplication example for the data notebook | Quiz week |
| 9 | Nov 4 | Compute budgets and scaling | One resource estimate and the measured value | Tag a release; pin the environment with `uv.lock` | Scaling-fit figure | A2 due (private) |
| 10 | Nov 11 | Evaluation | One benchmark item audited for contamination or prompt sensitivity | Write a reproducibility checklist into your README | Evaluation error-analysis template | Peer-reproduction pairs announced |
| 11 | Nov 18 | Supervised fine-tuning | Before and after adaptation on one prompt set | Open an issue in a classmate's project repo | SFT masking test | Quiz week |
| 12 | Nov 25 | Preferences and alignment | A checked preference loss and one reward failure | Respond to an issue on your own repo and close it with a commit | DPO worked example figure | — |
| 13 | Dec 2 | Retrieval and RAG | Retrieval failures and generation failures measured separately | Peer reproduction: clone a classmate's repo, run the baseline, file the result as an issue | RAG evaluation examples | Progress update due as a PR; peer-reproduction issue filed |
| 14 | Dec 9 | Efficient inference | Latency and throughput table with a correctness check | Use a release to publish results with the exact commit | Benchmark script | A3 due (private) |
| 15 | Dec 16 | Diffusion language models | Sampling comparison with stated assumptions and budget | Write a changelog from `git log` | Diffusion sampling figure | Quiz week |
| 16 | Dec 23 | Agents and synthesis | Claim audit of your own project: each claim linked to a commit or file | Final release tag `v1.0`; archive the repository state | — | Final claim audit |
| 17 | Dec 30 | No lecture | none | none | — | Final report, code, and experiment record at the tagged release |

Fourteen report weeks exist (Weeks 2–4, 6–16). Ten merged reports earn the full
participation mark, which tolerates three absences plus the holiday.

## What students learn that is useful outside the course

By Week 16 every student has done each of these at least once, and most of
them many times:

1. Forked and synced a repository, worked on a branch, and opened a PR from
   the command line (Weeks 1–2, then weekly).
2. Read a failing CI check and fixed the branch (Week 3, then whenever red).
3. Reviewed a peer's PR and received reviews on their own (Week 4 onward).
4. Resolved a merge conflict on purpose, in a safe setting (Week 6).
5. Set up a repository from scratch with README, ignore rules, issues,
   milestones, CI, and releases (Weeks 7–9).
6. Published a result someone else could reproduce, and reproduced someone
   else's (Weeks 10–13).
7. Linked every claim in a report to a commit or file (Week 16).

These are the habits a research lab or an engineering team expects on day one.

## One-time setup (Weeks 1–2)

1. `CONTRIBUTING.md` at the repo root: the weekly rhythm, filename and title
   rules, the "no `Closes #N`" rule, privacy rule (GitHub username only, no
   real names or student IDs), and how to claim a task. The survey guide in
   `surveys/lecture-01/README.md` is most of this text already.
2. `.github/ISSUE_TEMPLATE/`: `weekly-report.md` (for the teaching team),
   `claimable-task.md`, and `bug.md`.
3. `.github/pull_request_template.md`: three checkboxes (only my file changed,
   title follows the rule, `Related to #N` present).
4. `.github/workflows/ci.yml`: runs `uv run pytest tests/` and a new
   `scripts/check_report.py` that validates any file under `reports/` changed
   in the PR. Runs on `pull_request` only; no secrets needed.
5. Labels: `week-02` … `week-16`, `weekly`, `help wanted`, `good first issue`,
   `project`, `needs-fix`. Milestones: one per week for the claimable tasks.
6. Branch protection on `main`: require the CI check and one approving review
   from the teaching team. Students cannot merge; that keeps the review step
   meaningful.
7. `scripts/participation.py`: counts merged `week-NN` PRs per username from
   the `reports/` tree and prints the table. Students can run it themselves,
   so the participation mark is transparent before grades are released.
8. A project template repository (`baojian/llm-26-fall-project-template`) with
   README skeleton, `.gitignore`, a smoke-test workflow, and the proposal,
   progress, and final report templates.
9. Turn on GitHub Discussions for questions, so issues stay for work items.
10. A private mapping from GitHub username to student ID, kept in the
    instructors repository only.

## Review load and who does what

- CI does format checking. Humans only confirm the check is green and the
  file is the student's own. Target: one minute per PR.
- With 40–60 students, that is about one hour of merging per week. Split it
  across the teaching team by PR number parity, as the survey guide suggests.
- Claimable tasks need real review. Cap them at three per week so the
  instructor can review each properly, and give a decision within one week.
- Peer review (Week 4 onward) does not replace the teaching team's merge; it
  is a learning activity and a light sanity check.

## Risks and how the plan handles them

| Risk | Handling |
| --- | --- |
| Participation drops after Week 3 | Same rhythm every week; results shown in class; the participation table is public and computable by students. |
| Answer leakage through merged PRs | Weekly reports are individual by construction (own text, seed, word, prompt). Graded assignments never pass through public PRs. |
| Review backlog discourages students | CI makes merges mechanical; 48-hour merge target; templated comment for red checks. |
| Students edit course files by accident | PR template checkbox plus CI rule "only files under `reports/week-NN/` changed" for `week-NN` PRs. |
| Privacy | Username only. No names, IDs, or personal data in reports. Stated in `CONTRIBUTING.md` and every weekly issue. |
| Fork drift and confusing conflicts | Week 2 teaches `upstream` sync; Week 6 teaches conflict resolution before the project phase needs it. |
| Free-riding on claimable tasks | One student per task, assigned on claim, unassigned after seven days without a PR. |
| Weeks with heavy deadlines (4, 9, 14) | Those weeks' reports are the smallest in the table and reuse output the assignment already produces. |

## Decisions for the instructor

1. Confirm "ten of fourteen merged reports = full participation" and the
   "two claimable tasks replace one report" rule.
2. Confirm that practicals stay private and only weekly reports and projects
   use public PRs.
3. Choose GitHub Classroom or plain template repositories for the project.
   Classroom gives private per-student repos and a roster; plain templates are
   simpler and public by default.
4. Decide whether peer reproduction (Week 13) is required or optional.
5. Set the weekly due time. This draft uses Tuesday 23:59 so results can be
   shown on Wednesday.

## First actions

1. Publish `CONTRIBUTING.md`, the PR template, and the CI workflow before
   the Week 2 lecture on September 16.
2. Open the Week 2 report issue at the end of that lecture and demonstrate the
   command-line PR live, as the survey was demonstrated in Week 1.
3. Show the Week 1 survey chart in the first five minutes of Week 2.
4. Merge the two open survey PRs (#34, #36) and close #6 once responses stop.
