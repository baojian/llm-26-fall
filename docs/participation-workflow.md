# Participation workflow: timelines, issues, and pull requests

This page is the reusable operating chart for the course's GitHub-based
participation: when the teaching team opens issues, when students submit,
who checks what, and how the results return to the class. The diagrams are
[Mermaid](https://mermaid.js.org/) and render on GitHub; edit the text to
change them. Dates come from [schedule.md](schedule.md) and the
[course website](../index.html); change them there first, then here.

Related: the Week 1 survey ([surveys/lecture-01](../surveys/lecture-01/)),
which is the template for every weekly activity: one file per student, a
mechanical check, nothing to leak, and a chart shown at the next lecture.

## 1. Semester timeline

```mermaid
gantt
    title Fall 2026 (CS40008.01) · lectures, coursework, project, weekly tasks
    dateFormat YYYY-MM-DD
    axisFormat %b %d
    todayMarker off

    section Lectures (Wed)
    L01 Tokenization              :milestone, l01, 2026-09-09, 0d
    L02 N-gram LMs                :milestone, l02, 2026-09-16, 0d
    L03 Embeddings and PyTorch    :milestone, l03, 2026-09-23, 0d
    L04 Neural LMs and attention  :milestone, l04, 2026-09-30, 0d
    L05 Make-up class (Sat)       :milestone, l05, 2026-10-10, 0d
    L06 Transformer               :milestone, l06, 2026-10-14, 0d
    L07 Pretraining and decoding  :milestone, l07, 2026-10-21, 0d
    L08 Data preparation          :milestone, l08, 2026-10-28, 0d
    L09 Compute and scaling       :milestone, l09, 2026-11-04, 0d
    L10 Evaluation                :milestone, l10, 2026-11-11, 0d
    L11 Supervised fine-tuning    :milestone, l11, 2026-11-18, 0d
    L12 Preferences and alignment :milestone, l12, 2026-11-25, 0d
    L13 Retrieval and RAG         :milestone, l13, 2026-12-02, 0d
    L14 Efficient inference       :milestone, l14, 2026-12-09, 0d
    L15 Diffusion LMs             :milestone, l15, 2026-12-16, 0d
    L16 Agents and synthesis      :milestone, l16, 2026-12-23, 0d

    section Quizzes (in class)
    Quiz 1 :milestone, 2026-09-23, 0d
    Quiz 2 :milestone, 2026-10-14, 0d
    Quiz 3 :milestone, 2026-10-28, 0d
    Quiz 4 :milestone, 2026-11-18, 0d
    Quiz 5 :milestone, 2026-12-16, 0d

    section Assignments (private submission)
    A1 Text and probability :a1, 2026-09-16, 2026-09-30
    A2                      :a2, 2026-10-14, 2026-11-04
    A3                      :a3, 2026-11-18, 2026-12-09

    section Individual project
    Proposal due        :milestone, 2026-10-21, 0d
    Progress update due :milestone, 2026-12-02, 0d
    Final report due    :milestone, 2026-12-30, 0d

    section Weekly tasks (issues open Wed, PRs close Tue 23:59)
    W02 :t02, 2026-09-16, 2026-09-22
    W03 :t03, 2026-09-23, 2026-09-29
    W04 :t04, 2026-09-30, 2026-10-06
    W05 :t05, 2026-10-10, 2026-10-13
    W06 :t06, 2026-10-14, 2026-10-20
    W07 :t07, 2026-10-21, 2026-10-27
    W08 :t08, 2026-10-28, 2026-11-03
    W09 :t09, 2026-11-04, 2026-11-10
    W10 :t10, 2026-11-11, 2026-11-17
    W11 :t11, 2026-11-18, 2026-11-24
    W12 :t12, 2026-11-25, 2026-12-01
    W13 :t13, 2026-12-02, 2026-12-08
    W14 :t14, 2026-12-09, 2026-12-15
    W15 :t15, 2026-12-16, 2026-12-22
    W16 :t16, 2026-12-23, 2026-12-29
```

Week 5's window is short because the make-up class is on Saturday, Oct 10
and the next lecture is Wednesday, Oct 14.

## 2. One week, three roles

Every teaching week runs the same loop. The instructor's lecture preparation
for the *next* week overlaps with the students' task window for *this* week.

```mermaid
flowchart TB
    subgraph I[Instructor and TAs]
        direction LR
        I1[Wed: lecture ends;\nopen 3–5 task issues\nlabelled l02-ngram + task] --> I2[Thu–Sun: prepare next lecture;\nslides, notes, notebook\nby Sun 23:59] --> I3[Mon–Tue: check and PDF the deck;\nreview and merge green PRs] --> I4[Next Wed: show the\nresults chart, 5 minutes]
    end
    subgraph S[Students]
        direction LR
        S1[Read the issue;\ncomment: I'll take this] --> S2[Fork, branch, work in workspace/;\nadd tasks/l02-ngram/slug/submissions/username.py] --> S3[Open PR;\ntitle l02-ngram/slug: username;\nbody Related to #issue] --> S4[Red check?\npush a fix to the same branch]
    end
    subgraph C[CI on every PR]
        direction LR
        C1[Run the task's tests/test_task.py\non the submission;\nonly one file changed] --> C2{green?}
    end
    I1 -. assigns after comment .-> S1
    S3 --> C1
    S4 --> C1
    C2 -- no --> S4
    C2 -- yes --> I3
    I3 -- merged files --> R[(tasks/l02-ngram/)]
    R -- results script --> I4
```

Deadlines inside one week:

| Day | Instructor and TAs | Students |
| --- | --- | --- |
| Wed (lecture) | Open the lecture's task issues (labels `lNN-topic` + `task`); announce the deadline | Pick a task; comment to claim |
| Thu–Sun | Prepare next week's lecture: slides, notes, notebook by **Sun 23:59** (a tracking issue like #39 per lecture) | Work on the task; open the PR early so CI can run |
| Mon–Tue | Run `npm --prefix slides run check` and `pdf` on the new deck; merge green task PRs | Fix red checks; final push by **Tue 23:59** |
| Wed (next lecture) | Regenerate the chart from `tasks/lNN-topic/`; show it | See the class result |

## 3. Life of a task issue

```mermaid
stateDiagram-v2
    [*] --> Open: team opens issue with labels lNN-topic and task
    Open --> Claimed: student comments, team assigns (max 2 per student per week)
    Claimed --> InReview: PR opened with Related to #N
    InReview --> Red: CI check fails
    Red --> InReview: fix pushed to the same branch
    InReview --> Merged: CI green and team approves
    Merged --> Counted: results script runs after Tue 23:59
    Counted --> [*]
    Open --> Expired: unclaimed by Tue 23:59
    Claimed --> Expired: no PR by Tue 23:59
    Expired --> [*]: closed with label not-done
```

Late PRs are merged if correct but not counted for that week. An issue is
never closed by a student PR: PR bodies say `Related to #N`, not `Fixes #N`,
so several students can submit to the same issue.

## 4. Labels and naming rules (do not vary them)

Every issue carries **one lecture label and one type label**. Lecture labels
are stable even when dates move (the Oct 7 class moved to Oct 10, but L05 is
still L05), and zero-padding keeps them sorted.

| Lecture labels (blue) | Type labels |
| --- | --- |
| `l01-tokenization`, `l02-ngram`, `l03-embeddings`, `l04-attention`, `l05-makeup`, `l06-transformer`, `l07-pretraining`, `l08-data`, `l09-scaling`, `l10-evaluation`, `l11-sft`, `l12-alignment`, `l13-rag`, `l14-inference`, `l15-diffusion`, `l16-agents` | `task` (student exercise, many submissions), `help wanted` (one student, improves the repo), `bug`, `figure`, `survey`, `prep` (teaching team's own lecture preparation, e.g. #39), `project`, `not-done` |

| Item | Rule | Example |
| --- | --- | --- |
| Issue title | `<lecture label>: <short title>` | `l02-ngram: bigram perplexity on your own text` |
| Issue labels | one lecture label + one type label | `l02-ngram`, `task` |
| Issue body | Link to the task folder; the folder's `instruction.md` holds goal, interface, examples, checker, deadline | `tasks/l02-ngram/bigram-perplexity/` |
| Student file | `tasks/<lecture label>/<slug>/submissions/<username>.py`, username in lowercase (see [tasks/README.md](../tasks/README.md)) | `tasks/l02-ngram/bigram-perplexity/submissions/octocat.py` |
| PR title | `<lecture label>/<slug>: <username>` | `l02-ngram/bigram-perplexity: octocat` |
| PR body | `Related to #<issue>` | `Related to #45` |
| Branch in the fork | `<lecture label>-<slug>` | `l02-ngram-bigram-perplexity` |
| Deadline | Tuesday 23:59 (Asia/Shanghai) of the same week | — |

Privacy: GitHub username only, no real names or student IDs; the same rule as
the survey. Assignments A1–A3 stay on the private channel because their
solutions are shared; weekly tasks are public because every answer differs.

## 5. Kinds of task that check themselves

| Kind | What the student submits | How it is checked | Cost to the team |
| --- | --- | --- | --- |
| Measured number | A number computed on an input the student chose, with the input stated | Script recomputes from the stated input | none after setup |
| Self-checked cell | Output of a notebook cell that contains an assert | CI runs the file's declared cells | none after setup |
| Comparison table | A small table with fixed columns | CI checks the columns; a TA skims values | minutes |
| Figure or diagram | An SVG/PNG plus the script that made it | TA runs the script once | minutes |
| Fix in the course repo | A PR that changes course files (typo, bug, test) | Normal code review | review time |

Aim for at most one review-heavy task per week.

## 6. Reusing this next semester

1. Update the dates in `docs/schedule.md` and the week rows of `index.html`
   (with `assets/translations.js`), then the Gantt dates in section 1.
2. Keep the file layout, the labels, and the naming rules in section 4
   unchanged; scripts and CI depend on them.
3. Recreate the lecture and type labels from section 4 (rename the lecture
   slugs if topics change) and one milestone per lecture.
4. Re-run the survey flow in Week 1 as the first PR exercise; the results
   script `scripts/survey_results.py` shows the closed loop.
5. Open a lecture-preparation issue per week for the instructor with the
   Sunday 23:59 deadline (issue #39 is the first one).
