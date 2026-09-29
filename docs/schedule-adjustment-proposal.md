# October 10 schedule adjustment: review proposal

**Prepared:** September 29, 2026. **Status:** draft for instructor review.

The instructor requested that Lecture 04 remain on September 30, that the
Transformer lecture move to the October 10 make-up meeting, and that later
topics advance accordingly. The instructor also requested corresponding
assessment-date proposals for review. The tables below distinguish the
previous dates from the proposal previewed on this branch.

Assessment changes take effect after instructor approval and a synchronized
update to the published course page and eLearning. Until then, students follow
the currently published dates. A1's September 30 deadline remains in place.

## Lecture sequence

| Lecture | Topic | Previous date | Proposed date |
| ---: | --- | --- | --- |
| 04 | Neural language models and attention | Sep 30 | Sep 30 |
| 05 | The Transformer as a working model | Oct 14 | **Sat, Oct 10** |
| 06 | Pretraining and decoding | Oct 21 | Oct 14 |
| 07 | Data preparation and quality | Oct 28 | Oct 21 |
| 08 | Compute budgets and scaling | Nov 4 | Oct 28 |
| 09 | Evaluation and experimental design | Nov 11 | Nov 4 |
| 10 | Supervised fine-tuning | Nov 18 | Nov 11 |
| 11 | Preferences, rewards, and alignment | Nov 25 | Nov 18 |
| 12 | Retrieval and RAG | Dec 2 | Nov 25 |
| 13 | Efficient inference | Dec 9 | Dec 2 |
| 14 | Diffusion language models | Dec 16 | Dec 9 |
| 15 | Agents and tool use | Dec 23, combined with synthesis | Dec 16 |
| 16 | Research synthesis and project revision | Dec 23, combined with agents | Dec 23 |

All dates are in 2026. Lectures 01–03 retain their dates and materials.
October 7 has no meeting; October 10 replaces it. The university make-up notice
sets the time and room. There are 16 meetings and 48 academic periods.

The additional session gives agents a full meeting on December 16. On
December 23, students connect course methods to their project evidence and
revise claims, comparisons, limitations, and writing. This remains guided
classroom practice within the individual written-project format.

## Proposed assessment dates

All submission deadlines are **23:59, Asia/Shanghai (UTC+08:00)**. Quizzes
take place in class and cover the common core already taught.

| Item | Previously published | Proposal | Reason |
| --- | --- | --- | --- |
| Quiz 1 | Sep 23 | Sep 23 | Retain the past scheduled date |
| Quiz 2 | Oct 14 | **Sat, Oct 10** | Keep it with the Transformer meeting; assess prior embeddings and attention material |
| Quiz 3 | Oct 28 | Oct 21 | Keep it with the data meeting, after pretraining |
| Quiz 4 | Nov 18 | Nov 11 | Keep it with SFT, after evaluation |
| Quiz 5 | Dec 16 | Dec 9 | Keep it with diffusion, after retrieval and inference |
| A1 release → due | Sep 16 → Sep 30 | Sep 16 → Sep 30 | Lecture 04 and the released assignment stay in place |
| A2 release → due | Oct 14 → Nov 4 | **Oct 10 → Sat, Oct 31** | Release with Transformers and preserve the full 21-day work window |
| A3 release → due | Nov 18 → Dec 9 | Nov 11 → Dec 2 | Release with SFT and preserve the full 21-day work window |
| Project proposal | Oct 21 | Oct 14 | Align with the pretraining meeting and establish the experiment plan earlier |
| Project progress update | Dec 2 | Nov 25 | Retain six weeks after the proposal and allow more time for feedback |
| Final report, code, and experiment record | Dec 30 | Dec 30 | Use December 23 for revision before the existing final deadline |

Moving A2's deadline to October 28 would leave only 18 days because the
make-up meeting is on a Saturday. October 31 preserves three weeks. Assignment
deadlines remain separate from project deadlines. The proposal advances two
project checkpoints by seven days and gives students five weeks between the
progress update and final submission.

Weights remain quizzes 10%, assignments 45%, and individual project 45%.
All five quizzes count at 2% each; assignments count at 15% each. Undergraduate
and master's students share the same required work, rubric, points, and
maximum scores. Graded submissions remain private through eLearning.

## Review and publication

- [ ] Approve the proposed quiz dates, including Quiz 2 on Saturday, October 10.
- [ ] Approve A2's Saturday deadline and the A3 dates.
- [ ] Approve the earlier project proposal and progress-update dates.
- [ ] Confirm the October 10 time and room against the university notice.
- [ ] Synchronize approved dates with eLearning and publish the course-page
  changes together. Record approval here before merging the schedule PR.

The branch updates the English and Chinese course page, dated schedule,
participation timeline, preparation maps, task-proposal form, and forward
references in Lectures 02 and 04. Released A1 files and its announcement
retain their dates. It contains no student submissions or grading data.

### GitHub follow-up after approval

Update the future lecture labels using the mapping below. Rename existing
labels so linked issues retain their topic association; update each label's
lecture-number description. Published task paths for Lectures 01–04 stay
stable. The task-proposal form and participation guide on this branch use the
new names.

| Existing label | New label |
| --- | --- |
| `l05-makeup` | `l16-synthesis` |
| `l06-transformer` | `l05-transformer` |
| `l07-pretraining` | `l06-pretraining` |
| `l08-data` | `l07-data` |
| `l09-scaling` | `l08-scaling` |
| `l10-evaluation` | `l09-evaluation` |
| `l11-sft` | `l10-sft` |
| `l12-alignment` | `l11-alignment` |
| `l13-rag` | `l12-rag` |
| `l14-inference` | `l13-inference` |
| `l15-diffusion` | `l14-diffusion` |
| `l16-agents` | `l15-agents` |

Update the future-lecture references in the
[Lecture 04 preparation issue](https://github.com/baojian/llm-26-fall/issues/234)
and the course-placement dates in the
[MiMo case-study issue](https://github.com/baojian/llm-26-fall/issues/177).
The latter now supports SFT on November 11 and alignment on November 18.
These issue edits and label changes remain pending with the schedule review.
