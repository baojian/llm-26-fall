# Preserved MiMo operator notices

All observed revisions are retained, including notices later removed upstream.
Publication times below use the source Unix timestamp; first/last capture times are in [notices.json](notices.json).

## 2026-09-16T17:58:17.495634+00:00

ID: `n-5ef2b1` · revision `9926f4747120`

> we have updated the latest deepswe results for flash step 12 & pro step 8. we will keep posting as the offline evaluation results come out.

## 2026-09-16T20:08:00.378206+00:00

ID: `n-92030c` · revision `b3d1b6d38a20`

> the mimo-v2.6-pro run is restarting due to a vram issue on one node.

## 2026-09-17T02:27:38.930446+00:00

ID: `n-b60f90` · revision `a872ef401d5d`

> we restarted the flash run from step 15. reason: a type of infra error on one of datasets was not correctly detected over the past ~3 hours.

## 2026-09-17T12:20:11.047444+00:00

ID: `n-15ac72` · revision `0d243664f635`

> there was a network connectivity issue between the pro training cluster and the grader deployment. we have restarted the run. we also removed the cyber dataset from the upcoming pro run, since we observed some bad patterns in the rollout logs.

## 2026-09-18T03:49:42.889267+00:00

ID: `n-4e29eb` · revision `2f32df3b3768`

> the pro run restarted at step 17 due to a GPU OOM issue caused by expert load imbalance. we have adjusted the training parallelism strategy.

## 2026-09-19T10:14:26.861832+00:00

ID: `n-6b9c4a` · revision `8ebf15111d2a`

> we filtered out tasks that are relatively easy for the current pro model.

