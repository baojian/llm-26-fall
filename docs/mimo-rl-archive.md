# MiMo RL public training archive

The [MiMo training dashboard](https://mimo.xiaomi.com/rl/#overview) exposes a
production post-training case study for Lectures 11 and 12. The instructor's
[case-study issue #177](https://github.com/baojian/llm-26-fall/issues/177) records
the initial state, six operator notices, and questions for lecture preparation.

## Preserved evidence

Open the [archive branch](https://github.com/baojian/llm-26-fall/tree/archive/mimo-rl)
for the data and its latest capture status. Start with
[preserved notices](https://github.com/baojian/llm-26-fall/blob/archive/mimo-rl/notices.md)
or [the latest observation](https://github.com/baojian/llm-26-fall/blob/archive/mimo-rl/LATEST.json).

The [recorder](../scripts/archive_mimo_rl.py) saves the exact response bytes from
these unauthenticated public endpoints:

| Endpoint under `https://mimo.xiaomi.com/rl/api/` | Evidence |
| --- | --- |
| `notices` | Operator notices, IDs, publication timestamps, and revisions |
| `runs` | Run discovery, metric descriptions, chart configuration, and dataset categories |
| `benchmarks` | All currently published offline evaluation checkpoints, including late results |
| `status?run=...` | Progress, restarts, recent events, token and sequence counts, timing, and publisher cost estimates |
| `live?run=...` | The currently exposed sampler reports, including step and dataset information |
| `tags?run=...` | Complete scalar metric inventory and source version |
| `series?run=...&v=...&tags=...` | All exposed historical scalar arrays, their steps, wall times, and run start |

The baseline from September 20, 2026 includes 2,029 Pro and 2,062 Flash scalar
series, all six supplied notices, and three benchmark suites. The instructor's
original relative-age notice transcription is retained under `provenance/`;
those relative ages are not used to infer exact publication times.

This is a specifically requested archive of public third-party evidence. It
belongs on the separate `archive/mimo-rl` branch. Course source, the recorder,
and tests remain on `main`; student submissions, grades, private training data,
credentials, model weights, and local experiment outputs are not collected.

## Capture schedule and retention

The [GitHub Actions workflow](https://github.com/baojian/llm-26-fall/actions/workflows/archive-mimo-rl.yml)
requests a capture every five minutes, starting at minute 2 of each hour. It runs
on GitHub and does not require the instructor's computer to remain awake. Only
this course repository runs scheduled captures; forks skip the job.

Every capture reads notices, benchmarks, run configuration, status, sampler
reports, and tags. It requests every scalar series in batches of at most 96
when the source version or tag inventory changes, and at least hourly when a
successful capture occurs. A manual run can force a full refresh. Ended runs
remain eligible for collection while publicly listed, so later evaluation
results can still be retained. Source requests carry no GitHub credentials.

| Archive path | Meaning |
| --- | --- |
| `objects/<prefix>/<sha256>.json.gz` | Immutable gzip objects; SHA-256 covers the original uncompressed response bytes |
| `observations/YYYY-MM-DD/*.json` | Immutable capture records with source URLs, acquisition times, selected response headers, object hashes, errors, and warnings |
| `series/<hash>.json` | Immutable catalogs linking a complete run/version capture to its metric batches |
| `notices.json`, `notices.md` | Accumulated notice revisions with first/last observation times and a readable chronology |
| `state.json` | Current cache pointers, sampler boundaries, last attempt, and last complete capture |
| `LATEST.json`, `README.md` | Current capture summary and entry points |

An upstream correction or disappearance never deletes earlier objects,
observations, catalogs, or notice revisions. Identical bodies are deduplicated.
Repeated training steps and restart events retain their original order and
wall times. The archive does not average, rewrite, or replace earlier values.

A failed endpoint does not prevent attempts to collect the other endpoints.
Raw error responses and partial observations are published before the workflow
reports failure. A failed series refresh is retried at the next capture.
The recorder rejects mixed source versions, inconsistent batch axes, missing
metrics, and mismatched array lengths as incomplete captures. Read `errors`
and `warnings` before interpreting a snapshot; `complete: true` means the
requested capture succeeded, not that all historical events are available.

## Retrieve and inspect

Clone the data separately from the course working tree:

```sh
git clone --single-branch --branch archive/mimo-rl \
  https://github.com/baojian/llm-26-fall.git /tmp/mimo-rl-history
```

From the course checkout, read a captured response and verify its exact bytes:

```sh
uv run python - <<'PY'
import gzip
import hashlib
import json
from pathlib import Path

root = Path('/tmp/mimo-rl-history')
latest = json.loads((root / 'LATEST.json').read_text())
request = next(r for r in latest['requests'] if r.get('status') == 200)
raw = gzip.decompress((root / request['object']).read_bytes())
assert hashlib.sha256(raw).hexdigest() == request['sha256']
print(request['url'], request['fetched_at'])
print(json.dumps(json.loads(raw), indent=2, ensure_ascii=False))
PY
```

For charts, follow a run's `series_manifest` in an observation, then each batch's
`object`. Each decompressed batch contains `steps`, `walls`, `run_start`, and a
`series` dictionary keyed by metric name. Preserve missing/null values and align
by the recorded axes. A cached observation points to an earlier immutable
catalog; use its `captured_at` to distinguish metric freshness from capture time.

## Operate the recorder

Inspect recent workflow runs or request a full capture with GitHub CLI:

```sh
gh run list --repo baojian/llm-26-fall --workflow archive-mimo-rl.yml --limit 10
gh workflow run archive-mimo-rl.yml --repo baojian/llm-26-fall -f force_series=true
```

For local troubleshooting, create a separate checkout of the data branch and
run the recorder from the course checkout. This changes local archive files;
it does not push them automatically. Avoid concurrent manual pushes while the
scheduled workflow is publishing.

```sh
uv run python scripts/archive_mimo_rl.py --output /tmp/mimo-rl-history --force-series
uv run python -m pytest tests/test_archive_mimo_rl.py
```

Workflow failures are visible in Actions; the workflow uses the normal GitHub
notification settings. When the source is retired or collection should stop,
disable the workflow without deleting the evidence branch:

```sh
gh workflow disable archive-mimo-rl.yml --repo baojian/llm-26-fall
```

Use `gh workflow enable` with the same arguments to resume it. No automatic
expiration or deletion policy is configured for archived evidence.

## Coverage limits

This is a best-effort record beginning with the initial capture. Earlier
deleted notices, overwritten run attempts, private rollout text, unpublished
checkpoints, and internal implementation details cannot be recovered from
these endpoints. A public scalar name does not establish the training
algorithm, exact loss equation, hardware count, or a causal explanation.

In the initial captures, each status response exposed 30 recent events and each
sampler response exposed 60 reports, covering roughly 19–29 minutes. The
browser displayed only the latest 40 combined feed items. The recorder uses
the API windows and marks potentially missed sampler reports when consecutive
windows do not overlap; this cannot prove how many events were missed. Changes
that appear and disappear between captures may never be observed.

Five minutes is a requested schedule, not a delivery guarantee.
[GitHub documents](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)
that scheduled jobs may be delayed or dropped under load, run only from the
default branch, and can be disabled after 60 days of inactivity in a public
repository. Review the latest successful capture periodically and before the
post-training lectures. Repository or source availability, rate limits, schema
changes, and Actions limits can also interrupt collection. A failed push is
visible as a failed workflow and may require manual recovery.

No fixed 24-hour notice expiry was verified: the initial API response still
contained notices more than three days old. Retention upstream may change;
the course archive keeps every notice revision it actually observes.
