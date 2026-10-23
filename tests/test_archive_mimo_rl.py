"""Exercise evidence retention, source revisions, and partial collection failures."""

import datetime as dt
import gzip
import hashlib
import http.client
import json
import urllib.parse

import pytest

from scripts import archive_mimo_rl as archive


class Source:
    def __init__(self):
        self.keys = ["pro", "flash"]
        self.notices = [{"id": "n-1", "t": 1000, "text": "original notice", "run": None}]
        self.calls = []
        self.failure = None
        self.bad_series = None
        self.stamps = [1000, 1001]
        self.benchmark = 50
        self.scalar = 0.5

    def __call__(self, url):
        parts = urllib.parse.urlsplit(url)
        endpoint = parts.path.rsplit("/", 1)[-1]
        query = urllib.parse.parse_qs(parts.query)
        key = query.get("run", [None])[0]
        self.calls.append((endpoint, key))
        if (endpoint, key) == self.failure:
            return b'{"error":"temporary unavailable"}', {}, 503
        if endpoint == "notices":
            data = {"notices": self.notices}
        elif endpoint == "benchmarks":
            data = {"benchmarks": [{"key": "eval", "results": {"pro": {"1": self.benchmark}}}]}
        elif endpoint == "runs":
            data = {"runs": [{"key": run} for run in self.keys]}
        elif endpoint == "status":
            data = {"run": {"key": key, "mode": "live"}, "step": {"last": 1},
                    "clock": {"now": 1002}, "version": "v1",
                    "events": [{"kind": "step", "step": 1, "t": 900, "redo": False},
                               {"kind": "step", "step": 1, "t": 950, "redo": True}]}
        elif endpoint == "live":
            data = {"entries": [{"t": stamp, "step": 2} for stamp in self.stamps]}
        elif endpoint == "tags":
            data = {"run": key, "version": "v1", "tags": ["dynsam/avg@n", "actor/pg_loss"]}
        elif endpoint == "series":
            tags = query["tags"][0].split(",")
            data = {"run": key, "version": "v1", "steps": [1], "walls": [1000], "run_start": 800,
                    "series": {tag: [self.scalar] for tag in tags}}
            if key == "pro" and "actor/pg_loss" in tags:
                if self.bad_series == "version":
                    data["version"] = "v2"
                elif self.bad_series == "axes":
                    data["walls"] = [1001]
                elif self.bad_series == "missing":
                    data["series"].pop("actor/pg_loss")
                elif self.bad_series == "length":
                    data["series"]["actor/pg_loss"] = []
        else:
            raise AssertionError(endpoint)
        return json.dumps(data).encode(), {"date": "source clock"}, 200


def test_archive_preserves_exact_bytes_and_repeated_step_events(tmp_path):
    source = Source()
    report = archive.capture(tmp_path, source)
    assert report["complete"]
    assert report["runs"]["pro"]["tag_count"] == 2
    for request in report["requests"]:
        raw = gzip.decompress((tmp_path / request["object"]).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == request["sha256"]
        assert len(raw) == request["bytes"]
        if "/status?" in request["url"]:
            events = json.loads(raw)["events"]
            assert [e["step"] for e in events] == [1, 1]
            assert [e["redo"] for e in events] == [False, True]


def test_series_cache_does_not_skip_notices_or_new_evaluations(tmp_path):
    source = Source()
    first = archive.capture(tmp_path, source)
    original_objects = set((tmp_path / "objects").rglob("*.gz"))
    source.calls.clear()
    source.benchmark = 60
    second = archive.capture(tmp_path, source)
    assert second["complete"]
    assert not any(endpoint == "series" for endpoint, _ in source.calls)
    assert {endpoint for endpoint, _ in source.calls} >= {"notices", "benchmarks", "status", "live", "tags"}
    assert second["runs"]["pro"]["series"] == "cached"
    assert original_objects <= set((tmp_path / "objects").rglob("*.gz"))
    assert len(list((tmp_path / "observations").rglob("*.json"))) == 2
    before = next(r["sha256"] for r in first["requests"] if r["url"].endswith("benchmarks"))
    after = next(r["sha256"] for r in second["requests"] if r["url"].endswith("benchmarks"))
    assert before != after


def test_notice_edit_and_removal_do_not_erase_history(tmp_path):
    source = Source()
    archive.capture(tmp_path, source)
    source.notices[0]["text"] = "corrected notice"
    archive.capture(tmp_path, source)
    source.notices = []
    report = archive.capture(tmp_path, source)
    notices = json.loads((tmp_path / "notices.json").read_text())["revisions"]
    assert {item["notice"]["text"] for item in notices.values()} == {"original notice", "corrected notice"}
    assert report["notice_count"] == 0 and report["notice_revisions_retained"] == 2
    assert "original notice" in (tmp_path / "notices.md").read_text()


def test_partial_failure_is_saved_and_retried_without_losing_last_success(tmp_path):
    source = Source()
    archive.capture(tmp_path, source)
    previous = json.loads((tmp_path / "state.json").read_text())["last_successful"]
    source.failure = ("series", "pro")
    failed = archive.capture(tmp_path, source, force_series=True)
    assert not failed["complete"]
    assert failed["runs"]["flash"]["series"] == "fetched"
    assert json.loads((tmp_path / "state.json").read_text())["last_successful"] == previous
    error_request = next(r for r in failed["requests"] if r.get("status") == 503)
    assert b"temporary unavailable" in gzip.decompress((tmp_path / error_request["object"]).read_bytes())
    source.failure = None
    source.calls.clear()
    recovered = archive.capture(tmp_path, source)
    assert recovered["complete"]
    assert ("series", "pro") in source.calls


@pytest.mark.parametrize("problem", ["version", "axes", "missing", "length"])
def test_mixed_or_incomplete_series_are_not_marked_complete(tmp_path, monkeypatch, problem):
    monkeypatch.setattr(archive, "BATCH_SIZE", 1)
    source = Source()
    source.bad_series = problem
    failed = archive.capture(tmp_path, source)
    assert not failed["complete"]
    assert failed["runs"]["pro"]["series"] == "partial"
    assert "series_manifest" not in failed["runs"]["pro"]
    assert failed["runs"]["flash"]["series"] == "fetched"
    source.bad_series = None
    assert archive.capture(tmp_path, source)["complete"]


def test_discovery_failure_uses_known_runs_and_retains_notices(tmp_path):
    source = Source()
    archive.capture(tmp_path, source)
    source.failure = ("runs", None)
    source.notices.append({"id": "n-2", "t": 1002, "text": "new notice"})
    report = archive.capture(tmp_path, source)
    assert not report["complete"]
    assert set(report["runs"]) == {"pro", "flash"}
    assert report["notice_revisions_retained"] == 2


def test_disappearing_run_does_not_delete_its_objects(tmp_path):
    source = Source()
    first = archive.capture(tmp_path, source)
    objects = set((tmp_path / "objects").rglob("*.gz"))
    source.keys = ["pro"]
    second = archive.capture(tmp_path, source)
    assert second["complete"] and set(second["runs"]) == {"pro"}
    assert objects <= set((tmp_path / "objects").rglob("*.gz"))
    assert (tmp_path / first["runs"]["flash"]["series_manifest"]).exists()


def test_possible_sampler_gap_is_recorded_without_claiming_missing_events(tmp_path):
    source = Source()
    archive.capture(tmp_path, source)
    source.stamps = [2000, 2001]
    report = archive.capture(tmp_path, source)
    assert report["complete"]
    assert {w["run"] for w in report["warnings"]} == {"pro", "flash"}
    assert all(w["kind"] == "sampler_window_no_overlap" for w in report["warnings"])


def test_untrusted_run_keys_cannot_escape_archive_directory(tmp_path):
    source = Source()
    source.keys = ["../../outside"]
    root = tmp_path / "archive"
    report = archive.capture(root, source)
    assert report["complete"]
    assert not (tmp_path / "outside").exists()
    assert any("run=..%2F..%2Foutside" in r["url"] for r in report["requests"])
    assert len(list((root / "series").glob("*.json"))) == 1


def test_hourly_refresh_rechecks_data_even_if_version_is_unchanged(tmp_path, monkeypatch):
    source = Source()
    archive.capture(tmp_path, source)
    later = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=2)).isoformat()
    monkeypatch.setattr(archive, "now", lambda: later)
    source.calls.clear()
    report = archive.capture(tmp_path, source)
    assert report["complete"]
    assert ("series", "pro") in source.calls and ("series", "flash") in source.calls


def test_same_version_correction_keeps_old_catalog_and_values(tmp_path):
    source = Source()
    first = archive.capture(tmp_path, source)
    catalog = tmp_path / first["runs"]["pro"]["series_manifest"]
    original = catalog.read_bytes()
    source.scalar = 0.75
    second = archive.capture(tmp_path, source, force_series=True)
    assert second["complete"]
    assert first["runs"]["pro"]["series_manifest"] != second["runs"]["pro"]["series_manifest"]
    assert catalog.read_bytes() == original
    batch = json.loads(original)["batches"][0]
    old = json.loads(gzip.decompress((tmp_path / batch["object"]).read_bytes()))
    assert old["series"]["dynsam/avg@n"] == [0.5]


def test_invalid_json_is_kept_and_other_endpoints_still_work(tmp_path):
    source = Source()
    def fetch(url):
        return (b"upstream maintenance", {}, 200) if url.endswith("notices") else source(url)
    report = archive.capture(tmp_path, fetch)
    assert not report["complete"]
    assert report["runs"]["pro"]["series"] == "fetched"
    request = report["requests"][0]
    assert gzip.decompress((tmp_path / request["object"]).read_bytes()) == b"upstream maintenance"


def test_off_origin_redirect_is_rejected():
    handler = archive.SameOriginRedirect()
    with pytest.raises(ValueError, match="outside"):
        handler.redirect_request(None, None, 302, "", {}, "http://127.0.0.1/private")


def test_interrupted_http_body_does_not_discard_other_evidence(tmp_path):
    source = Source()
    def fetch(url):
        if url.endswith("notices"):
            raise http.client.IncompleteRead(b"partial", 100)
        return source(url)
    report = archive.capture(tmp_path, fetch)
    assert not report["complete"]
    assert report["runs"]["pro"]["series"] == "fetched"
    assert "IncompleteRead" in report["errors"][0]["message"]
    assert (tmp_path / "LATEST.json").exists()
