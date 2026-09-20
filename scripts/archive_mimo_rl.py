"""Archive the public MiMo RL JSON endpoints without overwriting earlier evidence.

Standard library only. The output belongs on the archive/mimo-rl data branch,
not in the course's main tree. See docs/mimo-rl-archive.md and issue #177.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

SOURCE = "https://mimo.xiaomi.com/rl/"
BATCH_SIZE = 96  # Same maximum batch size as the public dashboard client.
MAX_BYTES = 16 * 1024 * 1024
SERIES_REFRESH_SECONDS = 3600
UTC = dt.timezone.utc


def now() -> str:
    return dt.datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def read_json(path: Path, default):
    return json.loads(path.read_bytes()) if path.exists() else default


def store_object(root: Path, raw: bytes) -> dict:
    """Keep exact response bytes in an immutable, content-addressed gzip file."""
    digest = hashlib.sha256(raw).hexdigest()
    relative = Path("objects") / digest[:2] / f"{digest}.json.gz"
    path = root / relative
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(raw, mtime=0))
    return {"sha256": digest, "object": relative.as_posix(), "bytes": len(raw)}


class SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl)[:2] != urllib.parse.urlsplit(SOURCE)[:2]:
            raise ValueError("Refusing a redirect outside the public source origin")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url: str) -> tuple[bytes, dict, int]:
    """Bound response size and retry transient failures; never send credentials."""
    opener = urllib.request.build_opener(SameOriginRedirect())
    request = urllib.request.Request(url, headers={
        "User-Agent": "Fudan-LLM-course-public-archive/1.0 (+https://github.com/baojian/llm-26-fall/issues/177)",
        "Accept": "application/json", "Accept-Encoding": "identity", "Cache-Control": "no-cache",
    })
    for attempt in range(3):
        try:
            try:
                response = opener.open(request, timeout=30)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                raw = response.read(MAX_BYTES + 1)
                if len(raw) > MAX_BYTES:
                    raise ValueError("Response exceeds the archive's 16 MiB limit")
                headers = {k.lower(): v for k, v in response.headers.items()
                           if k.lower() in {"date", "content-type", "etag", "last-modified"}}
                status = response.code
            if status not in {429, 500, 502, 503, 504} or attempt == 2:
                return raw, headers, status
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("Download did not return a response")


class Recorder:
    def __init__(self, root: Path, fetch=download):
        self.root = root
        self.fetch = fetch
        self.started = now()
        self.observation = {
            "schema_version": 1, "source": SOURCE, "started_at": self.started,
            "finished_at": None, "complete": False, "requests": [], "runs": {},
            "errors": [], "warnings": [],
        }
        self.state = read_json(root / "state.json", {"runs": {}, "run_keys": []})
        self.state.setdefault("series_cache", {})

    def error(self, context: str, error) -> None:
        self.observation["errors"].append({"context": context, "message": str(error)})

    def get(self, endpoint: str, **params):
        url = SOURCE + "api/" + endpoint
        if params:
            url += "?" + urllib.parse.urlencode(params)
        record = {"url": url, "fetched_at": now()}
        self.observation["requests"].append(record)
        try:
            raw, headers, status = self.fetch(url)
            record.update(store_object(self.root, raw))
            record.update(headers=headers, status=status)
            if status != 200:
                raise ValueError(f"HTTP {status}")
            data = json.loads(raw)
            if not isinstance(data, dict) or data.get("error"):
                raise ValueError("Endpoint returned an error or a non-object payload")
            return data, record
        except (OSError, ValueError, TypeError, urllib.error.URLError, http.client.HTTPException) as error:
            record["error"] = str(error)
            self.error(endpoint + (f"/{params['run']}" if "run" in params else ""), error)
            return None, record

    def notices(self, data: dict) -> None:
        notices = data["notices"]
        if not isinstance(notices, list):
            raise ValueError("notices must be a list")
        index = read_json(self.root / "notices.json", {"revisions": {}})
        for notice in notices:
            if not isinstance(notice, dict) or not isinstance(notice.get("text"), str):
                raise ValueError("Malformed notice")
            revision = hashlib.sha256(json_bytes(notice)).hexdigest()
            item = index["revisions"].setdefault(revision, {
                "notice": notice, "first_seen": self.started, "source": SOURCE + "api/notices",
            })
            item["last_seen"] = self.started
        write_json(self.root / "notices.json", index)
        self.observation["notice_count"] = len(notices)
        self.observation["notice_revisions_retained"] = len(index["revisions"])
        lines = ["# Preserved MiMo operator notices", "", "All observed revisions are retained, including notices later removed upstream.",
                 "Publication times below use the source Unix timestamp; first/last capture times are in [notices.json](notices.json).", ""]
        for revision, item in sorted(index["revisions"].items(), key=lambda p: p[1]["notice"].get("t") or 0):
            notice = item["notice"]
            stamp = dt.datetime.fromtimestamp(notice["t"], UTC).isoformat() if isinstance(notice.get("t"), (int, float)) else "unknown publication time"
            lines += [f"## {stamp}", "", f"ID: `{notice.get('id', 'not supplied')}` · revision `{revision[:12]}`", ""]
            lines += ["> " + line.replace("<", "&lt;").replace(">", "&gt;") for line in notice["text"].splitlines()]
            lines.append("")
        (self.root / "notices.md").write_text("\n".join(lines) + "\n")

    def run(self, key: str, force_series: bool) -> None:
        status, _ = self.get("status", run=key)
        live, _ = self.get("live", run=key)
        tags, _ = self.get("tags", run=key)
        report = self.observation["runs"].setdefault(key, {})
        previous = self.state["runs"].get(key, {})
        if live is not None:
            entries = live["entries"]
            if not isinstance(entries, list):
                raise ValueError("live.entries must be a list")
            stamps = [e["t"] for e in entries if isinstance(e, dict) and isinstance(e.get("t"), (int, float))]
            report["sampler_reports"] = len(entries)
            if stamps:
                report.update(sampler_first=min(stamps), sampler_last=max(stamps))
                if previous.get("sampler_last") is not None and min(stamps) > previous["sampler_last"]:
                    self.observation["warnings"].append({
                        "run": key, "kind": "sampler_window_no_overlap",
                        "previous_last": previous["sampler_last"], "current_first": min(stamps),
                        "message": "Possible collection gap; the two retained windows have no report in common.",
                    })
        if status is not None:
            report.update(mode=status["run"]["mode"], step=status["step"]["last"],
                          version=status["version"], source_clock=status["clock"]["now"])
        # Keep the previous sampler boundary if a live request failed.
        self.state["runs"][key] = {**previous, **report, "last_seen": self.started}
        if status is None or tags is None:
            return
        names = tags["tags"]
        if not isinstance(names, list) or not all(isinstance(tag, str) and tag for tag in names):
            raise ValueError("Malformed tag inventory")
        if len(names) != len(set(names)) or not names:
            raise ValueError("Tag inventory is empty or contains duplicates")
        version = tags["version"]
        if tags["run"] != key or status["run"]["key"] != key or status["version"] != version:
            raise ValueError("Run/version changed during status and tag acquisition; retry next capture")
        report["tag_count"] = len(names)
        identity = hashlib.sha256(json_bytes([key, version, names])).hexdigest()
        cached = self.state["series_cache"].get(identity, {})
        relative = Path(cached["manifest"]) if cached else None
        existing = read_json(self.root / relative, {}) if relative else {}
        age = (dt.datetime.fromisoformat(self.started) - dt.datetime.fromisoformat(existing["captured_at"])).total_seconds() if existing else None
        if existing and not cached.get("needs_retry") and not force_series and 0 <= age < SERIES_REFRESH_SECONDS:
            report.update(series="cached", series_manifest=relative.as_posix())
            return
        batches = []
        axes = None
        failed = False
        for offset in range(0, len(names), BATCH_SIZE):
            chunk = names[offset:offset + BATCH_SIZE]
            data, record = self.get("series", run=key, v=version, tags=",".join(chunk))
            if data is None:
                failed = True
                continue
            try:
                if data["run"] != key or data["version"] != version:
                    raise ValueError("Series version changed during collection")
                current_axes = (data["steps"], data["walls"], data["run_start"])
                if len(data["steps"]) != len(data["walls"]) or (axes is not None and current_axes != axes):
                    raise ValueError("Series batches have inconsistent step/wall-time axes")
                axes = current_axes
                for tag in chunk:
                    values = data["series"][tag]
                    if values is not None and (not isinstance(values, list) or len(values) != len(data["steps"])):
                        raise ValueError(f"Series length mismatch: {tag}")
                batches.append({"tags": chunk, "object": record["object"], "sha256": record["sha256"]})
            except (KeyError, ValueError, TypeError) as error:
                failed = True
                self.error(f"series/{key}/batch-{offset // BATCH_SIZE}", error)
        report["series"] = "partial" if failed else "fetched"
        if not failed:
            manifest = {"run": key, "version": version, "captured_at": self.started,
                        "tag_count": len(names), "batches": batches}
            relative = Path("series") / f"{hashlib.sha256(json_bytes(manifest)).hexdigest()}.json"
            write_json(self.root / relative, manifest)
            self.state["series_cache"][identity] = {"manifest": relative.as_posix()}
            report["series_manifest"] = relative.as_posix()
        elif existing:
            # A failed forced refresh must be retried even inside the cache TTL.
            self.state["series_cache"][identity]["needs_retry"] = True

    def finish(self) -> dict:
        self.observation["finished_at"] = now()
        self.observation["complete"] = not self.observation["errors"]
        capture_id = self.started.replace(":", "-") + "-" + uuid.uuid4().hex[:8]
        relative = Path("observations") / self.started[:10] / f"{capture_id}.json"
        write_json(self.root / relative, self.observation)
        self.state["last_attempt"] = relative.as_posix()
        if self.observation["complete"]:
            self.state["last_successful"] = relative.as_posix()
        write_json(self.root / "state.json", self.state)
        write_json(self.root / "LATEST.json", {"observation": relative.as_posix(), **self.observation})
        lines = ["# MiMo RL public evidence archive", "",
                 "Source: https://mimo.xiaomi.com/rl/", "",
                 "Course case study: https://github.com/baojian/llm-26-fall/issues/177", "",
                 "Recorder and retrieval guide: https://github.com/baojian/llm-26-fall/blob/main/docs/mimo-rl-archive.md", "",
                 f"Latest attempt: {self.started} · complete: {self.observation['complete']}", "",
                 f"[Latest observation]({relative.as_posix()}) · [Latest summary](LATEST.json) · [All observations](observations/)", "",
                 "[Preserved notices](notices.md) · [Notice revisions](notices.json) · [Series catalogs](series/)", "",
                 "| Run | Mode | Last completed step | Public tags | Series acquisition |", "| --- | --- | ---: | ---: | --- |"]
        for key, report in self.observation["runs"].items():
            lines.append(f"| {key} | {report.get('mode', 'unavailable')} | {report.get('step', '—')} | {report.get('tag_count', '—')} | {report.get('series', 'unavailable')} |")
        lines += ["", "This branch contains only public third-party source evidence requested by the instructor.",
                  "Response objects are gzip-compressed, keyed by SHA-256 of their exact uncompressed bytes.",
                  "Old objects, observations, and notice revisions are retained. Mutable indexes point to the latest observations.",
                  "This is a best-effort archive of exposed data from the first capture onward; it cannot recover already-deleted or private history.",
                  "See each observation's errors and warnings before treating a capture as complete.", ""]
        (self.root / "README.md").write_text("\n".join(lines))
        return self.observation


def capture(root: Path, fetch=download, force_series: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(root, fetch)
    # Preserve the most ephemeral evidence even if discovery or a run fails.
    notices, _ = recorder.get("notices")
    if notices is not None:
        try:
            recorder.notices(notices)
        except (KeyError, ValueError, TypeError, OverflowError) as error:
            recorder.error("notices/schema", error)
    benchmarks, _ = recorder.get("benchmarks")
    if benchmarks is not None and not isinstance(benchmarks.get("benchmarks"), list):
        recorder.error("benchmarks/schema", "Missing benchmark list")
    config, _ = recorder.get("runs")
    keys = recorder.state["run_keys"]
    if config is not None:
        try:
            discovered = [run["key"] for run in config["runs"]]
            if not discovered or not all(isinstance(key, str) and 0 < len(key) <= 100 for key in discovered):
                raise ValueError("Missing or malformed run keys")
            keys = list(dict.fromkeys(discovered))
            recorder.state["run_keys"] = keys
        except (KeyError, ValueError, TypeError) as error:
            recorder.error("runs/schema", error)
    for key in keys:
        try:
            recorder.run(key, force_series)
        except (KeyError, ValueError, TypeError, OverflowError) as error:
            recorder.error(f"run/{key}/schema", error)
    return recorder.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="checkout of archive/mimo-rl")
    parser.add_argument("--force-series", action="store_true", help="ignore the one-hour series cache")
    args = parser.parse_args()
    report = capture(args.output, force_series=args.force_series)
    print(json.dumps({"complete": report["complete"], "requests": len(report["requests"]),
                      "runs": report["runs"], "errors": report["errors"], "warnings": report["warnings"]}))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
