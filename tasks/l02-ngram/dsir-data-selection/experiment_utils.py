"""Supplied I/O, reproducibility, and analysis helpers (no third-party packages)."""

import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_submission(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("task_submission", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "solve", None)):
        raise ValueError("submission must define solve(case)")
    return module


def uniforms_for(ids, seed):
    """Supplied per-ID random inputs; 52 bits keep values strictly inside (0, 1)."""
    values = {}
    for doc_id in ids:
        payload = json.dumps([seed, doc_id], ensure_ascii=False).encode("utf-8")
        bits = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") >> 12
        values[doc_id] = (bits + 0.5) / 2**52
    return values


def provenance(submission, folder, config):
    folder = Path(folder).resolve()
    try:
        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(folder), *args], stderr=subprocess.DEVNULL
            ).decode().strip()
        revision = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain"))
        diff_hash = hashlib.sha256(git("diff", "HEAD").encode()).hexdigest()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision, dirty, diff_hash = None, None, None
    return {
        "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "command": [sys.executable, *sys.argv],
        "git_revision": revision, "git_dirty": dirty, "git_diff_sha256": diff_hash,
        "submission_sha256": sha256(submission),
        "files_sha256": {
            str(path.relative_to(folder)): sha256(path)
            for path in sorted(folder.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".json"}
            and "__pycache__" not in path.parts and "submissions" not in path.parts
        },
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "device": "CPU", "model": None,
        "configuration": config,
    }


def write_report(path, report):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    report["artifact_path"] = str(path)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Report: {path}")
