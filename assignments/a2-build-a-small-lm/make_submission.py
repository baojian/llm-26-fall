"""Supplied allowlist packager. Run: uv run python make_submission.py STUDENT_ID."""
import json
import re
import sys
import tomllib
import zipfile
from pathlib import Path

from validate_results import check_file

HERE = Path(__file__).resolve().parent


def main(arguments):
    if len(arguments) != 1 or not re.fullmatch(r"[A-Za-z0-9_]+", arguments[0]):
        print("Usage: make_submission.py STUDENT_ID (letters, digits, underscores)")
        return 2
    manifest = tomllib.loads((HERE / "manifest.toml").read_text())
    required = manifest["submission"]["required"]
    problems = []
    optional = []
    evidence = HERE / "evidence"
    if evidence.is_symlink():
        problems.append("evidence/ must not be a symlink")
    else:
        for pattern in manifest["submission"].get("optional", []):
            optional.extend(path.relative_to(HERE).as_posix() for path in HERE.glob(pattern))
    optional = sorted(set(optional))
    if evidence.is_dir() and not evidence.is_symlink():
        ignored = sorted(path.relative_to(HERE).as_posix() for path in evidence.iterdir()
                         if path.relative_to(HERE).as_posix() not in optional)
        if ignored:
            print("Not included from evidence/ (unsupported file type or subdirectory):\n"
                  + "\n".join(f"- {name}" for name in ignored))
    optional_limit = manifest["submission"].get("optional_max_bytes", 0)
    optional_bytes = 0
    for name in optional:
        path = HERE / name
        if not path.is_file() or path.is_symlink() or HERE not in path.resolve().parents:
            problems.append(f"unsafe optional evidence file: {name}")
            continue
        size = path.stat().st_size
        optional_bytes += size
        if optional_bytes > optional_limit or path.suffix.lower() != ".json":
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError) as exc:
            problems.append(f"invalid optional JSON: {name}: {exc}")
    if optional_bytes > optional_limit:
        problems.append(f"optional evidence exceeds {optional_limit / (1024 * 1024):g} MiB in total")
    for name in required:
        path = HERE / name
        if not path.is_file() or path.is_symlink() or HERE not in path.resolve().parents:
            problems.append(f"missing or unsafe required file: {name}")
        elif path.stat().st_size > 2 * 1024 * 1024:
            problems.append(f"file exceeds 2 MiB: {name}")
    problems.extend(check_file(HERE / "results.json"))
    report = HERE / "report.md"
    if report.is_file():
        text = report.read_text()
        if "TODO" in text:
            problems.append("report.md still contains TODO placeholders")
        if "disclosure" not in text.lower():
            problems.append("report.md needs the AI-use disclosure")
    if problems:
        print("Not packaged:\n" + "\n".join(f"- {message}" for message in problems))
        return 1
    target = HERE / manifest["submission"]["zip_name"].replace("<student-id>", arguments[0])
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in required + optional:
            archive.write(HERE / name, name)
    print(f"Wrote {target.name}: {len(required)} required and {len(optional)} optional files. Submit privately through eLearning.")
    if optional:
        print("Included optional evidence:\n" + "\n".join(f"- {name}" for name in optional))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
