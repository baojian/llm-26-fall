"""Supplied allowlist packager. Run: uv run python make_submission.py STUDENT_ID."""
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
        for name in required:
            archive.write(HERE / name, name)
    print(f"Wrote {target.name}: {len(required)} allowlisted files. Submit privately through eLearning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
