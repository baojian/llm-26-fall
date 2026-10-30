"""Check the required files, results.json, and the report, then write the eLearning zip.

    python make_submission.py <student-id>

Reads the allowlist and the results layout from manifest.toml, refuses to
package if anything is missing or a TODO from the report template is left, and
writes a1-<student-id>.zip containing exactly the required files. Standard
library only.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDENT_ID = re.compile(r"[A-Za-z0-9_]+")
PLACEHOLDER = "TODO"


def check_results(path: Path, spec: dict) -> list[str]:
    """Return a list of problems with results.json (empty = fine)."""
    problems: list[str] = []
    try:
        r = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path.name}: cannot read ({exc})"]
    for source in spec["sources"]:
        if source not in r:
            problems.append(f"results.json: missing source {source!r}")
            continue
        for key in spec["per_source"]:
            if key not in r[source]:
                problems.append(f"results.json[{source!r}]: missing {key!r}")
        for order in spec["orders"]:
            row = r[source].get("orders", {}).get(order)
            if row is None:
                problems.append(f"results.json[{source!r}]['orders']: missing order {order!r}")
                continue
            for kind in spec["per_order"]:
                if kind not in row or "bits_per_byte" not in row[kind] or "perplexity" not in row[kind]:
                    problems.append(f"results.json[{source!r}]['orders'][{order!r}]: {kind!r} needs perplexity and bits_per_byte")
    nplm = r.get(spec["nplm_source"], {}).get("nplm")
    if nplm is None:
        problems.append(f"results.json[{spec['nplm_source']!r}]: missing 'nplm' (run Part 3)")
    else:
        for key in spec["nplm_keys"]:
            if key not in nplm:
                problems.append(f"results.json nplm: missing {key!r}")
    return problems


def check_report(path: Path) -> list[str]:
    """The report must have the AI-use disclosure box and no template placeholders left."""
    problems: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    if "disclosure" not in text.lower():
        problems.append("report.md: no AI-use disclosure box found (the word 'disclosure' does not appear)")
    left = text.count(PLACEHOLDER)
    if left:
        problems.append(f"report.md: {left} {PLACEHOLDER} placeholder(s) left from the template")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 1 or not STUDENT_ID.fullmatch(argv[0].strip()):
        print(__doc__)
        print("The student id may contain letters, digits, and underscores only.")
        return 2
    student = argv[0].strip()
    manifest = tomllib.loads((HERE / "manifest.toml").read_text(encoding="utf-8"))
    required = manifest["submission"]["required"]
    problems = [f"missing required file: {rel}" for rel in required if not (HERE / rel).is_file()]
    if (HERE / "results.json").is_file():
        problems += check_results(HERE / "results.json", manifest["results"])
    if (HERE / "report.md").is_file():
        problems += check_report(HERE / "report.md")
    if problems:
        print("Not packaged. Fix these first:")
        for p in problems:
            print("  -", p)
        return 1
    out = HERE / manifest["submission"]["zip_name"].replace("<student-id>", student)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in required:
            zf.write(HERE / rel, rel)
    print(f"wrote {out.name} ({out.stat().st_size / 1024:.0f} KB) with {len(required)} files:")
    for rel in required:
        print("  ", rel)
    print("Upload this zip on eLearning. Do not rename it.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
