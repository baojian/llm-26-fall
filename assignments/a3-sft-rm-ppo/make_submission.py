"""Package required files and optional evidence, with explicit missing evidence."""
import argparse
import importlib.util
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Load the sibling explicitly so importing this script from a grading tool also
# validates against this package, rather than another directory on sys.path.
_spec = importlib.util.spec_from_file_location('a3_results_validator', ROOT / 'validate_results.py')
validator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validator)
RECORDS = validator.RECORDS
EVIDENCE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.svg', '.pdf', '.csv', '.tsv',
                     '.json', '.txt', '.md', '.py', '.ipynb'}
EVIDENCE_LIMIT = 20 * 1024 * 1024


def evidence_files():
    evidence = ROOT / 'evidence'
    if evidence.is_symlink():
        raise ValueError('evidence must be an ordinary directory')
    if not evidence.exists():
        return []
    if not evidence.is_dir():
        raise ValueError('evidence must be a directory')
    files = []
    for source in sorted(evidence.rglob('*')):
        if source.is_symlink():
            raise ValueError(f'symlinks are not packaged: {source.name}')
        if source.is_file():
            if source.suffix.lower() not in EVIDENCE_SUFFIXES:
                raise ValueError(f'unsupported evidence file: {source.name}; move it outside evidence/')
            files.append(source)
    if sum(p.stat().st_size for p in files) > EVIDENCE_LIMIT:
        raise ValueError('optional evidence exceeds 20 MiB uncompressed')
    return files


def package(student_id, run, report, partial_reason=None):
    if not re.fullmatch(r'[A-Za-z0-9_]+', student_id):
        raise ValueError('student ID must contain only letters, digits, or underscores')
    if not report.is_file() or report.is_symlink():
        raise ValueError('complete an ordinary report.md file before packaging')
    if not (ROOT / 'a3.py').is_file() or (ROOT / 'a3.py').is_symlink():
        raise ValueError('a3.py must be an ordinary file')
    extras = evidence_files()
    results = validator.collect_results(run, ROOT, partial_reason)
    serialized = json.dumps(results, indent=2, allow_nan=False)+'\n'
    (ROOT / 'results.json').write_text(serialized)
    dest = ROOT / f'a3-{student_id}.zip'
    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(ROOT / 'a3.py', 'a3.py')
        archive.write(report, 'report.md')
        archive.writestr('results.json', serialized)
        for path in extras:
            archive.write(path, path.relative_to(ROOT).as_posix())
    return dest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('student_id')
    parser.add_argument('--run', type=Path, default=ROOT/'outputs/main')
    parser.add_argument('--report', type=Path, default=ROOT/'report.md')
    parser.add_argument('--partial-reason')
    args = parser.parse_args()
    try:
        print(package(args.student_id, args.run, args.report, args.partial_reason))
    except (ValueError, OSError) as exc:
        parser.exit(1, f'error: {exc}\n')
