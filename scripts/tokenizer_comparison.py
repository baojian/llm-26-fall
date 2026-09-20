"""Compare pinned public tokenizers on the same text; no model weights required.

    uv run --extra tokenizer-comparison python scripts/tokenizer_comparison.py fetch
    uv run --extra tokenizer-comparison python scripts/tokenizer_comparison.py compare
    uv run --extra tokenizer-comparison python scripts/tokenizer_comparison.py measure PATH
    uv run --extra tokenizer-comparison python scripts/tokenizer_comparison.py check PATH

Only ``fetch`` uses the network. Downloads and reports belong in workspace/.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/l01-tokenization/tokenizer-comparison"
DEFAULT_CACHE = ROOT / "workspace/tokenizer-comparison"
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def manifest() -> dict:
    data = read_json(TASK / "tokenizers.json")
    ids = [spec["id"] for spec in data["tokenizers"]]
    if data["schema_version"] != 1 or len(ids) != len(set(ids)):
        raise ValueError("Invalid tokenizer manifest version or duplicate tokenizer IDs")
    for spec in data["tokenizers"]:
        if not re.fullmatch(r"[a-z0-9_-]+", spec["id"]):
            raise ValueError("Tokenizer IDs must be safe filenames")
        if not spec["url"].startswith("https://") or not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]):
            raise ValueError(f"{spec['id']}: expected an HTTPS source and SHA-256")
    return data


def check_runtime(data: dict) -> dict:
    installed = {}
    for package, expected in data["runtime_versions"].items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "not installed"
        if actual != expected:
            raise ValueError(
                f"{package}: expected {expected}, found {actual}. "
                "Run uv sync --locked --extra tokenizer-comparison."
            )
        installed[package] = actual
    return installed


def verified_bytes(spec: dict, cache: Path) -> bytes:
    path = cache / f"{spec['id']}.artifact"
    if not path.is_file():
        raise ValueError(f"Missing {path}; run the fetch command first")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != spec["sha256"]:
        raise ValueError(f"{path}: SHA-256 mismatch; run fetch to replace this cache entry")
    return content


def fetch(spec: dict, cache: Path) -> None:
    try:
        verified_bytes(spec, cache)
        return
    except ValueError:
        pass
    with urllib.request.urlopen(spec["url"], timeout=60) as response:
        content = response.read(MAX_ARTIFACT_BYTES + 1)
    if len(content) > MAX_ARTIFACT_BYTES:
        raise ValueError(f"{spec['id']}: artifact exceeds the tokenizer download limit")
    if hashlib.sha256(content).hexdigest() != spec["sha256"]:
        raise ValueError(f"{spec['id']}: downloaded SHA-256 mismatch; cache was not replaced")
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        path = Path(temporary) / "artifact"
        path.write_bytes(content)
        path.replace(cache / f"{spec['id']}.artifact")


def component_summary(value):
    """Describe normalization rules without copying large compiled maps into reports."""
    if isinstance(value, dict):
        return {key: ({"sha256": hashlib.sha256(base64.b64decode(item)).hexdigest(),
                       "bytes": len(base64.b64decode(item))} if key == "precompiled_charsmap"
                      else component_summary(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [component_summary(item) for item in value]
    return value


class Engine:
    """Use the published tokenizer pipeline, with no chat template or wrappers."""

    def __init__(self, spec: dict, cache: Path):
        self.spec = spec
        content = verified_bytes(spec, cache)
        if spec["backend"] == "tiktoken":
            import regex
            import tiktoken

            ranks = {base64.b64decode(piece): int(rank) for piece, rank in
                     (line.split() for line in content.splitlines() if line)}
            self.tokenizer = tiktoken.Encoding(
                name=spec["id"], pat_str=spec["pattern"], mergeable_ranks=ranks,
                special_tokens=spec["special_tokens"],
            )
            self.pattern = regex.compile(spec["pattern"])
            all_ids = list(ranks.values()) + list(spec["special_tokens"].values())
            self.metadata = {
                "algorithm": "BPE", "normalizer": None,
                "pre_tokenizer": {"type": "Regex", "pattern": spec["pattern"]},
                "decoder": {"type": "UTF-8 bytes"},
                "base_vocabulary_entries": len(ranks),
                "vocabulary_entries": len(ranks) + len(spec["special_tokens"]),
                "special_token_entries": len(spec["special_tokens"]),
                "highest_token_id_plus_one": max(all_ids) + 1,
            }
        elif spec["backend"] == "huggingface":
            from tokenizers import Tokenizer

            self.tokenizer = Tokenizer.from_str(content.decode("utf-8"))
            self.tokenizer.no_padding()
            self.tokenizer.no_truncation()
            # True means special-looking input is processed as ordinary text.
            # Other added tokens still use the artifact's matching rules.
            self.tokenizer.encode_special_tokens = True
            config = json.loads(self.tokenizer.to_str())
            vocab = self.tokenizer.get_vocab(with_added_tokens=True)
            self.metadata = {
                "algorithm": config["model"]["type"], "normalizer": component_summary(config["normalizer"]),
                "pre_tokenizer": config["pre_tokenizer"], "decoder": config["decoder"],
                "base_vocabulary_entries": self.tokenizer.get_vocab_size(with_added_tokens=False),
                "vocabulary_entries": len(vocab),
                "special_token_entries": sum(token["special"] for token in config["added_tokens"]),
                "highest_token_id_plus_one": max(vocab.values()) + 1,
            }
        else:
            raise ValueError(f"Unknown backend: {spec['backend']}")

    def observe(self, text: str) -> dict:
        if self.spec["backend"] == "tiktoken":
            ids = self.tokenizer.encode_ordinary(text)
            normalized = text
            chunks = self.pattern.findall(text)
            pieces = [repr(self.tokenizer.decode_single_token_bytes(token)) for token in ids]
            piece_format = "Python bytes repr (a token may contain incomplete UTF-8)"
            decoded = self.tokenizer.decode(ids)
        else:
            normalizer = self.tokenizer.normalizer
            normalized = normalizer.normalize_str(text) if normalizer is not None else text
            pre_tokenizer = self.tokenizer.pre_tokenizer
            chunks = [piece for piece, _ in pre_tokenizer.pre_tokenize_str(normalized)] if pre_tokenizer else [normalized]
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            ids, pieces = encoded.ids, encoded.tokens
            piece_format = "Native tokenizer notation (not literal decoded bytes)"
            decoded = self.tokenizer.decode(ids, skip_special_tokens=False)
        return {
            "input_codepoints": len(text), "input_utf8_bytes": len(text.encode("utf-8")),
            "normalized_text": normalized,
            "pre_tokenizer_preview": chunks,
            "token_ids": ids, "token_pieces": pieces, "piece_format": piece_format,
            "token_count": len(ids),
            "input_bytes_per_token": len(text.encode("utf-8")) / len(ids) if ids else None,
            "decoded_text": decoded, "round_trip_exact": decoded == text,
        }


def select_specs(data: dict, names: list[str] | None) -> list[dict]:
    available = {spec["id"]: spec for spec in data["tokenizers"]}
    if names is None:
        return list(available.values())
    if not names or len(names) != len(set(names)) or any(name not in available for name in names):
        raise ValueError(f"Choose distinct tokenizer IDs from: {', '.join(available)}")
    return [available[name] for name in names]


def provenance(data: dict, specs: list[dict]) -> dict:
    return {
        "runtime_versions": check_runtime(data),
        "manifest_sha256": hashlib.sha256((TASK / "tokenizers.json").read_bytes()).hexdigest(),
        "artifacts": {spec["id"]: {"revision": spec["revision"], "sha256": spec["sha256"]} for spec in specs},
        "policy": "ordinary text; no padding, truncation, chat template, or automatic special-token wrappers",
    }


def validate_contribution(record: dict, data: dict) -> list[dict]:
    """Check structure; explanations and predictions still need human review."""
    if record.get("schema_version") != 1:
        raise ValueError("Contribution schema_version must be 1")
    contributors = record.get("contributors", [])
    if not isinstance(contributors, list) or not contributors or len(contributors) != len(set(contributors)):
        raise ValueError("List one or more distinct contributors")
    if any(not isinstance(user, str) or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,37}[a-z0-9])?", user) for user in contributors):
        raise ValueError("Use lowercase GitHub usernames in contributors")
    names = record.get("tokenizers", [])
    if not isinstance(names, list) or len(names) != 2:
        raise ValueError("Choose exactly two tokenizers for one contribution")
    specs = select_specs(data, names)
    cases = record.get("cases", [])
    shared = {case["id"]: case["text"] for case in read_json(TASK / "inputs.json") if case["required"]}
    if not isinstance(cases, list) or len(cases) < len(shared) + 1:
        raise ValueError("Include all three required shared cases and at least one original case")
    ids, texts = set(), set()
    for case in cases:
        case_id, text = case.get("id"), case.get("text")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in ids:
            raise ValueError("Case IDs must be nonempty and distinct")
        if not isinstance(text, str) or text in texts or len(text) > 2000:
            raise ValueError("Case texts must be distinct strings of at most 2000 code points")
        # Reject unpaired surrogates so byte counts are well defined.
        text.encode("utf-8")
        if not isinstance(case.get("prediction"), str) or not case["prediction"].strip():
            raise ValueError(f"{case_id}: write a prediction before measuring")
        ids.add(case_id)
        texts.add(text)
    given = {case["id"]: case["text"] for case in cases}
    if any(given.get(case_id) != text for case_id, text in shared.items()):
        raise ValueError("Keep the required shared case IDs and text unchanged")
    supplied_texts = {case["text"] for case in read_json(TASK / "inputs.json")}
    if not texts - supplied_texts:
        raise ValueError("Add at least one original input beyond the supplied cases")
    for field in ("question", "notes", "sources_and_notes"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"Write {field}; the checker cannot judge its reasoning")
    return specs


def check_contribution(record: dict, data: dict, cache: Path) -> None:
    specs = validate_contribution(record, data)
    if record.get("provenance") != provenance(data, specs):
        raise ValueError("Provenance differs from the pinned setup; run measure again")
    engines = {spec["id"]: Engine(spec, cache) for spec in specs}
    for case in record["cases"]:
        expected = {name: engine.observe(case["text"]) for name, engine in engines.items()}
        if case.get("observed") != expected:
            raise ValueError(f"{case['id']}: observed data differ from an independent re-encoding; run measure again")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("fetch", help="Download and verify only public tokenizer artifacts")
    download.add_argument("--tokenizers", nargs="+")
    compare = commands.add_parser("compare", help="Print a JSON report for shared inputs or --text")
    compare.add_argument("--tokenizers", nargs="+")
    compare.add_argument("--text", action="append")
    for name in ("measure", "check"):
        command = commands.add_parser(name, help="Fill observations without changing predictions" if name == "measure" else "Recompute and verify a contribution offline")
        command.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        data = manifest()
        if args.command == "fetch":
            for spec in select_specs(data, args.tokenizers):
                fetch(spec, args.cache)
                print(f"{spec['id']}: verified {spec['sha256']}")
        elif args.command == "compare":
            specs = select_specs(data, args.tokenizers)
            report = {"provenance": provenance(data, specs)}
            engines = {spec["id"]: Engine(spec, args.cache) for spec in specs}
            report["tokenizers"] = {name: engine.metadata for name, engine in engines.items()}
            cases = read_json(TASK / "inputs.json") if args.text is None else [
                {"id": f"custom-{i + 1}", "text": text} for i, text in enumerate(args.text)
            ]
            report["cases"] = [dict(id=case["id"], text=case["text"], observed={
                name: engine.observe(case["text"]) for name, engine in engines.items()
            }) for case in cases]
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            record = read_json(args.path)
            specs = validate_contribution(record, data)
            if args.command == "measure":
                record["provenance"] = provenance(data, specs)
                engines = {spec["id"]: Engine(spec, args.cache) for spec in specs}
                for case in record["cases"]:
                    case["observed"] = {name: engine.observe(case["text"]) for name, engine in engines.items()}
                write_json(args.path, record)
                print(f"Measured {args.path}; predictions and notes preserved. Revise your explanation, then run check.")
            else:
                check_contribution(record, data, args.cache)
                print(f"{args.path}: all observations match the pinned tokenizers; explanations need peer review")
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
