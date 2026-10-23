"""Compare pinned public tokenizers on the same text; no model weights required.

    uv run --extra tokenizer-comparison python explorations/tokenizer-comparison/run.py fetch
    uv run --extra tokenizer-comparison python explorations/tokenizer-comparison/run.py compare
    uv run --extra tokenizer-comparison python explorations/tokenizer-comparison/run.py build
    uv run --extra tokenizer-comparison python explorations/tokenizer-comparison/run.py build --check

Only ``fetch`` uses the network. Downloads and reports belong in workspace/.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import re
import subprocess
import unicodedata
import tempfile
import urllib.request
from pathlib import Path

TOPIC = Path(__file__).resolve().parent
ROOT = TOPIC.parents[1]
DEFAULT_CACHE = ROOT / "workspace/tokenizer-comparison"
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def manifest() -> dict:
    data = read_json(TOPIC / "tokenizers.json")
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
        self.unknown_id = None
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
            model = config["model"]
            self.unknown_id = model.get("unk_id")
            if model.get("unk_token") is not None:
                self.unknown_id = vocab[model["unk_token"]]
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
        unknown_count = ids.count(self.unknown_id) if self.unknown_id is not None else 0
        if decoded == text:
            fidelity = "exact"
        elif unknown_count:
            fidelity = "unknown-tokens"
        elif unicodedata.normalize("NFC", decoded) == unicodedata.normalize("NFC", text):
            fidelity = "nfc-equivalent"
        else:
            fidelity = "text-changed"
        return {
            "input_codepoints": len(text), "input_utf8_bytes": len(text.encode("utf-8")),
            "normalized_text": normalized,
            "pre_tokenizer_preview": chunks,
            "token_ids": ids, "token_pieces": pieces, "piece_format": piece_format,
            "token_count": len(ids),
            "input_bytes_per_token": len(text.encode("utf-8")) / len(ids) if ids else None,
            "decoded_text": decoded, "round_trip_exact": decoded == text,
            "unknown_token_count": unknown_count, "fidelity": fidelity,
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
        "unicode_database_version": unicodedata.unidata_version,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((TOPIC / "tokenizers.json").read_bytes()).hexdigest(),
        "artifacts": {spec["id"]: {"revision": spec["revision"], "sha256": spec["sha256"]} for spec in specs},
        "policy": "ordinary text; no padding, truncation, chat template, or automatic special-token wrappers",
    }


def load_cases() -> list[dict]:
    cases = read_json(TOPIC / "inputs.json")
    seen = set()
    for case in cases:
        if not re.fullmatch(r"[a-z0-9-]+", case["id"]) or case["id"] in seen:
            raise ValueError("Case IDs must be distinct lowercase slugs")
        if not isinstance(case["text"], str) or len(case["text"]) > 2000:
            raise ValueError("Inputs must be strings of at most 2000 code points")
        case["text"].encode("utf-8")
        if not case.get("group") or not case.get("question"):
            raise ValueError("Each case needs a group and a question")
        seen.add(case["id"])
    return cases


def compare(data: dict, specs: list[dict], cases: list[dict], cache: Path) -> dict:
    engines = {spec["id"]: Engine(spec, cache) for spec in specs}
    report = {
        "schema_version": 1,
        "provenance": provenance(data, specs),
        "tokenizers": {name: dict(engine.metadata, label=engine.spec["label"],
                                  source=engine.spec["url"]) for name, engine in engines.items()},
        "cases": [dict(case, observed={name: engine.observe(case["text"])
                                      for name, engine in engines.items()}) for case in cases],
    }
    encoded_cases = json.dumps(cases, ensure_ascii=False, sort_keys=True).encode("utf-8")
    report["provenance"]["inputs_sha256"] = hashlib.sha256(encoded_cases).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("fetch", help="Download and verify public tokenizer files")
    download.add_argument("--tokenizers", nargs="+")
    inspect = commands.add_parser("compare", help="Measure supplied examples or custom text offline")
    inspect.add_argument("--tokenizers", nargs="+")
    inspect.add_argument("--text", action="append")
    build = commands.add_parser("build", help="Rebuild the teaching tables and explorable page offline")
    build.add_argument("--check", action="store_true", help="Fail on stale published files without changing them")
    args = parser.parse_args()
    try:
        data = manifest()
        specs = select_specs(data, getattr(args, "tokenizers", None))
        if args.command == "fetch":
            for spec in specs:
                fetch(spec, args.cache)
                print(f"{spec['id']}: verified {spec['sha256']}")
            return
        cases = load_cases()
        if args.command == "compare" and args.text is not None:
            cases = [{"id": f"custom-{i + 1}", "text": value, "group": "Custom",
                      "question": "What changes in this input?"} for i, value in enumerate(args.text)]
        report = compare(data, specs, cases, args.cache)
        encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.command == "compare":
            print(encoded, end="")
            return
        command = ["node", str(TOPIC / "build.mjs")]
        if args.check:
            command.append("--check")
        subprocess.run(command, input=encoded, text=True, check=True)
        if not args.check:
            DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
            write_json(DEFAULT_CACHE / "report.json", report)
            print(f"Full measurement record: {DEFAULT_CACHE / 'report.json'}")
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
