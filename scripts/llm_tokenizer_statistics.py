"""Validate the curated inventory and reproduce counts from pinned tokenizer files.

No model weights, inference, remote Python execution, or paid APIs are used.
SentencePiece files require sentencepiece==0.2.1 and protobuf==6.33.6.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "model-tokenizer-statistics"
COUNT_FIELDS = (
    "tokenizer_vocab_size", "tokenizer_base_vocab_size",
    "tokenizer_added_vocab_size", "tokenizer_special_token_count",
    "tokenizer_id_capacity",
)


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def vocabulary_statistics(base: dict[int, str | bytes], additions: list[dict],
                          special_ids: set[int]) -> dict:
    """Count entries once, including specials already present in the base model."""
    vocab = dict(base)
    for token in additions:
        index, content = int(token["id"]), token["content"]
        if index in vocab and vocab[index] != content:
            raise ValueError(f"Conflicting token definitions at ID {index}")
        vocab[index] = content
        if token.get("special"):
            special_ids.add(index)
    if not vocab or min(vocab) < 0:
        raise ValueError("Empty vocabulary or negative token ID")
    if not special_ids <= vocab.keys():
        raise ValueError("Special-token IDs must exist in the vocabulary")
    return {
        "tokenizer_vocab_size": len(vocab),
        "tokenizer_base_vocab_size": len(base),
        "tokenizer_added_vocab_size": len(vocab.keys() - base.keys()),
        "tokenizer_special_token_count": len(special_ids),
        "tokenizer_id_capacity": max(vocab) + 1,
    }


def token_contents(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict) and "content" in value:
        yield value["content"]
    elif isinstance(value, list):
        for item in value:
            yield from token_contents(item)


def measure(folder: Path, profile: str | None = None) -> dict:
    """Inspect serialized vocabularies and configuration without importing repo code."""
    config = read_json(folder / "tokenizer_config.json", {})
    additions, special_ids = [], set()
    primary = folder / "tokenizer.json"
    if primary.exists():
        document = read_json(primary)
        model = document["model"]
        serialized = model["vocab"]
        base = ({int(index): token for token, index in serialized.items()}
                if isinstance(serialized, dict)
                else {index: item[0] for index, item in enumerate(serialized)})
        additions.extend(document.get("added_tokens", []))
        algorithm = model.get("type")
        if algorithm is None:
            if "merges" in model:
                algorithm = "BPE"
            elif "continuing_subword_prefix" in model:
                algorithm = "WordPiece"
            elif isinstance(serialized, list) and "unk_id" in model:
                algorithm = "Unigram"
            else:
                raise ValueError("Unrecognized legacy tokenizer model")
        backend = "huggingface_tokenizer_json"
    elif (folder / "encoding.json").exists():
        document = read_json(folder / "encoding.json")
        primary = folder / document["rank_file"]
        base = read_ranks(primary)
        additions.extend({"id": index, "content": token, "special": True}
                         for token, index in document["special_tokens"].items())
        algorithm, backend = "BPE", "tiktoken"
    elif (folder / "qwen.tiktoken").exists() or (folder / "tiktoken.model").exists():
        primary = folder / ("qwen.tiktoken" if profile == "qwen1" else "tiktoken.model")
        base = read_ranks(primary)
        if profile == "qwen1":
            # Pinned tokenization_qwen.py: three markers plus 205 extras.
            if len(base) != 151643:
                raise ValueError("Unexpected Qwen 1 base vocabulary")
            specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
            specials.extend(f"<|extra_{i}|>" for i in range(205))
            additions.extend({"id": 151643 + i, "content": text, "special": True}
                             for i, text in enumerate(specials))
        elif profile in ("kimi_k2", "kimi_k25"):
            # The pinned K2 implementation adds 258; K2.5 changes this to 256.
            reserved_count = 258 if profile == "kimi_k2" else 256
            decoder = config.get("added_tokens_decoder", {})
            for index in range(len(base), len(base) + reserved_count):
                text = decoder.get(str(index), {}).get("content", f"<|reserved_token_{index}|>")
                additions.append({"id": index, "content": text, "special": True})
        else:
            raise ValueError("Rank-file special tokens need a reviewed profile")
        algorithm, backend = "BPE", "tiktoken"
    elif (folder / "vocab.json").exists():
        primary = folder / "vocab.json"
        base = {int(index): token for token, index in read_json(primary).items()}
        algorithm, backend = "BPE", "vocab_json_and_merges"
    else:
        import sentencepiece as spm
        from sentencepiece import sentencepiece_model_pb2

        primary = next((folder / name for name in ("spiece.model", "tokenizer.model")
                        if (folder / name).exists()), None)
        if primary is None:
            raise ValueError(f"No supported tokenizer artifact: {folder}")
        processor = spm.SentencePieceProcessor(model_file=str(primary))
        base = {i: processor.id_to_piece(i) for i in range(processor.get_piece_size())}
        special_ids.update(i for i in base if processor.is_control(i) or processor.is_unknown(i))
        proto = sentencepiece_model_pb2.ModelProto()
        proto.ParseFromString(primary.read_bytes())
        algorithm = {1: "Unigram", 2: "BPE", 3: "Word", 4: "Char"}[proto.trainer_spec.model_type]
        backend = "sentencepiece"
        for offset in range(config.get("extra_ids", 0)):
            additions.append({"id": len(base) + offset,
                              "content": f"<extra_id_{config['extra_ids'] - offset - 1}>",
                              "special": True})

    for index, item in config.get("added_tokens_decoder", {}).items():
        additions.append({"id": int(index), **item})
    for token, index in read_json(folder / "added_tokens.json", {}).items():
        additions.append({"id": int(index), "content": token})
    combined = dict(base)
    combined.update((int(item["id"]), item["content"]) for item in additions)
    lookup = {token: index for index, token in combined.items()}
    markers = {**config, **read_json(folder / "special_tokens_map.json", {})}
    for key, value in markers.items():
        if key.endswith("_token") or key in ("additional_special_tokens", "extra_special_tokens"):
            for text in token_contents(value):
                if text in lookup:
                    special_ids.add(lookup[text])
    return {**vocabulary_statistics(base, additions, special_ids),
            "tokenizer_algorithm": algorithm, "tokenizer_format": backend,
            "tokenizer_primary_file": primary.name}


def read_ranks(path: Path) -> dict[int, bytes]:
    result = {}
    for line in path.read_bytes().splitlines():
        if not line.strip():
            continue
        token, rank = line.split()
        index = int(rank)
        if index in result:
            raise ValueError(f"Duplicate rank {index}")
        result[index] = base64.b64decode(token, validate=True)
    return result


def csv_text(rows: list[dict]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def validate(rows: list[dict]):
    if not rows:
        raise ValueError("No model records")
    seen = set()
    for row in rows:
        if set(row) != set(rows[0]) or row["model_id"] in seen:
            raise ValueError("Inconsistent columns or duplicate model ID")
        seen.add(row["model_id"])
        date = row["release_date"]
        if date is not None:
            if not re.fullmatch(r"\d{4}(?:-\d{2})?(?:-\d{2})?", date):
                raise ValueError(f"Invalid date: {date}")
            datetime.date.fromisoformat(date + {4: "-01-01", 7: "-01", 10: ""}[len(date)])
            if row["release_date_precision"] != {4: "year", 7: "month", 10: "day"}[len(date)]:
                raise ValueError("Release-date precision disagrees with the date")
            if not row["release_date_source_url"]:
                raise ValueError("Release date without a source")
        elif row["release_date_precision"] is not None:
            raise ValueError("Date precision without a date")
        if row["weights_access"] not in ("open_weights", "closed_weights"):
            raise ValueError("Unknown weights access category")
        if row["tokenizer_status"] not in ("downloaded", "gated", "not_verified", "unavailable_at_collection"):
            raise ValueError("Unknown tokenizer status")
        for key in COUNT_FIELDS + ("pretraining_dataset_tokens", "pretraining_tokens_processed"):
            value = row[key]
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"Invalid count: {row['model_id']} {key}")
        size = row["tokenizer_vocab_size"]
        if size is not None:
            if not row["tokenizer_local_path"] or row["tokenizer_status"] != "downloaded":
                raise ValueError("Measured vocabulary without a downloaded artifact")
            for key in ("tokenizer_revision", "tokenizer_source_url", "tokenizer_sha256", "tokenizer_count_scope"):
                if not row[key]:
                    raise ValueError(f"Measured vocabulary without {key}")
            if size != row["tokenizer_base_vocab_size"] + row["tokenizer_added_vocab_size"]:
                raise ValueError("Vocabulary components disagree")
            if not 0 <= row["tokenizer_special_token_count"] <= size <= row["tokenizer_id_capacity"]:
                raise ValueError("Invalid special-token count or ID capacity")
        elif any(row[key] is not None for key in COUNT_FIELDS):
            raise ValueError("Partial measured counts")
        elif row["tokenizer_status"] == "downloaded" or row["tokenizer_local_path"] is not None:
            raise ValueError("Downloaded artifact without measured counts")
        if any(row[key] is not None for key in ("pretraining_dataset_tokens", "pretraining_tokens_processed")):
            if not row["pretraining_source_url"] or not row["pretraining_count_notes"]:
                raise ValueError("Training count without source and scope")


def validate_manifest(rows: list[dict], manifest: list[dict]):
    paths = {item["local_path"]: item for item in manifest}
    if len(paths) != len(manifest):
        raise ValueError("Duplicate artifact path")
    for item in manifest:
        relative = Path(item["local_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("workspace", "llm-tokenizers"):
            raise ValueError("Artifact path must remain inside the tokenizer cache")
        if not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]):
            raise ValueError("Invalid artifact checksum")
        if type(item["bytes"]) is not int or item["bytes"] <= 0:
            raise ValueError("Invalid artifact size")
        if bool(item.get("url")) == bool(item.get("source_path")):
            raise ValueError("Artifact needs exactly one download URL or tracked source")
        if item.get("url") and not item["url"].startswith("https://"):
            raise ValueError("Artifact download must use HTTPS")
        if item.get("source_path"):
            source = Path(item["source_path"])
            if source.is_absolute() or ".." in source.parts:
                raise ValueError("Invalid tracked artifact definition path")
    for row in rows:
        if row["tokenizer_status"] == "downloaded":
            primary = str(Path(row["tokenizer_local_path"]) / row["tokenizer_primary_file"])
            if primary not in paths or paths[primary]["sha256"] != row["tokenizer_sha256"]:
                raise ValueError(f"Missing or inconsistent primary artifact: {row['model_id']}")


def verify_files(manifest: list[dict], root: Path, fetch: bool = False):
    validate_manifest([], manifest)
    for item in manifest:
        relative = Path(item["local_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("workspace", "llm-tokenizers"):
            raise ValueError("Artifact path must remain inside the tokenizer cache")
        path = root / relative
        if fetch and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            if item.get("source_path"):
                source = Path(item["source_path"])
                if source.is_absolute() or ".." in source.parts:
                    raise ValueError("Invalid tracked artifact definition path")
                payload = (ROOT / source).read_bytes()
            else:
                with urllib.request.urlopen(item["url"], timeout=120) as response:
                    payload = response.read()
            if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError(f"Downloaded checksum mismatch: {path}")
            path.write_bytes(payload)
        if (not path.is_file() or path.stat().st_size != item["bytes"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]):
            raise ValueError(f"Missing artifact or checksum mismatch: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "export", "fetch", "measure"))
    parser.add_argument("--cache-root", type=Path, default=ROOT,
                        help="Repository root relative to which local paths resolve")
    args = parser.parse_args()
    rows = read_json(DATA / "models.json")
    validate(rows)
    manifest = read_json(DATA / "tokenizer-files.json")
    validate_manifest(rows, manifest)
    if args.command in ("fetch", "measure"):
        verify_files(manifest, args.cache_root, fetch=args.command == "fetch")
    if args.command == "measure":
        for row in rows:
            if row["tokenizer_status"] != "downloaded":
                continue
            measured = measure(args.cache_root / row["tokenizer_local_path"], row["tokenizer_profile"])
            for key, value in measured.items():
                if row[key] != value:
                    raise ValueError(f"Measurement mismatch: {row['model_id']} {key}: {row[key]} != {value}")
    expected = csv_text(rows)
    if args.command == "export":
        (DATA / "models.csv").write_text(expected, encoding="utf-8", newline="")
    elif args.command == "check" and (DATA / "models.csv").read_text() != expected:
        raise ValueError("CSV differs from canonical JSON; run export")
    print(f"{args.command}: {len(rows)} model records validated")


if __name__ == "__main__":
    main()
