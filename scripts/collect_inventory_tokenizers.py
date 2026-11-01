"""Inspect additional official repositories named by the reviewed inventory.

Uses the existing bounded tokenizer-JSON collector; downloads no weights and
executes no publisher code. Downloads observed today are deliberately omitted:
the note's popularity ranking uses the separate October 31 snapshot only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from build_tokenizer_note import DATA, read_snapshot
from collect_hf_tokenizer_catalog import inspect_tokenizer, request, save_json


def collect(identifier, cache):
    observed = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    response = request(f"https://huggingface.co/api/models/{identifier}")
    record = {"model_id": identifier, "observed_at": observed,
              "metadata_source_url": response.url}
    if response.status_code != 200:
        return {**record, "tokenizer_status": f"metadata_http_{response.status_code}"}
    metadata = response.json()
    # A redirect may change an exact identifier. Review it rather than silently
    # joining a guessed repository name to another checkpoint.
    if metadata.get("id") != identifier:
        return {**record, "tokenizer_status": "repository_rename_needs_review",
                "returned_id": metadata.get("id")}
    save_json(cache / "metadata" / (identifier.replace("/", "--") + ".json"), metadata)
    measured = inspect_tokenizer(metadata, cache)
    return {**record, "tokenizer_status": measured["status"],
            "tokenizer_vocab_size": measured.get("tokenizer_vocab_size"),
            "tokenizer_count_scope": measured.get("count_scope"),
            "tokenizer_source_url": measured.get("source_url"),
            "tokenizer_revision": measured.get("revision") or metadata.get("sha"),
            "tokenizer_sha256": measured.get("sha256"),
            "tokenizer_bytes": measured.get("bytes"),
            "tokenizer_local_path": str(Path(measured["local_path"]).relative_to(DATA.parents[1]))
                if measured.get("local_path") else None,
            "model_source_url": f'https://huggingface.co/{identifier}/blob/{measured.get("revision") or metadata.get("sha")}/README.md',
            "error": measured.get("error")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DATA.parents[1] / "workspace/tokenizer-inventory-review-2026-11-01/additional")
    args = parser.parse_args()
    catalog, _ = read_snapshot(DATA / "snapshots/2026-10-31")
    curated = json.loads((DATA / "models.json").read_text())
    known = {r["model_id"] for r in catalog + curated}
    mappings = json.loads((DATA / "inventory-mappings.json").read_text())["mappings"]
    identifiers = sorted({r["representative_id"] for r in mappings
                          if r["representative_id"] and r["representative_id"] not in known})
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(lambda identifier: collect(identifier, args.cache), identifiers))
    save_json(DATA / "inventory-additional-measurements.json", {
        "scope": "Root tokenizer.json only; exact publisher repositories absent from the October 31 snapshot.",
        "downloads_policy": "No new download counts are mixed into the fixed October 31 ranking.",
        "models": rows})
    for row in rows:
        print(row["model_id"], row["tokenizer_status"], row.get("tokenizer_vocab_size"))


if __name__ == "__main__":
    main()
