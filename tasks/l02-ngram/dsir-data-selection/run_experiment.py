"""Compare data selection on development text, then evaluate a frozen choice.

Use --stage dev --output <dev.json>, then --stage final --selection <dev.json>
--output <final.json>. The final stage rejects a changed submission or corpus.
"""

import argparse
import copy
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

from experiment_utils import load_submission, provenance, sha256, uniforms_for, write_report

ROOT = Path(__file__).resolve().parent


def exact_features(documents):
    """Supplied diagnostic: unhashed unigram/bigram frequencies."""
    counts = Counter()
    for tokens in documents:
        counts.update((token,) for token in tokens)
        counts.update(zip(tokens, tokens[1:]))
    if not counts:
        raise ValueError("feature diagnostics require a nonempty corpus")
    total = sum(counts.values())
    return {key: count / total for key, count in counts.items()}


def js_divergence(left, right):
    """Jensen–Shannon divergence in nats, using the union of exact features."""
    result = 0.0
    for key in sorted(left.keys() | right.keys()):
        p, q = left.get(key, 0.0), right.get(key, 0.0)
        middle = (p + q) / 2
        if p:
            result += 0.5 * p * math.log(p / middle)
        if q:
            result += 0.5 * q * math.log(q / middle)
    return result


def make_case(corpus, m, seed):
    return {
        "raw_fit": [record["tokens"] for record in corpus["raw_fit"]],
        "target_fit": [record["tokens"] for record in corpus["target_fit"]],
        "pool": [{"id": record["id"], "tokens": record["tokens"]} for record in corpus["pool"]],
        "num_buckets": m, "alpha": corpus["experiment"]["alpha"],
        "k": corpus["experiment"]["selection_count"],
        "uniforms": uniforms_for([record["id"] for record in corpus["pool"]], seed),
    }


def compare(solve, corpus, baselines, bucket_sizes, stage):
    reference = exact_features([record["tokens"] for record in corpus[stage]])
    by_id = {record["id"]: record for record in corpus["pool"]}
    selections = [copy.deepcopy(row) for row in baselines if row["num_buckets"] in bucket_sizes]
    for m in bucket_sizes:
        for seed in corpus["experiment"]["seeds"]:
            result = solve(make_case(corpus, m, seed))
            selections.append({
                "method": "dsir", "num_buckets": m, "seed": seed,
                "selected_ids": result["selected_ids"],
                "max_weight": max(result["weights"].values()),
            })
    for row in selections:
        ids = row["selected_ids"]
        budget = corpus["experiment"]["selection_count"]
        if len(ids) != budget or len(set(ids)) != budget or not set(ids) <= by_id.keys():
            raise ValueError(f"each method must select exactly {budget} distinct pool IDs")
        selected = [by_id[key] for key in ids]
        row["selected_tokens"] = sum(len(record["tokens"]) for record in selected)
        row["js_nats"] = js_divergence(exact_features([record["tokens"] for record in selected]), reference)
        row["domain_counts"] = dict(sorted(Counter(record["domain"] for record in selected).items()))
        row["subtopic_counts"] = dict(sorted(Counter(record["subtopic"] for record in selected).items()))
        row["style_counts"] = dict(sorted(Counter(record["style"] for record in selected).items()))
    summaries = []
    for m in bucket_sizes:
        for method in ("uniform", "target-only", "ratio-top-k", "dsir"):
            group = [row for row in selections if row["num_buckets"] == m and row["method"] == method]
            values = [row["js_nats"] for row in group]
            summaries.append({
                "num_buckets": m, "method": method, "runs": len(values),
                "js_mean": statistics.mean(values), "js_min": min(values), "js_max": max(values),
            })
    return selections, summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--stage", choices=("dev", "final"), required=True)
    parser.add_argument("--selection", type=Path, help="development report freezing M")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    start = time.perf_counter()
    corpus_path, baseline_path = ROOT / "data/corpus.json", ROOT / "data/baselines.json"
    corpus = json.loads(corpus_path.read_text())
    config = corpus["experiment"]
    baselines = json.loads(baseline_path.read_text())
    hashes = {"corpus_sha256": sha256(corpus_path), "baselines_sha256": sha256(baseline_path),
              "submission_sha256": sha256(args.submission)}
    if args.stage == "final":
        if args.selection is None:
            parser.error("final evaluation requires --selection <development-report.json>")
        frozen = json.loads(args.selection.read_text())
        if frozen.get("stage") != "dev" or any(frozen.get(key) != value for key, value in hashes.items()):
            parser.error("use the development report for this exact submission and data version")
        chosen = frozen["chosen_num_buckets"]
        if chosen not in config["bucket_sizes"]:
            parser.error("frozen num_buckets must be one of the supplied settings")
        bucket_sizes = [chosen]
    else:
        if args.selection is not None:
            parser.error("--selection is only used for final evaluation")
        bucket_sizes = config["bucket_sizes"]
    module = load_submission(args.submission)
    selections, summaries = compare(module.solve, corpus, baselines, bucket_sizes, args.stage)
    if args.stage == "dev":
        dsir_rows = [row for row in summaries if row["method"] == "dsir"]
        chosen = min(dsir_rows, key=lambda row: (row["js_mean"], row["num_buckets"]))["num_buckets"]
    report = {
        "stage": args.stage, "corpus_version": corpus["version"], **hashes, "chosen_num_buckets": chosen,
        "selection_rule": "lowest mean DSIR development JS; smaller M breaks an exact tie",
        "frozen_selection_sha256": sha256(args.selection) if args.selection else None,
        "provenance": provenance(args.submission, ROOT, {**config, "bucket_sizes": bucket_sizes,
                                                       "stage": args.stage}),
        "selections": selections, "summary": summaries,
        "collision_examples": {str(m): corpus["collision_examples"][str(m)] for m in bucket_sizes},
        "elapsed_seconds": time.perf_counter() - start,
    }
    print("M    method        mean JS    range                 runs")
    for row in summaries:
        print(f"{row['num_buckets']:<4} {row['method']:<13} {row['js_mean']:.6f}  "
              f"[{row['js_min']:.6f}, {row['js_max']:.6f}]  {row['runs']}")
    budget = config["selection_count"]
    print(f"Frozen M: {chosen}. Selection budget: {budget} documents / {budget * config['chunk_tokens']:,} tokens.")
    write_report(args.output, report)


if __name__ == "__main__":
    main()
