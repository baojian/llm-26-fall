"""Compare four/eight LSH bands and inspect a constructed perplexity audit."""

import argparse
import copy
import json
import time
from pathlib import Path

from experiment_utils import load_submission, provenance, write_report

ROOT = Path(__file__).resolve().parent


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def compare(solve, corpus):
    truth = {tuple(pair) for pair in corpus["analysis"]["exact_jaccard_pairs"]}
    containment = {tuple(pair) for pair in corpus["analysis"]["containment_pairs"]}
    rows = []
    for bands in (4, 8):
        case = {**corpus["parameters"], "num_bands": bands, "documents": copy.deepcopy(corpus["documents"])}
        result = solve(case)
        candidates = {tuple(pair) for pair in result["candidate_pairs"]}
        verified = {tuple(pair) for pair in result["verified_pairs"]}
        flags = {tuple(pair) for pair in result["contamination_pairs"]}
        if verified != candidates & truth:
            raise ValueError("verification disagrees with exact Jaccard for this fixed corpus; run the checks")
        if flags != containment:
            raise ValueError("containment disagrees with the exhaustive audit; run the checks")
        hits = candidates & truth
        rows.append({
            "num_bands": bands, "candidate_count": len(candidates),
            "exact_pair_count": len(truth), "candidate_recall": ratio(len(hits), len(truth)),
            "candidate_precision": ratio(len(hits), len(candidates)),
            "missed_jaccard_pairs": [list(pair) for pair in sorted(truth - candidates)],
            "false_candidates": [list(pair) for pair in sorted(candidates - truth)],
            "containment_outside_candidates": [list(pair) for pair in sorted(flags)
                                                if tuple(sorted(pair)) not in candidates],
            **result,
        })
    return rows


def loss_audit(solve, audit):
    result = solve({"documents": copy.deepcopy(audit["documents"]), "n": 2, "num_hashes": 4,
                    "num_bands": 2, "jaccard_threshold": 0.8, "containment_threshold": 1.0})
    flagged = {pair[1] for pair in result["contamination_pairs"]}
    scores = audit["scores"]
    output = {"description": audit["description"], "flagged_eval_ids": sorted(flagged), "runs": {}}
    for run in ("a", "b"):
        strata = {}
        for name, subset in (("full", scores), ("clean", [row for row in scores if row["id"] not in flagged])):
            loss = sum(row[f"run_{run}_loss_bits"] for row in subset)
            tokens = sum(row["scored_tokens"] for row in subset)
            strata[name] = {"eval_ids": [row["id"] for row in subset], "loss_bits": loss,
                            "scored_tokens": tokens, "perplexity": 2 ** (loss / tokens) if tokens else None}
        output["runs"][run] = strata
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    start = time.perf_counter()
    corpus = json.loads((ROOT / "data/corpus.json").read_text())
    audit = json.loads((ROOT / "data/evaluation-audit.json").read_text())
    module = load_submission(args.submission)
    rows = compare(module.solve, corpus)
    report = {"corpus_version": corpus["version"],
              "provenance": provenance(args.submission, ROOT, {**corpus["parameters"], "bands": [4, 8]}),
              "comparisons": rows, "loss_audit": loss_audit(module.solve, audit),
              "elapsed_seconds": time.perf_counter() - start}
    print("Bands  candidates  exact pairs  recall  precision")
    for row in rows:
        recall = f"{row['candidate_recall']:.3f}" if row["candidate_recall"] is not None else "N/A"
        precision = f"{row['candidate_precision']:.3f}" if row["candidate_precision"] is not None else "N/A"
        print(f"{row['num_bands']:>5}  {row['candidate_count']:>10}  {row['exact_pair_count']:>11}  {recall:>6}  {precision:>9}")
    print("Constructed scores: perplexity on full / shared clean evaluation subsets")
    for name, row in report["loss_audit"]["runs"].items():
        print(f"Run {name}: {row['full']['perplexity']} / {row['clean']['perplexity']}")
    write_report(args.output, report)


if __name__ == "__main__":
    main()
