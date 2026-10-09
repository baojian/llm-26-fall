"""Supplied standard-library validator for the public results schema and controls.

Structural validation does not establish that reported experiments were performed.
"""
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_BYTES = 2 * 1024 * 1024


def validate(value):
    problems = []
    root_schema = json.loads((HERE / "results-schema.json").read_text())
    schema = {**root_schema["$defs"]["complete"], "$defs": root_schema["$defs"]}

    def visit(item, rule, path):
        if "$ref" in rule:
            rule = schema["$defs"][rule["$ref"].split("/")[-1]]
        kind = rule.get("type")
        correct = {"object": lambda: isinstance(item, dict), "array": lambda: isinstance(item, list),
                   "number": lambda: type(item) in (int, float) and math.isfinite(item),
                   "integer": lambda: type(item) is int, "string": lambda: isinstance(item, str),
                   "boolean": lambda: type(item) is bool}
        if kind in correct and not correct[kind]():
            problems.append(f"{path}: expected finite {kind}")
            return
        if "enum" in rule and item not in rule["enum"]:
            problems.append(f"{path}: expected one of {rule['enum']}")
        if isinstance(item, str) and "pattern" in rule and not re.fullmatch(rule["pattern"], item):
            problems.append(f"{path}: invalid string format")
        if type(item) in (int, float):
            if not math.isfinite(item) or item < rule.get("minimum", -math.inf) or item > rule.get("maximum", math.inf):
                problems.append(f"{path}: invalid numeric value")
        if isinstance(item, dict):
            for key in rule.get("required", []):
                if key not in item:
                    problems.append(f"{path}: missing {key}")
            for key, child in item.items():
                if rule.get("additionalProperties") is False and key not in rule.get("properties", {}):
                    problems.append(f"{path}: unexpected key {key}")
                visit(child, rule.get("properties", {}).get(key, {}), f"{path}.{key}")
        if isinstance(item, list):
            if not rule.get("minItems", 0) <= len(item) <= rule.get("maxItems", 10000):
                problems.append(f"{path}: invalid array length")
            for index, child in enumerate(item):
                visit(child, rule.get("items", {}), f"{path}[{index}]")
    partial = isinstance(value, dict) and value.get("submission_status") == "partial"
    if partial:
        # Partial credit must not require invented measurements for a broken part.
        explanations = value.get("unavailable_evidence", {})
        if not isinstance(explanations, dict):
            return ["partial results need an unavailable_evidence object"]
        for key in schema["required"]:
            if key not in value:
                problems.append(f"results: missing {key}; use null and explain unavailable evidence")
            elif value[key] is None:
                reason = explanations.get(key)
                if not isinstance(reason, str) or len(reason.strip()) < 10 or "TODO" in reason:
                    problems.append(f"{key}: explain why this evidence is unavailable")
            else:
                visit(value[key], schema["properties"][key], f"results.{key}")
        # Check extra fields for NaN/Infinity as well.
        for key, item in value.items():
            if key not in schema["required"]:
                visit(item, {}, f"results.{key}")
        if value.get("schema_version") != 1:
            problems.append("schema_version must be 1")
    else:
        visit(value, schema, "results")
    if problems:
        return problems
    runs = value["runs"]
    if runs is None:
        return problems
    baseline = runs["baseline"]
    for name, run in runs.items():
        config = run["config"]
        if run["condition"] != name:
            problems.append(f"{name}: condition label mismatch")
        budget = run["completed_steps"] * config["batch_size"] * config["model"]["context"]
        if run["tokens_processed"] != budget or run["completed_steps"] != config["steps"]:
            problems.append(f"{name}: incomplete or inconsistent training-token budget")
        if len(run["history"]) != run["completed_steps"]:
            problems.append(f"{name}: need one history record per completed update")
        if [row["step"] for row in run["history"]] != list(range(1, run["completed_steps"] + 1)):
            problems.append(f"{name}: history update indices are not contiguous")
        for field in ("tokens_processed", "initial_model_sha256", "data_manifest_sha256", "tokenizer_sha256"):
            if run[field] != baseline[field]:
                problems.append(f"{name}: unmatched {field}")
        expected = json.loads(json.dumps(baseline["config"]))
        expected["model"]["norm"] = "post" if name == "post" else "pre"
        if config != expected:
            problems.append(f"{name}: only the specified norm/data condition may change")
    for record in value["decoding"] or []:
        if record["checkpoint_sha256"] != baseline["checkpoint_sha256"]:
            problems.append("decoding: checkpoint must match the baseline")
    if value["final_evaluation"] is not None:
        for name in ("baseline", "dedup"):
            if value["final_evaluation"]["checkpoint_sha256"][name] != runs[name]["checkpoint_sha256"]:
                problems.append(f"{name}: final evaluation checkpoint mismatch")
    return problems


def check_file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        return ["results.json is missing or exceeds 2 MiB"]
    try:
        return validate(json.loads(path.read_text()))
    except (OSError, ValueError, TypeError, RecursionError, OverflowError) as exc:
        return [f"results.json cannot be validated: {exc}"]


if __name__ == "__main__":
    import sys
    issues = check_file(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results.json")
    print("\n".join(issues) if issues else "results.json passes structure and arithmetic checks")
    raise SystemExit(bool(issues))
