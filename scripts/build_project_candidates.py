"""Render the editable project catalog using only Python's standard library."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import re
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/project-candidates.json"
TEMPLATE = ROOT / "docs/project-candidates.template.html"
OUTPUT = ROOT / "docs/project-candidates.html"
TEXT_FIELDS = (
    "title", "summary", "question", "scope", "baseline", "evaluation",
    "outcome", "preparation", "fallback", "status",
)


def validate(data: dict) -> None:
    """Reject entries that would break navigation or omit proposal essentials."""
    policy = data["resource_policy"]
    if policy["max_gpus"] != 1 or policy["status"] not in {"proposed", "confirmed"}:
        raise ValueError("The catalog supports CPU or one shared GPU")
    if type(policy["max_gpu_hours"]) not in (int, float) or not 0 < policy["max_gpu_hours"] < float("inf"):
        raise ValueError("Invalid catalog GPU-time ceiling")
    ids = set()
    for number, project in enumerate(data["projects"], 1):
        identifier = project["id"]
        if not re.fullmatch(r"[a-z][a-z0-9-]*", identifier) or identifier in ids:
            raise ValueError(f"Invalid or duplicate project ID: {identifier}")
        ids.add(identifier)
        if project["number"] != number:
            raise ValueError(f"Non-sequential project number: {identifier}")
        for field in TEXT_FIELDS:
            if not isinstance(project.get(field), str) or not project[field].strip():
                raise ValueError(f"Missing {field}: {identifier}")
        if not project["categories"] or not set(project["categories"]) <= set(data["categories"]):
            raise ValueError(f"Unknown or empty category: {identifier}")
        if project["topic"] not in data["topics"] or project["compute"] not in data["compute"]:
            raise ValueError(f"Unknown topic or compute budget: {identifier}")
        if not set(project["sources"]) <= data["sources"].keys():
            raise ValueError(f"Missing source: {identifier}")
        if len(project["sources"]) < 2 or len(set(project["sources"])) != len(project["sources"]):
            raise ValueError(f"Provide at least two distinct references: {identifier}")
        if len({data["sources"][key]["url"] for key in project["sources"]}) != len(project["sources"]):
            raise ValueError(f"References must use distinct primary URLs: {identifier}")
        reading_notes = project.get("reading_notes", {})
        if set(reading_notes) != set(project["sources"]) or any(
            not isinstance(note, str) or not note.strip() for note in reading_notes.values()
        ):
            raise ValueError(f"Explain each reference: {identifier}")
        resources = project.get("resources", {})
        if type(resources.get("max_gpus")) is not int or resources["max_gpus"] not in (0, 1):
            raise ValueError(f"Project must fit CPU or one GPU: {identifier}")
        for field in ("gpu_hours", "gpu_memory_gb"):
            value = resources.get(field)
            if type(value) not in (int, float) or not 0 <= value < float("inf"):
                raise ValueError(f"Invalid resource estimate: {identifier}")
        if resources["gpu_hours"] > data["resource_policy"]["max_gpu_hours"]:
            raise ValueError(f"GPU time exceeds catalog ceiling: {identifier}")
        if resources["max_gpus"] == 0:
            if resources["gpu_hours"] != 0 or resources["gpu_memory_gb"] != 0 or project["compute"] != "CPU":
                raise ValueError(f"Inconsistent CPU budget: {identifier}")
        elif project["compute"] == "CPU" or not resources["gpu_hours"] or not resources["gpu_memory_gb"]:
            raise ValueError(f"Missing GPU budget: {identifier}")
        if not isinstance(resources.get("note"), str) or not resources["note"].strip():
            raise ValueError(f"Missing resource scope: {identifier}")
    if not set(data["featured"]) <= ids:
        raise ValueError("Featured project does not exist")
    for key, source in data["sources"].items():
        for field in ("url", "artifact"):
            if field in source:
                url = urlsplit(source[field])
                if url.scheme != "https" or not url.netloc or url.username:
                    raise ValueError(f"Unsafe source URL: {key}")


def options(values: list | dict) -> str:
    return "".join(f'<option value="{escape(value)}">{escape(value)}</option>' for value in values)


def project_label(project: dict) -> str:
    return escape(project.get("display_label", f"{project['number']:02d}"))


def render(data: dict, template: str) -> str:
    validate(data)
    cards = []
    for project in sorted(data["projects"], key=lambda item: item.get("display_priority", 1)):
        e = {key: escape(value) for key, value in project.items() if isinstance(value, str)}
        resources = project["resources"]
        if resources["max_gpus"] == 0:
            resource_class, resource_label = "cpu", "CPU · no GPU required"
            budget = "Laptop CPU, roughly 8–16 GB RAM."
        else:
            resource_class = "training" if project["compute"] == "Training" else "inference"
            resource_label = "1 GPU · short training" if resource_class == "training" else "1 GPU · inference"
            if project["compute"] == "Propose a budget":
                resource_label = "CPU or 1 GPU · propose scope"
            budget = (
                f"{resources['gpu_memory_gb']:g} GB GPU memory target; "
                f"up to {resources['gpu_hours']:g} GPU-hours total."
            )
        budget_status = "Proposed ceiling" if data["resource_policy"]["status"] == "proposed" else "Planning ceiling"
        tags = "".join(f'<span class="tag">{escape(category)}</span>' for category in project["categories"])
        fields = [
            ("Minimum scope and data", e["scope"]),
            ("Baseline and controls", e["baseline"]),
            ("Evaluation", e["evaluation"]),
            ("Minimum complete outcome", e["outcome"]),
            ("Preparation", e["preparation"]),
            ("Planning budget", escape(data["compute"][project["compute"]])),
            ("Risks and smaller fallback", e["fallback"]),
        ]
        if project["upstream"]:
            fields.append(("Possible upstream contribution", e["upstream"]))
        details = "".join(f"<dt>{label}</dt><dd>{value}</dd>" for label, value in fields)
        references = []
        search_parts = [str(value) for value in project.values()]
        for key in project["sources"]:
            source = data["sources"][key]
            search_parts.extend(str(value) for value in source.values())
            artifact = (
                f' · <a href="{escape(source["artifact"])}">'
                f'{escape(source.get("artifact_label", "Code / data / model"))}</a>'
                if "artifact" in source else ""
            )
            references.append(
                f'<li><a href="{escape(source["url"])}">{escape(source["title"])}</a>'
                f' — {escape(source["authors"])}; {escape(source["year"])}.'
                f' <span class="source-kind">{escape(source["kind"])}</span>{artifact}'
                f'<p class="reading-purpose">{escape(project["reading_notes"][key])}</p></li>'
            )
        search = escape(" ".join(search_parts).lower())
        categories = escape("|".join(project["categories"]))
        cards.append(f'''<article class="project-card" id="{e['id']}" data-category="{categories}" data-topic="{e['topic']}" data-compute="{e['compute']}" data-search="{search}">
  <div class="card-meta"><span>{e['topic']}</span><span class="resource-badge resource-{resource_class}">{resource_label}</span><span class="review-status">{e['status']}</span></div>
  <h2><a class="project-link" href="#{e['id']}"><span class="project-number">{project_label(project)}</span> {e['title']}</a></h2>
  <p class="project-summary">{e['summary']}</p>
  <div class="tags" aria-label="Project approaches">{tags}</div>
  <p class="question"><strong>Question:</strong> {e['question']}</p>
  <div class="resource-summary"><p><strong>{budget_status if resources['max_gpus'] else 'Compute'}:</strong> {budget}</p><p>{escape(resources['note'])}</p></div>
  <div class="project-readings"><h3>Read first</h3><ol class="sources">{''.join(references)}</ol></div>
  <details><summary>Experiment plan, evaluation and smaller fallback</summary><dl>{details}</dl></details>
</article>''')
    lookup = {project["id"]: project for project in data["projects"]}
    featured = "".join(
        f'<li><a href="#{identifier}">{project_label(lookup[identifier])} · {escape(lookup[identifier]["title"])}</a></li>'
        for identifier in data["featured"]
    )
    substitutions = {
        "COUNT": str(len(data["projects"])), "VERIFIED_ON": escape(data["verified_on"]),
        "GPU_HOURS": str(data["resource_policy"]["max_gpu_hours"]),
        "BUDGET_STATUS": "Proposed planning ceiling" if data["resource_policy"]["status"] == "proposed" else "Planning ceiling",
        "CATEGORY_OPTIONS": options(data["categories"]), "TOPIC_OPTIONS": options(data["topics"]),
        "COMPUTE_OPTIONS": options(data["compute"]), "FEATURED": featured,
        "CARDS": "\n".join(cards),
    }
    for key, value in substitutions.items():
        token = "{{" + key + "}}"
        if token not in template:
            raise ValueError(f"Template is missing {token}")
        template = template.replace(token, value)
    if re.search(r"\{\{[A-Z_]+\}\}", template):
        raise ValueError("Unresolved template field")
    return template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the committed HTML is stale")
    args = parser.parse_args()
    result = render(json.loads(DATA.read_text()), TEMPLATE.read_text())
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != result:
            raise SystemExit("Catalog HTML is stale. Run: uv run python scripts/build_project_candidates.py")
        print("Catalog HTML is current.")
    else:
        OUTPUT.write_text(result)
        print(f"Built {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
