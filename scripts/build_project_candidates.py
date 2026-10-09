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
OPEN_CHOICE_ID = "student-proposed-project"
TEXT_FIELDS = (
    "title", "summary", "question", "scope", "baseline", "evaluation",
    "outcome", "preparation", "fallback", "status",
)


def validate(data: dict) -> None:
    """Reject entries that would break navigation or omit proposal essentials."""
    policy = data["resource_policy"]
    if type(policy["recommended_max_gpus"]) is not int or policy["recommended_max_gpus"] < 1:
        raise ValueError("Invalid recommended GPU count")
    if policy["status"] not in {"proposed", "confirmed"}:
        raise ValueError("Invalid resource policy status")
    if type(policy["max_gpu_hours"]) not in (int, float) or not 0 < policy["max_gpu_hours"] < float("inf"):
        raise ValueError("Invalid catalog GPU-time ceiling")
    if "Survey paper" in data["categories"]:
        raise ValueError("Survey-only projects are not eligible")
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
        if len(set(project["sources"])) != len(project["sources"]):
            raise ValueError(f"Duplicate references: {identifier}")
        if len({data["sources"][key]["url"] for key in project["sources"]}) != len(project["sources"]):
            raise ValueError(f"References must use distinct primary URLs: {identifier}")
        reading_notes = project.get("reading_notes", {})
        if set(reading_notes) != set(project["sources"]) or any(
            not isinstance(note, str) or not note.strip() for note in reading_notes.values()
        ):
            raise ValueError(f"Explain each reference: {identifier}")
        resources = project.get("resources", {})
        if type(resources.get("max_gpus")) is not int or resources["max_gpus"] < 0:
            raise ValueError(f"Invalid GPU count: {identifier}")
        minimum = resources.get("min_gpus", resources["max_gpus"])
        if type(minimum) is not int or not 0 <= minimum <= resources["max_gpus"]:
            raise ValueError(f"Invalid GPU range: {identifier}")
        if minimum == 0 and resources["max_gpus"] and project["compute"] != "Propose a budget":
            raise ValueError(f"CPU/GPU choice needs a proposed scope: {identifier}")
        for field in ("gpu_hours", "gpu_memory_gb"):
            value = resources.get(field)
            if type(value) not in (int, float) or not 0 <= value < float("inf"):
                raise ValueError(f"Invalid resource estimate: {identifier}")
        shared_hours = resources.get("shared_gpu_hours", resources["gpu_hours"])
        if type(shared_hours) not in (int, float) or not 0 <= shared_hours <= resources["gpu_hours"]:
            raise ValueError(f"Invalid shared-pool GPU time: {identifier}")
        if shared_hours > policy["max_gpu_hours"]:
            raise ValueError(f"GPU time exceeds shared-pool planning ceiling: {identifier}")
        if resources["max_gpus"] == 0:
            if resources["gpu_hours"] != 0 or resources["gpu_memory_gb"] != 0 or project["compute"] != "CPU":
                raise ValueError(f"Inconsistent CPU budget: {identifier}")
        elif project["compute"] == "CPU" or not resources["gpu_hours"] or not resources["gpu_memory_gb"]:
            raise ValueError(f"Missing GPU budget: {identifier}")
        if not isinstance(resources.get("note"), str) or not resources["note"].strip():
            raise ValueError(f"Missing resource scope: {identifier}")
    if not set(data["featured"]) <= ids:
        raise ValueError("Featured project does not exist")
    if OPEN_CHOICE_ID not in ids:
        raise ValueError("Missing student-proposed project")
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


def topic_id(topic: str) -> str:
    return "topic-" + re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")


def render(data: dict, template: str) -> str:
    validate(data)
    cards = {}
    for project in sorted(data["projects"], key=lambda item: item.get("display_priority", 1)):
        e = {key: escape(value) for key, value in project.items() if isinstance(value, str)}
        resources = project["resources"]
        if resources["max_gpus"] == 0:
            resource_class, resource_label = "cpu", "CPU · no GPU required"
            budget = "Laptop CPU, roughly 8–16 GB RAM."
        else:
            resource_class = "training" if project["compute"] == "Training" else "inference"
            maximum = resources["max_gpus"]
            minimum = resources.get("min_gpus", maximum)
            if minimum == 0:
                hardware = "CPU or 1 GPU" if maximum == 1 else f"CPU, 1–{maximum} GPUs"
            elif minimum == maximum:
                hardware = f"{maximum} GPU" + ("s" if maximum != 1 else "")
            else:
                hardware = f"{minimum}–{maximum} GPUs"
            activity = "short training" if resource_class == "training" else "inference"
            if project["compute"] == "Propose a budget":
                activity = "propose scope"
            resource_label = f"{hardware} · {activity}"
            if maximum > data["resource_policy"]["recommended_max_gpus"]:
                resource_class = "heavy"
                resource_label += " · high demand"
            budget = (
                f"{resources['gpu_memory_gb']:g} GB memory target per GPU; "
                f"up to {resources['gpu_hours']:g} GPU-hours total across all devices and runs."
            )
            if "shared_gpu_hours" in resources:
                budget += f" Up to {resources['shared_gpu_hours']:g} GPU-hours would use the course pool."
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
        heading = 3 if project["id"] == OPEN_CHOICE_ID else 4
        readings = (
            f'<div class="project-readings"><h{heading + 1} class="readings-title">Read first</h{heading + 1}>'
            f'<ol class="sources">{"".join(references)}</ol></div>'
            if references else ""
        )
        cards[project["id"]] = f'''<article class="project-card" id="{e['id']}" data-category="{categories}" data-topic="{e['topic']}" data-compute="{e['compute']}" data-search="{search}">
  <div class="card-meta"><span>{e['topic']}</span><span class="resource-badge resource-{resource_class}">{resource_label}</span><span class="review-status">{e['status']}</span></div>
  <h{heading} class="project-title"><a class="project-link" href="#{e['id']}"><span class="project-number">{project_label(project)}</span> {e['title']}</a></h{heading}>
  <p class="project-summary">{e['summary']}</p>
  <div class="tags" aria-label="Project approaches">{tags}</div>
  <p class="question"><strong>Question:</strong> {e['question']}</p>
  <div class="resource-summary"><p><strong>{budget_status if resources['max_gpus'] else 'Compute'}:</strong> {budget}</p><p>{escape(resources['note'])}</p></div>
  {readings}
  <details><summary>Experiment plan, evaluation and smaller fallback</summary><dl>{details}</dl></details>
</article>'''
    candidates = [project for project in data["projects"] if project["id"] != OPEN_CHOICE_ID]
    topics = [topic for topic in data["topics"] if any(project["topic"] == topic for project in candidates)]
    groups = []
    topic_links = []
    for topic in topics:
        projects = [project for project in candidates if project["topic"] == topic]
        identifier = topic_id(topic)
        groups.append(
            f'<section class="topic-group" id="{identifier}" data-topic="{escape(topic)}" aria-labelledby="{identifier}-title">'
            f'<h3 class="topic-title" id="{identifier}-title">{escape(topic)} <span>{len(projects)} projects</span></h3>'
            + "\n".join(cards[project["id"]] for project in projects) + '</section>'
        )
        topic_links.append(f'<li><a href="#{identifier}">{escape(topic)} <span>{len(projects)}</span></a></li>')
    lookup = {project["id"]: project for project in data["projects"]}
    featured = "".join(
        f'<li><a href="#{identifier}">{project_label(lookup[identifier])} · {escape(lookup[identifier]["title"])}</a></li>'
        for identifier in data["featured"]
    )
    substitutions = {
        "COUNT": str(len(candidates)), "VERIFIED_ON": escape(data["verified_on"]),
        "RECOMMENDED_GPUS": str(data["resource_policy"]["recommended_max_gpus"]),
        "HIGH_DEMAND_GPUS": str(data["resource_policy"]["recommended_max_gpus"] + 1),
        "CATEGORY_OPTIONS": options([value for value in data["categories"] if any(value in project["categories"] for project in candidates)]),
        "TOPIC_OPTIONS": options(topics),
        "COMPUTE_OPTIONS": options([value for value in data["compute"] if any(project["compute"] == value for project in candidates)]),
        "FEATURED": featured, "OPEN_CHOICE": cards[OPEN_CHOICE_ID],
        "TOPIC_LINKS": "\n".join(topic_links), "CARDS": "\n".join(groups),
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
