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
        if not project["sources"] or not set(project["sources"]) <= data["sources"].keys():
            raise ValueError(f"Missing source: {identifier}")
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


def render(data: dict, template: str) -> str:
    validate(data)
    cards = []
    for project in data["projects"]:
        e = {key: escape(value) for key, value in project.items() if isinstance(value, str)}
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
                f' · <a href="{escape(source["artifact"])}">Code / data / model</a>'
                if "artifact" in source else ""
            )
            references.append(
                f'<li><a href="{escape(source["url"])}">{escape(source["title"])}</a>'
                f' — {escape(source["authors"])}; {escape(source["year"])}.'
                f' <span class="source-kind">{escape(source["kind"])}</span>{artifact}</li>'
            )
        search = escape(" ".join(search_parts).lower())
        categories = escape("|".join(project["categories"]))
        cards.append(f'''<article class="project-card" id="{e['id']}" data-category="{categories}" data-topic="{e['topic']}" data-compute="{e['compute']}" data-search="{search}">
  <div class="card-meta"><span>{e['topic']}</span><span>{e['compute']}</span><span class="review-status">{e['status']}</span></div>
  <h2><a class="project-link" href="#{e['id']}"><span class="project-number">{project['number']:02d}</span> {e['title']}</a></h2>
  <p class="project-summary">{e['summary']}</p>
  <div class="tags" aria-label="Project approaches">{tags}</div>
  <p class="question"><strong>Question:</strong> {e['question']}</p>
  <details><summary>Scope, evaluation, resources and sources</summary><dl>{details}</dl>
    <h3>Starting sources</h3><ul class="sources">{''.join(references)}</ul>
  </details>
</article>''')
    lookup = {project["id"]: project for project in data["projects"]}
    featured = "".join(
        f'<li><a href="#{identifier}">{lookup[identifier]["number"]:02d} · {escape(lookup[identifier]["title"])}</a></li>'
        for identifier in data["featured"]
    )
    substitutions = {
        "COUNT": str(len(data["projects"])), "VERIFIED_ON": escape(data["verified_on"]),
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
