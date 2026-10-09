"""Checks for catalog publication, navigation, and safe rendering."""

from copy import deepcopy
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("project_catalog", ROOT / "scripts/build_project_candidates.py")
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)
DATA = json.loads((ROOT / "docs/project-candidates.json").read_text())
TEMPLATE = (ROOT / "docs/project-candidates.template.html").read_text()


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.links = []
        self.articles = 0

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.append(attributes["id"])
        if tag in {"a", "link", "script"}:
            self.links.append(attributes.get("href", attributes.get("src", "")))
        if tag == "article":
            self.articles += 1


def test_published_catalog_is_current_and_readable_without_javascript():
    output = catalog.render(DATA, TEMPLATE)
    assert output == (ROOT / "docs/project-candidates.html").read_text()
    document = Document()
    document.feed(output)
    assert document.articles == 45
    assert len(document.ids) == len(set(document.ids))
    assert "clm-context-management" in document.ids
    assert set(DATA["categories"]) == {category for project in DATA["projects"] for category in project["categories"]}
    for project in DATA["projects"]:
        assert project["question"].replace("&", "&amp;") in output
    for link in document.links:
        parsed = urlsplit(link)
        if parsed.scheme or parsed.netloc:
            assert parsed.scheme == "https"
        elif not parsed.path:
            assert unquote(parsed.fragment) in document.ids
        else:
            assert (ROOT / "docs" / unquote(parsed.path)).exists(), link


@pytest.mark.parametrize("field,value", [("id", "bad id"), ("categories", ["Bonus track"]), ("sources", ["missing"]), ("scope", "")])
def test_incomplete_or_invalid_candidates_are_rejected(field, value):
    data = deepcopy(DATA)
    data["projects"][0][field] = value
    with pytest.raises(ValueError):
        catalog.render(data, TEMPLATE)


def test_duplicate_ids_and_missing_featured_projects_are_rejected():
    data = deepcopy(DATA)
    data["projects"][1]["id"] = data["projects"][0]["id"]
    with pytest.raises(ValueError, match="duplicate"):
        catalog.validate(data)
    data = deepcopy(DATA)
    data["featured"].append("does-not-exist")
    with pytest.raises(ValueError, match="Featured"):
        catalog.validate(data)


def test_catalog_text_is_escaped_and_executable_source_links_are_rejected():
    data = deepcopy(DATA)
    data["projects"][0]["question"] = '<script>alert("x")</script>'
    output = catalog.render(data, TEMPLATE)
    assert '<script>alert("x")</script>' not in output
    assert '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;' in output
    data["sources"]["qwen35"]["url"] = "javascript:alert(1)"
    with pytest.raises(ValueError, match="Unsafe"):
        catalog.validate(data)
