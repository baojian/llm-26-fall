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
    assert document.articles == len(DATA["projects"])
    assert 30 <= document.articles <= 50
    assert len(document.ids) == len(set(document.ids))
    assert "clm-context-management" in document.ids
    assert DATA["categories"][0] == "Student-proposed project"
    assert document.ids.index("student-proposed-project") < document.ids.index("multilingual-tokenizer-audit")
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


@pytest.mark.parametrize("source_ids", [[], ["qwen35"]])
def test_reading_lists_have_no_fixed_count_requirement(source_ids):
    data = deepcopy(DATA)
    project = data["projects"][0]
    project["sources"] = source_ids
    project["reading_notes"] = {key: project["reading_notes"][key] for key in source_ids}
    output = catalog.render(data, TEMPLATE)
    card = output.split(f'id="{project["id"]}"', 1)[1].split('</article>', 1)[0]
    assert ('class="project-readings"' in card) == bool(source_ids)


def test_duplicate_references_are_rejected():
    data = deepcopy(DATA)
    data["projects"][0]["sources"] = ["qwen35", "qwen35"]
    with pytest.raises(ValueError, match="Duplicate references"):
        catalog.validate(data)


def test_each_reference_requires_a_reason_to_read_it():
    data = deepcopy(DATA)
    data["projects"][0]["reading_notes"].pop("qwen35")
    with pytest.raises(ValueError, match="Explain each reference"):
        catalog.validate(data)


@pytest.mark.parametrize("field,value", [("max_gpus", -1), ("gpu_hours", 9), ("gpu_hours", -1)])
def test_projects_cannot_exceed_the_shared_gpu_budget(field, value):
    data = deepcopy(DATA)
    project = next(p for p in data["projects"] if p["compute"] == "Training")
    project["resources"][field] = value
    with pytest.raises(ValueError):
        catalog.validate(data)


def test_cpu_filter_cannot_hide_a_gpu_requirement():
    data = deepcopy(DATA)
    data["projects"][0]["resources"]["gpu_hours"] = 1
    with pytest.raises(ValueError, match="Inconsistent CPU"):
        catalog.validate(data)


@pytest.mark.parametrize("gpu_count", [1, 2, 4])
def test_gpu_counts_use_per_device_memory_and_total_hours(gpu_count):
    data = deepcopy(DATA)
    project = next(p for p in data["projects"] if p["compute"] == "Training")
    project["resources"].update(max_gpus=gpu_count, gpu_hours=4)
    output = catalog.render(data, TEMPLATE)
    card = output.split(f'id="{project["id"]}"', 1)[1].split('</article>', 1)[0]
    label = f"{gpu_count} GPU" + ("s" if gpu_count != 1 else "")
    assert f"{label} · short training" in card
    assert "memory target per GPU" in card
    assert "up to 4 GPU-hours total across all devices and runs" in card
    assert ("high demand" in card) == (gpu_count > data["resource_policy"]["recommended_max_gpus"])


def test_optional_two_gpu_plan_preserves_the_one_gpu_minimum():
    output = catalog.render(DATA, TEMPLATE)
    card = output.split('id="small-scale-scaling-prediction"', 1)[1].split('</article>', 1)[0]
    assert "1–2 GPUs · short training" in card
    assert "Minimum: one GPU" in card
    data = deepcopy(DATA)
    project = next(p for p in data["projects"] if p["id"] == "small-scale-scaling-prediction")
    project["resources"]["min_gpus"] = 3
    with pytest.raises(ValueError, match="GPU range"):
        catalog.validate(data)


def test_external_resources_are_separate_from_the_course_pool_request():
    data = deepcopy(DATA)
    project = next(p for p in data["projects"] if p["compute"] == "Training")
    project["resources"].update(gpu_hours=16, shared_gpu_hours=4)
    output = catalog.render(data, TEMPLATE)
    assert "up to 16 GPU-hours total across all devices and runs" in output
    assert "Up to 4 GPU-hours would use the course pool" in output
    project["resources"]["shared_gpu_hours"] = 9
    with pytest.raises(ValueError, match="shared-pool planning ceiling"):
        catalog.validate(data)
    project["resources"]["shared_gpu_hours"] = 17
    with pytest.raises(ValueError, match="Invalid shared-pool"):
        catalog.validate(data)


def test_contents_follow_six_sections_and_link_every_candidate_topic():
    output = catalog.render(DATA, TEMPLATE)
    main = output.split('<main ', 1)[1]
    sections = ["overview", "compute-guide", "choose-your-own", "upstream", "catalog", "others"]
    assert sorted(sections, key=lambda identifier: main.index(f'id="{identifier}"')) == sections
    sidebar = output.split('<aside class="catalog-sidebar">', 1)[1].split('</aside>', 1)[0]
    for identifier in sections:
        assert f'href="#{identifier}"' in sidebar
    candidates = [project for project in DATA["projects"] if project["id"] != catalog.OPEN_CHOICE_ID]
    for topic in {project["topic"] for project in candidates}:
        assert f'href="#{catalog.topic_id(topic)}"' in sidebar
        assert f'id="{catalog.topic_id(topic)}"' in main
    candidate_list = main.split('<div id="project-list">', 1)[1].split('<section class="guide-panel" id="others"', 1)[0]
    assert candidate_list.count('<article ') == 48
    assert f'id="{catalog.OPEN_CHOICE_ID}"' not in candidate_list


def test_survey_only_project_category_cannot_be_reintroduced():
    data = deepcopy(DATA)
    data["categories"].append("Survey paper")
    with pytest.raises(ValueError, match="Survey-only"):
        catalog.validate(data)
