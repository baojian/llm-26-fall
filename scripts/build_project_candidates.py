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
TOPIC_ZH = {
    "Student-defined topic": "自拟主题",
    "Data and tokenization": "数据与分词",
    "Adaptation and training": "模型适配与训练",
    "Reasoning and evaluation": "推理与评测",
    "Retrieval and evidence": "检索与证据",
    "Agents and memory": "智能体与记忆",
    "Efficient inference": "高效推理",
    "Interpretability and language": "可解释性与语言",
    "Diffusion and multimodal": "扩散模型与多模态",
    "Open-source contributions": "开源贡献",
}
CATEGORY_ZH = {
    "Student-proposed project": "自拟项目",
    "Case study": "案例研究",
    "Paper replication": "论文复现",
    "New task": "新任务",
    "Research problem": "研究问题",
}
COMPUTE_ZH = {
    "CPU": "CPU",
    "Inference": "推理",
    "Training": "训练",
    "Propose a budget": "自拟算力预算",
}
COMPUTE_DETAILS_ZH = {
    "CPU": "笔记本电脑即可，约需 8–16 GB 内存。限制数据规模并缓存重复计算；最低配置无需 GPU，建议先测量一小段 CPU 运行时间。",
    "Inference": "先使用一张课程 GPU，通常以 12–24 GB 显存为目标，或使用合适的本地量化推理环境。复用已发布的 0.6B–2B 模型，限制上下文和输出长度，并缓存生成结果。候选题可提供双卡方案；所有设备和重跑都计入 GPU 总时长。",
    "Training": "先使用一张课程 GPU，通常以 16–24 GB 显存为目标。训练 5M–30M 参数教学模型，或在 0.6B 检查点上训练适配器。双卡可并行运行不同随机种子或配置；所有设备和运行都计入 GPU 总时长，并先试测显存与耗时。",
    "Propose a budget": "说明 CPU、单卡、双卡或更大规模配置，并区分最低需求与可选硬件。列出 CPU 核数/内存、GPU 型号和单卡显存、存储、运行次数、GPU 总时长及缩小版备选方案；另行注明自有资源和课程 GPU 池申请。",
}
DETAIL_LABEL_ZH = {
    "Minimum scope and data": "最低范围与数据",
    "Baseline and controls": "基线与对照",
    "Evaluation": "评测方式",
    "Minimum complete outcome": "最低完整成果",
    "Preparation": "前置准备",
    "Planning budget": "算力预算规划",
    "Risks and smaller fallback": "风险与缩小后的备选方案",
    "Possible upstream contribution": "可能的上游贡献",
}
TEXT_FIELDS = (
    "title", "summary", "question", "scope", "baseline", "evaluation",
    "outcome", "preparation", "fallback", "status", "title_zh", "summary_zh",
    "question_zh",
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
        if not isinstance(resources.get("note_zh"), str) or not resources["note_zh"].strip():
            raise ValueError(f"Missing Chinese resource scope: {identifier}")
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


def localized(text: str, chinese: str, tag: str = "span", extra: str = "") -> str:
    """Render text that the shared course language toggle can switch."""
    return (
        f'<{tag}{extra} data-en="{escape(text)}" data-zh="{escape(chinese)}">'
        f'{escape(text)}</{tag}>'
    )


def options(values: list | dict, translations: dict[str, str]) -> str:
    return "".join(
        f'<option value="{escape(value)}" data-en="{escape(value)}" '
        f'data-zh="{escape(translations[value])}">{escape(value)}</option>'
        for value in values
    )


def project_label(project: dict) -> str:
    return escape(project.get("display_label", f"{project['number']:02d}"))


def project_number(project: dict) -> str:
    label = project_label(project)
    chinese = "自拟选题" if project["id"] == OPEN_CHOICE_ID else label
    return localized(label, chinese, extra=' class="project-number"')


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
            resource_label_zh = "CPU · 无需 GPU"
            budget_zh = "笔记本 CPU，约 8–16 GB 内存。"
        else:
            resource_class = "training" if project["compute"] == "Training" else "inference"
            maximum = resources["max_gpus"]
            minimum = resources.get("min_gpus", maximum)
            if minimum == 0:
                hardware = "CPU or 1 GPU" if maximum == 1 else f"CPU, 1–{maximum} GPUs"
                hardware_zh = "CPU 或 1 张 GPU" if maximum == 1 else f"CPU、1–{maximum} 张 GPU"
            elif minimum == maximum:
                hardware = f"{maximum} GPU" + ("s" if maximum != 1 else "")
                hardware_zh = f"{maximum} 张 GPU"
            else:
                hardware = f"{minimum}–{maximum} GPUs"
                hardware_zh = f"{minimum}–{maximum} 张 GPU"
            activity = "short training" if resource_class == "training" else "inference"
            activity_zh = "短时训练" if resource_class == "training" else "推理"
            if project["compute"] == "Propose a budget":
                activity = "propose scope"
                activity_zh = "自拟方案"
            resource_label = f"{hardware} · {activity}"
            resource_label_zh = f"{hardware_zh} · {activity_zh}"
            if maximum > data["resource_policy"]["recommended_max_gpus"]:
                resource_class = "heavy"
                resource_label += " · high demand"
                resource_label_zh += " · 高需求"
            budget = (
                f"{resources['gpu_memory_gb']:g} GB memory target per GPU; "
                f"up to {resources['gpu_hours']:g} GPU-hours total across all devices and runs."
            )
            budget_zh = (
                f"每张 GPU 显存目标 {resources['gpu_memory_gb']:g} GB；所有设备和运行合计最多 "
                f"{resources['gpu_hours']:g} GPU 小时。"
            )
            if "shared_gpu_hours" in resources:
                budget += f" Up to {resources['shared_gpu_hours']:g} GPU-hours would use the course pool."
                budget_zh += f"其中课程 GPU 池最多使用 {resources['shared_gpu_hours']:g} GPU 小时。"
        budget_status = "Proposed ceiling" if data["resource_policy"]["status"] == "proposed" else "Planning ceiling"
        budget_status_zh = "建议上限" if data["resource_policy"]["status"] == "proposed" else "规划上限"
        tags = "".join(
            localized(category, CATEGORY_ZH[category], extra=' class="tag"')
            for category in project["categories"]
        )
        fields = [
            ("Minimum scope and data", DETAIL_LABEL_ZH["Minimum scope and data"], e["scope"]),
            ("Baseline and controls", DETAIL_LABEL_ZH["Baseline and controls"], e["baseline"]),
            ("Evaluation", DETAIL_LABEL_ZH["Evaluation"], e["evaluation"]),
            ("Minimum complete outcome", DETAIL_LABEL_ZH["Minimum complete outcome"], e["outcome"]),
            ("Preparation", DETAIL_LABEL_ZH["Preparation"], e["preparation"]),
            ("Planning budget", DETAIL_LABEL_ZH["Planning budget"], localized(
                data["compute"][project["compute"]], COMPUTE_DETAILS_ZH[project["compute"]]
            )),
            ("Risks and smaller fallback", DETAIL_LABEL_ZH["Risks and smaller fallback"], e["fallback"]),
        ]
        if project["upstream"]:
            fields.append(("Possible upstream contribution", DETAIL_LABEL_ZH["Possible upstream contribution"], e["upstream"]))
        details = "".join(
            f"{localized(label, label_zh, 'dt')}<dd>{value}</dd>"
            for label, label_zh, value in fields
        )
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
        readings_heading = localized(
            "Read first", "建议先读", f"h{heading + 1}", ' class="readings-title"'
        )
        readings = (
            f'<div class="project-readings">{readings_heading}'
            f'<ol class="sources">{"".join(references)}</ol></div>'
            if references else ""
        )
        title_zh = escape(project["title_zh"])
        summary_zh = escape(project["summary_zh"])
        question_zh = escape(project["question_zh"])
        topic = localized(project["topic"], TOPIC_ZH[project["topic"]])
        status_zh = "待审核" if project["status"] == "Unreviewed" else "待提交方案"
        budget_label = "Compute" if resources["max_gpus"] == 0 else budget_status
        budget_label_zh = "计算资源" if resources["max_gpus"] == 0 else budget_status_zh
        cards[project["id"]] = f'''<article class="project-card" id="{e['id']}" data-category="{categories}" data-topic="{e['topic']}" data-compute="{e['compute']}" data-search="{search}">
  <div class="card-meta">{topic}<span class="resource-badge resource-{resource_class}" data-en="{escape(resource_label)}" data-zh="{escape(resource_label_zh)}">{escape(resource_label)}</span>{localized(project['status'], status_zh, extra=' class="review-status"')}</div>
  <h{heading} class="project-title"><a class="project-link" href="#{e['id']}">{project_number(project)} {localized(project['title'], project['title_zh'])}</a></h{heading}>
  <p class="project-summary" data-en="{e['summary']}" data-zh="{summary_zh}">{e['summary']}</p>
  <div class="tags" aria-label="Project approaches" data-en-label="Project approaches" data-zh-label="项目类型">{tags}</div>
  <p class="question"><strong data-en="Question:" data-zh="研究问题：">Question:</strong> <span data-en="{e['question']}" data-zh="{question_zh}">{e['question']}</span></p>
  <div class="resource-summary"><p><strong data-en="{escape(budget_label)}:" data-zh="{escape(budget_label_zh)}：">{escape(budget_label)}:</strong> <span data-en="{escape(budget)}" data-zh="{escape(budget_zh)}">{escape(budget)}</span></p><p data-en="{escape(resources['note'])}" data-zh="{escape(resources['note_zh'])}">{escape(resources['note'])}</p></div>
  {readings}
  <details><summary data-en="Experiment plan, evaluation and smaller fallback" data-zh="详细实验方案与评测（正文为英文）">Experiment plan, evaluation and smaller fallback</summary><dl>{details}</dl></details>
</article>'''
    candidates = [project for project in data["projects"] if project["id"] != OPEN_CHOICE_ID]
    topics = [topic for topic in data["topics"] if any(project["topic"] == topic for project in candidates)]
    groups = []
    topic_links = []
    for topic in topics:
        projects = [project for project in candidates if project["topic"] == topic]
        identifier = topic_id(topic)
        count_label = localized(
            f"{len(projects)} projects", f"{len(projects)} 个选题", extra=' class="topic-count"'
        )
        groups.append(
            f'<section class="topic-group" id="{identifier}" data-topic="{escape(topic)}" aria-labelledby="{identifier}-title">'
            f'<h3 class="topic-title" id="{identifier}-title">{localized(topic, TOPIC_ZH[topic])} '
            f'{count_label}</h3>'
            + "\n".join(cards[project["id"]] for project in projects) + '</section>'
        )
        topic_links.append(f'<li><a href="#{identifier}">{localized(topic, TOPIC_ZH[topic])} <span>{len(projects)}</span></a></li>')
    lookup = {project["id"]: project for project in data["projects"]}
    featured = "".join(
        f'<li><a href="#{identifier}">{project_number(lookup[identifier])} · '
        f'{localized(lookup[identifier]["title"], lookup[identifier]["title_zh"])}</a></li>'
        for identifier in data["featured"]
    )
    substitutions = {
        "COUNT": str(len(candidates)), "VERIFIED_ON": escape(data["verified_on"]),
        "RECOMMENDED_GPUS": str(data["resource_policy"]["recommended_max_gpus"]),
        "MAX_GPU_HOURS": f"{data['resource_policy']['max_gpu_hours']:g}",
        "HIGH_DEMAND_GPUS": str(data["resource_policy"]["recommended_max_gpus"] + 1),
        "CATEGORY_OPTIONS": options(
            [value for value in data["categories"] if any(value in project["categories"] for project in candidates)],
            CATEGORY_ZH,
        ),
        "TOPIC_OPTIONS": options(topics, TOPIC_ZH),
        "COMPUTE_OPTIONS": options(
            [value for value in data["compute"] if any(project["compute"] == value for project in candidates)],
            COMPUTE_ZH,
        ),
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
