"""Build the complete Wikipedia inventory and standalone tokenizer note.

Run with Python 3.11 and matplotlib==3.11.1. No network or tokenizer cache is
needed. --check verifies that all committed reader/data/figure files are current.
"""
import argparse
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import re
import zipfile

from tokenizer_catalog_analysis import popularity_ranking

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/model-tokenizer-statistics"


def read_snapshot(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    archive = (directory / manifest["archive"]).read_bytes()
    if hashlib.sha256(archive).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Snapshot archive checksum mismatch")
    entry = next(item for item in manifest["files"] if item["path"] == "model-catalog.json")
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        raw = source.read(entry["path"])
    if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("Catalog checksum mismatch")
    return json.loads(raw), manifest


def select_models(selection, catalog, curated):
    """Use explicit reviewed membership; never infer 'base' from absent metadata."""
    by_id = {row["model_id"]: row for row in catalog}
    original = {row["model_id"]: row for row in curated}
    selected, seen = [], set()
    for decision in selection["models"]:
        identifier = decision["model_id"]
        curated_id = decision.get("curated_model_id", identifier)
        if curated_id != identifier and not decision.get("alias_evidence_url"):
            raise ValueError(f"Repository-rename evidence required: {identifier}")
        if identifier in seen:
            raise ValueError(f"Duplicate selection: {identifier}")
        seen.add(identifier)
        if decision["stage"] != "base" or not decision.get("base_evidence_url"):
            raise ValueError(f"Base-checkpoint evidence required: {identifier}")
        if identifier not in by_id and curated_id not in original:
            raise ValueError(f"Selected model absent from source data: {identifier}")
        if decision.get("release_year") is not None and not decision.get("release_year_source_url"):
            raise ValueError(f"Release-year source required: {identifier}")
        snapshot = by_id.get(identifier, {})
        source = original.get(curated_id) or snapshot
        # The curated measurements include inspected companion-file additions;
        # bulk-only rows retain their narrower serialized-JSON counting scope.
        selected.append({
            **decision,
            "curated_model_id": curated_id,
            "alias_evidence_url": decision.get("alias_evidence_url"),
            "tokenizer_vocab_size": source.get("tokenizer_vocab_size"),
            "tokenizer_count_scope": source.get("tokenizer_count_scope"),
            "tokenizer_status": source.get("tokenizer_status"),
            "tokenizer_source_url": source.get("tokenizer_source_url"),
            "tokenizer_revision": source.get("tokenizer_revision") or source.get("revision"),
            "tokenizer_sha256": source.get("tokenizer_sha256") or source.get("tokenizer_artifact_sha256"),
            "downloads_30d": snapshot.get("downloads_30d"),
            "downloads_observed_on": selection["snapshot_date"],
            "downloads_source_url": f"https://huggingface.co/api/models/{identifier}",
            "parameters_billion": source.get("parameters_billion") or (
                snapshot["parameters_total"] / 1e9 if snapshot.get("parameters_total") else None),
        })
    ranks = {row["model_id"]: rank for rank, row in enumerate(popularity_ranking(selected), 1)}
    for row in selected:
        row["popularity_rank"] = ranks.get(row["model_id"])
    return sorted(selected, key=lambda r: (r["popularity_rank"] or float("inf"), r["model_id"]))


def assemble_inventory(inventory, decisions, bases, catalog, curated, additional):
    """Retain every source row, with independent mapping and measurement status.

    No fuzzy family matching, inherited tokenizer, repository-date fallback, or
    newly observed popularity count is allowed in this join.
    """
    entries = inventory["entries"]
    mappings = decisions["mappings"]
    ids = [row["entry_id"] for row in entries]
    mapping_ids = [row["entry_id"] for row in mappings]
    if len(set(ids)) != len(ids) or len(set(mapping_ids)) != len(mapping_ids):
        raise ValueError("Duplicate inventory or mapping entry")
    if set(ids) != set(mapping_ids):
        raise ValueError("Mappings must account for every inventory row exactly once")
    by_entry = {row["entry_id"]: row for row in mappings}
    by_id = {row["model_id"]: row for row in catalog}
    originals = {row["model_id"]: row for row in curated}
    extra = {row["model_id"]: row for row in additional}
    base_rows = {row["model_id"]: row for row in bases}
    representatives, rows = set(), []
    for entry in entries:
        mapping = by_entry[entry["entry_id"]]
        if entry["listed_name"] != mapping["listed_name"]:
            raise ValueError("Source row changed; review its mapping")
        identifier = mapping["representative_id"]
        if identifier and (identifier in representatives or not mapping.get("mapping_evidence_url")):
            raise ValueError("Duplicate representative or missing mapping evidence")
        if identifier:
            representatives.add(identifier)
        base = base_rows.get(identifier, {})
        curated_id = base.get("curated_model_id", identifier)
        curated_row = originals.get(curated_id, {})
        snapshot = by_id.get(identifier, {})
        candidates = [curated_row, extra.get(identifier, {}), snapshot]
        source = next((r for r in candidates if r.get("tokenizer_vocab_size") is not None),
                      next((r for r in candidates if r), {}))
        count = source.get("tokenizer_vocab_size")
        scope = source.get("tokenizer_count_scope")
        if count is not None and not all((scope, source.get("tokenizer_source_url"),
                                         source.get("tokenizer_revision") or source.get("revision"),
                                         source.get("tokenizer_sha256") or source.get("tokenizer_artifact_sha256"))):
            raise ValueError(f"Incomplete measurement provenance: {identifier}")
        api_encoding = scope in {"public_text_encoding_only", "official_sdk_text_tokenizer"}
        status = ("public_text_encoding" if api_encoding else "measured_tokenizer") if count is not None else (
            "mapped_not_measured" if identifier else "not_mapped")
        date_source = next((r for r in [curated_row, snapshot]
                            if r.get("release_date") and r.get("release_date_source_url")), {})
        year = base.get("release_year") or (int(date_source["release_date"][:4]) if date_source else None)
        year_source = base.get("release_year_source_url") if base.get("release_year") else date_source.get("release_date_source_url")
        downloads = snapshot.get("downloads_30d")
        rows.append({
            **entry, "model_id": identifier, "label": entry["listed_name"],
            "category": mapping["category"], "mapping_evidence_url": mapping["mapping_evidence_url"],
            "mapping_note": mapping["mapping_note"],
            "curated_model_id": curated_id, "alias_evidence_url": base.get("alias_evidence_url"),
            "in_original_base_cohort": identifier in base_rows,
            "release_year": year, "release_year_source_url": year_source,
            "year_differs_from_listing": year is not None and year != entry["listed_year"],
            "timeline_year": year if year is not None else entry["listed_year"],
            "timeline_year_basis": "primary_source" if year is not None else "wikipedia_only",
            "measurement_status": status, "tokenizer_vocab_size": count,
            "tokenizer_count_scope": scope, "tokenizer_status": source.get("tokenizer_status"),
            "tokenizer_source_url": source.get("tokenizer_source_url"),
            "tokenizer_mapping_source_url": source.get("tokenizer_mapping_source_url"),
            "tokenizer_revision": source.get("tokenizer_revision") or source.get("revision"),
            "tokenizer_sha256": source.get("tokenizer_sha256") or source.get("tokenizer_artifact_sha256"),
            "measurement_observed_at": source.get("observed_at") or source.get("collected_at"),
            "downloads_30d": downloads,
            "downloads_observed_on": "2026-10-31" if downloads is not None else None,
            "downloads_source_url": f"https://huggingface.co/api/models/{identifier}" if downloads is not None else None,
        })
    ranks = {r["model_id"]: rank for rank, r in enumerate(popularity_ranking(rows), 1)}
    for row in rows:
        row["popularity_rank"] = ranks.get(row["model_id"])
    return rows


def csv_text(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                         if isinstance(value, (list, dict, bool)) else value for key, value in row.items()})
    return buffer.getvalue()


def inline_svg(raw, prefix, description):
    """Keep inline Matplotlib clip-path identifiers unique across the two figures."""
    text = raw[raw.index("<svg"):]
    identifiers = re.findall(r'\bid="([^"]+)"', text)
    for identifier in sorted(identifiers, key=len, reverse=True):
        text = text.replace(f'id="{identifier}"', f'id="{prefix}-{identifier}"')
        text = text.replace(f'#{identifier}"', f'#{prefix}-{identifier}"')
        text = text.replace(f'#{identifier})', f'#{prefix}-{identifier})')
    return text.replace("<svg ", f'<svg role="img" aria-label="{html.escape(description, quote=True)}" ', 1)


def make_figures(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": "foundation-tokenizers-v1", "font.size": 11})
    blue, amber, grey = "#126d78", "#b45721", "#94a1ae"

    def clean(ax):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#cad3da")
        ax.tick_params(length=0, pad=9, colors="#445260")
        ax.grid(axis="y", color="#e7ecef", linewidth=0.8)
        ax.set_axisbelow(True)

    def save(fig):
        buffer = io.StringIO()
        fig.savefig(buffer, format="svg", metadata={"Date": None}, facecolor="white")
        plt.close(fig)
        # The browser may not have Matplotlib's bundled DejaVu font installed.
        svg = buffer.getvalue().replace("font-family: 'DejaVu Sans'", "font-family: 'DejaVu Sans', Arial, sans-serif")
        return "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"

    measured = [r for r in rows if r["tokenizer_vocab_size"] is not None]
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 8.1), sharex=True,
                             gridspec_kw={"height_ratios": [2.25, 1]})
    fig.subplots_adjust(left=0.09, right=0.95, bottom=0.12, top=0.94, hspace=0.30)
    annotations = {
        "openai-community/openai-gpt": (12, -22),
        "openai-community/gpt2-xl": (-25, 18),
        "bigscience/bloom": (-45, -24),
        "mistralai/Mistral-7B-v0.1": (-75, -24),
        "ibm-granite/granite-3b-code-base-2k": (-105, 19),
        "mistralai/Mistral-Large-Instruct-2411": (15, -24),
        "Qwen/Qwen2.5-7B": (-60, 18),
        "Qwen/Qwen3.5-397B-A17B": (-90, 14),
        "openai/gpt-4-0314": (-47, 15),
        "google/gemini-2.5-pro": (-90, -22),
    }
    for ax, api, title in zip(axes, [False, True],
                              ["Public checkpoint tokenizers", "Closed APIs: mapped public text encodings only"]):
        clean(ax)
        ax.set_title(title, loc="left", fontsize=12, pad=12, color="#263947")
        for row in sorted(measured, key=lambda r: r["model_id"]):
            if (row["measurement_status"] == "public_text_encoding") != api:
                continue
            x, y = row["timeline_year"], row["tokenizer_vocab_size"]
            color = amber if y < 50000 else blue
            primary = row["timeline_year_basis"] == "primary_source"
            point = ax.scatter(x, y, s=58, facecolors=color if primary else "none",
                               edgecolors=color, linewidths=1.2, zorder=3)
            point.set_url(row["tokenizer_source_url"])
            if row["model_id"] in annotations:
                ax.annotate(row["label"], (x, y), xytext=annotations[row["model_id"]],
                            textcoords="offset points", fontsize=9, color="#263947",
                            arrowprops={"arrowstyle": "-", "color": "#b4c2ca", "lw": 0.7})
        ax.set_xticks(range(2018, 2027))
        ax.set_xlim(2017.65, 2026.5)
        ax.set_ylim(0, 285000)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
        ax.set_ylabel("Assigned token IDs", labelpad=12)
    axes[-1].set_xlabel("Release year — primary source when verified; Wikipedia listing otherwise", labelpad=13)
    timeline = save(fig)

    ranked = popularity_ranking(rows)[:12]
    fig, ax = plt.subplots(figsize=(11.4, 6.6))
    fig.subplots_adjust(left=0.29, right=0.79, bottom=0.12, top=0.88)
    clean(ax)
    ax.grid(False, axis="y")
    ax.grid(axis="x", color="#e7ecef", linewidth=0.8)
    for i, row in enumerate(ranked):
        count = row["tokenizer_vocab_size"]
        color = grey if count is None else amber if count < 50000 else blue
        ax.barh(i, row["downloads_30d"] / 1e6, color=color, height=0.57)
        ax.text(row["downloads_30d"] / 1e6 + 0.035, i, f'{row["downloads_30d"] / 1e6:.2f}M',
                va="center", fontsize=9, color="#445260")
        ax.text(1.13, i, f"{count:,}" if count is not None else "Not measured",
                transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=10, color=color)
    ax.text(1.13, 1.06, "Vocabulary", transform=ax.transAxes, fontsize=10, color="#445260")
    ax.set_yticks(range(len(ranked)), [f'{i}. {r["label"]}' for i, r in enumerate(ranked, 1)])
    ax.invert_yaxis()
    ax.set_xlim(0, max(r["downloads_30d"] for r in ranked) / 1e6 * 1.20)
    ax.set_xlabel("30-day repository downloads (millions)", labelpad=12)
    popularity = save(fig)
    return {"foundation-timeline.svg": timeline, "foundation-popularity.svg": popularity}


def base_table_rows(rows):
    output = []
    for row in rows:
        count = row["tokenizer_vocab_size"]
        value = f'{count:,}' if count is not None else "Not measured"
        count_html = (f'<a href="{html.escape(row["tokenizer_source_url"], quote=True)}">{value}</a>'
                      if row.get("tokenizer_source_url") and count is not None else value)
        year = row["release_year"]
        year_html = (f'<a href="{html.escape(row["release_year_source_url"], quote=True)}">{year}</a>'
                     if year is not None else "Unverified")
        parameters = row["parameters_billion"]
        params = f" · {parameters:.3g}B parameters" if parameters is not None else ""
        output.append(f'''<tr data-model="{html.escape(row['model_id'], quote=True)}"
          data-rank="{row['popularity_rank'] or 9999}" data-year="{year or 0}"
          data-vocab="{count if count is not None else -1}" data-downloads="{row['downloads_30d'] if row['downloads_30d'] is not None else -1}">
          <td class="num">{row['popularity_rank'] or '—'}</td>
          <td><a href="{html.escape(row['base_evidence_url'], quote=True)}">{html.escape(row['label'])}</a>
          <small>{html.escape(row['model_id'])}{params}</small></td>
          <td>{year_html}</td><td class="num">{count_html}</td>
          <td class="num">{format(row['downloads_30d'], ',') if row['downloads_30d'] is not None else 'Unranked'}</td></tr>''')
    return "\n".join(output)


def table_rows(rows):
    categories = {"base": "Reviewed base checkpoint", "publisher_release": "Publisher release",
                  "historical_baseline": "Encoder / encoder–decoder baseline",
                  "closed_api": "Closed API", "derivative": "Derived model", "unmapped": "Not classified"}
    statuses = {"measured_tokenizer": "Tokenizer measured", "public_text_encoding": "Public text encoding only",
                "mapped_not_measured": "Mapped; not measured", "not_mapped": "Exact representative not mapped"}
    output = []
    escape = lambda value: html.escape(str(value), quote=True)
    for row in rows:
        count, year = row["tokenizer_vocab_size"], row["release_year"]
        value = f'{count:,}' if count is not None else "Not measured"
        count_html = f'<a href="{escape(row["tokenizer_source_url"])}">{value}</a>' if count is not None else value
        year_html = (f'<a href="{escape(row["release_year_source_url"])}">{year} verified</a>'
                     if year is not None else "Primary year unverified")
        evidence = row["mapping_evidence_url"] or row["source_url"]
        source_detail = row["tokenizer_count_scope"] if count is not None else row["tokenizer_status"]
        output.append(f'''<tr data-entry="{escape(row['entry_id'])}" data-model="{escape(row['model_id'] or '')}"
          data-rank="{row['popularity_rank'] or 9999}" data-year="{row['timeline_year']}"
          data-status="{escape(row['measurement_status'])}" data-category="{escape(row['category'])}"
          data-vocab="{count if count is not None else -1}">
          <td class="num">{row['popularity_rank'] or '—'}</td>
          <td><a href="{escape(evidence)}">{escape(row['listed_name'])}</a>
          <small>{escape(row['model_id'] or row['listed_developer'] or '')}</small>
          <small>{categories[row['category']]}</small></td>
          <td><a href="{escape(row['source_url'])}">{row['listed_year']} listed</a><small>{year_html}</small></td>
          <td class="num">{count_html}</td>
          <td>{statuses[row['measurement_status']]}<small>{escape((source_detail or '').replace('_', ' '))}</small></td>
          <td class="num">{format(row['downloads_30d'], ',') if row['downloads_30d'] is not None else 'Unranked'}</td></tr>''')
    return "\n".join(output)


def build_outputs():
    selection = json.loads((DATA / "foundation-selection.json").read_text())
    catalog, manifest = read_snapshot(DATA / "snapshots" / selection["snapshot_date"])
    curated = json.loads((DATA / "models.json").read_text())
    bases = select_models(selection, catalog, curated)
    inventory = json.loads((DATA / "wikipedia-inventory.json").read_text())
    decisions = json.loads((DATA / "inventory-mappings.json").read_text())
    additional = json.loads((DATA / "inventory-additional-measurements.json").read_text())["models"]
    rows = assemble_inventory(inventory, decisions, bases, catalog, curated, additional)
    figures = make_figures(rows)
    measured = [r for r in rows if r["tokenizer_vocab_size"] is not None]
    primary = [r for r in measured if r["release_year"] is not None]
    mapped = [r for r in rows if r["model_id"]]
    payload = {"snapshot_date": selection["snapshot_date"], "snapshot_sha256": manifest["archive_sha256"],
               "scope": "Every source-table row in the pinned Wikipedia revision, including unknown measurements; at most one reviewed representative per row.",
               "wikipedia_source": inventory["source"],
               "ranking": "Exact representative repositories ranked by known 30-day downloads in the October 31 snapshot; not a global or family-level ranking.",
               "models": rows, "base_models": bases}
    raw = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    coverage = []
    for year, total in inventory["source"]["rows_by_year"].items():
        count = sum(r["listed_year"] == int(year) for r in measured)
        coverage.append(f'<div><strong>{year}</strong><span>{total} listed</span><small>{count} measured</small></div>')
    replacements = {
        "TOTAL": str(len(rows)), "MEASURED": str(len(measured)), "MAPPED": str(len(mapped)),
        "UNMAPPED": str(len(rows) - len(mapped)), "MAPPED_MISSING": str(len(mapped) - len(measured)),
        "TOKENIZERS": str(sum(r["measurement_status"] == "measured_tokenizer" for r in rows)),
        "ENCODINGS": str(sum(r["measurement_status"] == "public_text_encoding" for r in rows)),
        "PRIMARY_DATES": str(len(primary)), "WIKI_DATES": str(len(measured) - len(primary)),
        "RANKED": str(len(popularity_ranking(rows))), "ROWS": table_rows(rows),
        "BASE_ROWS": base_table_rows(bases), "BASE_COUNT": str(len(bases)),
        "BASE_SUPPLEMENTAL": str(sum(r["model_id"] not in {m["model_id"] for m in rows} for r in bases)),
        "YEAR_COUNTS": "".join(coverage), "WIKI_SOURCE": html.escape(inventory["source"]["revision_url"], quote=True),
        "JSON": raw.replace("<", "\\u003c"),
        "TIMELINE": inline_svg(figures["foundation-timeline.svg"], "timeline", "Vocabulary measurements across the Wikipedia inventory; primary-sourced and Wikipedia-only years distinguished; API text encodings in a separate panel"),
        "POPULARITY": inline_svg(figures["foundation-popularity.svg"], "popularity", "Twelve highest-download mapped representatives in the Wikipedia inventory"),
    }
    page = (DATA / "note-template.html").read_text()
    for key, value in replacements.items():
        page = page.replace(f"@@{key}@@", value)
    if re.search(r"@@[A-Z_]+@@", page):
        raise ValueError("Unfilled reader placeholder")
    return {"foundation-models.html": page, "foundation-models.json": raw,
            "foundation-models.csv": csv_text(rows),
            "base-checkpoints.json": json.dumps({"scope": selection["scope"], "models": bases}, indent=2, ensure_ascii=False) + "\n",
            "base-checkpoints.csv": csv_text(bases),
            **{f"figures/{k}": v for k, v in figures.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build_outputs()
    stale = []
    for name, content in outputs.items():
        target = DATA / name
        if args.check:
            if not target.is_file() or target.read_bytes() != content.encode():
                stale.append(name)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.encode())
    if stale:
        parser.exit(1, "Stale tokenizer-note outputs: " + ", ".join(stale) + "\n")
    print(f"{'Checked' if args.check else 'Built'} {len(outputs)} tokenizer-note outputs.")


if __name__ == "__main__":
    main()
