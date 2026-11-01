"""Build the standalone foundation-model note from the pinned catalog snapshot.

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
        if identifier in seen:
            raise ValueError(f"Duplicate selection: {identifier}")
        seen.add(identifier)
        if decision["stage"] != "base" or not decision.get("base_evidence_url"):
            raise ValueError(f"Base-checkpoint evidence required: {identifier}")
        if identifier not in by_id and identifier not in original:
            raise ValueError(f"Selected model absent from source data: {identifier}")
        if decision.get("release_year") is not None and not decision.get("release_year_source_url"):
            raise ValueError(f"Release-year source required: {identifier}")
        snapshot = by_id.get(identifier, {})
        source = original.get(identifier) or snapshot
        # The curated measurements include inspected companion-file additions;
        # bulk-only rows retain their narrower serialized-JSON counting scope.
        selected.append({
            **decision,
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

    measured = [r for r in rows if r["tokenizer_vocab_size"] is not None and r["release_year"] is not None]
    fig, ax = plt.subplots(figsize=(11.4, 5.6))
    fig.subplots_adjust(left=0.09, right=0.95, bottom=0.16, top=0.91)
    clean(ax)
    # Year is the actual plotted variable. No January-1 imputation or date jitter.
    for row in sorted(measured, key=lambda r: r["model_id"]):
        x, y = row["release_year"], row["tokenizer_vocab_size"]
        point = ax.scatter(x, y, s=55, color=amber if y < 50000 else blue,
                           edgecolors="white", linewidths=0.7, zorder=3)
        point.set_url(row["base_evidence_url"])
    annotations = {
        "openai-community/openai-gpt": (12, -24),
        "openai-community/gpt2-xl": (-25, 20),
        "bigscience/bloom": (-45, -24),
        "mistralai/Mistral-7B-v0.1": (-62, -27),
        "Qwen/Qwen-7B": (-68, 15),
        "Qwen/Qwen2.5-7B": (-35, 19),
        "ibm-granite/granite-3.0-8b-base": (10, -25),
        "moonshotai/Kimi-K2-Base": (-50, 22),
        "Qwen/Qwen3.5-0.8B-Base": (-112, 17),
    }
    for row in measured:
        if row["model_id"] in annotations:
            ax.annotate(row["label"], (row["release_year"], row["tokenizer_vocab_size"]),
                        xytext=annotations[row["model_id"]], textcoords="offset points",
                        fontsize=9, color="#263947", arrowprops={"arrowstyle": "-", "color": "#b4c2ca", "lw": 0.7})
    ax.set_xticks(range(2018, 2027))
    ax.set_xlim(2017.65, 2026.5)
    ax.set_ylim(0, 280000)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
    ax.set_ylabel("Measured tokenizer entries", labelpad=12)
    ax.set_xlabel("Documented release year", labelpad=13)
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


def table_rows(rows):
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


def build_outputs():
    selection = json.loads((DATA / "foundation-selection.json").read_text())
    catalog, manifest = read_snapshot(DATA / "snapshots" / selection["snapshot_date"])
    curated = json.loads((DATA / "models.json").read_text())
    rows = select_models(selection, catalog, curated)
    figures = make_figures(rows)
    measured = [r for r in rows if r["tokenizer_vocab_size"] is not None]
    plotted = [r for r in measured if r["release_year"] is not None]
    unavailable = [r["label"] for r in rows if r["tokenizer_vocab_size"] is None]
    payload = {"snapshot_date": selection["snapshot_date"], "snapshot_sha256": manifest["archive_sha256"],
               "scope": selection["scope"], "ranking": selection["ranking"], "models": rows}
    raw = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    replacements = {
        "TOTAL": str(len(rows)), "MEASURED": str(len(measured)), "PLOTTED": str(len(plotted)),
        "RANKED": str(len(popularity_ranking(rows))), "ROWS": table_rows(rows),
        "MISSING": html.escape(", ".join(unavailable)), "MISSING_COUNT": str(len(unavailable)),
        "MISSING_DATES": str(len(measured) - len(plotted)),
        "JSON": raw.replace("<", "\\u003c"),
        "TIMELINE": inline_svg(figures["foundation-timeline.svg"], "timeline", "Measured vocabulary size by documented release year for reviewed foundation checkpoints"),
        "POPULARITY": inline_svg(figures["foundation-popularity.svg"], "popularity", "Twelve highest-download checkpoints within the reviewed selection, with measured vocabulary sizes"),
    }
    page = (DATA / "note-template.html").read_text()
    for key, value in replacements.items():
        page = page.replace(f"@@{key}@@", value)
    if re.search(r"@@[A-Z_]+@@", page):
        raise ValueError("Unfilled reader placeholder")
    return {"foundation-models.html": page, "foundation-models.json": raw,
            "foundation-models.csv": buffer.getvalue(), **{f"figures/{k}": v for k, v in figures.items()}}


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
