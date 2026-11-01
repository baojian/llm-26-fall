"""Extract a complete, attributed factual inventory from a saved Wikipedia page.

uv run --no-project --python 3.11 --with beautifulsoup4==4.13.4 \
  python scripts/import_wikipedia_llms.py --html /path/to/page.html

Only names, dates, developers, links, and source identifiers are retained;
the article's descriptive notes and training/cost claims are not copied.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
URL = "https://en.wikipedia.org/wiki/List_of_large_language_models"


def text_without_references(cell):
    from bs4 import BeautifulSoup
    copy = BeautifulSoup(str(cell), "html.parser")
    for node in copy.select("sup, .sortkey, .mw-editsection"):
        node.decompose()
    text = " ".join(copy.get_text().split())
    return re.sub(r"\s+(-\d)", r"\1", text)


def expanded_rows(table):
    """Expand row/column spans before accessing date and developer columns."""
    carry = {}
    body = table.find("tbody") or table
    for tr in body.find_all("tr", recursive=False):
        cells, column, current = [], 0, {}
        for cell in tr.find_all(["th", "td"], recursive=False):
            while column in carry:
                value, remaining = carry[column]
                cells.append(value)
                if remaining > 1:
                    current[column] = (value, remaining - 1)
                column += 1
            for _ in range(int(cell.get("colspan", 1))):
                cells.append(cell)
                rowspan = int(cell.get("rowspan", 1))
                if rowspan > 1:
                    current[column] = (cell, rowspan - 1)
                column += 1
        while column in carry:
            value, remaining = carry[column]
            cells.append(value)
            if remaining > 1:
                current[column] = (value, remaining - 1)
            column += 1
        carry = current
        yield cells


def extract_inventory(raw):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(raw, "html.parser")
    revision = re.search(r'"wgRevisionId":(\d+)', raw)
    if not revision:
        raise ValueError("A Wikipedia revision ID is required")
    revision_url = f"https://en.wikipedia.org/w/index.php?title=List_of_large_language_models&oldid={revision[1]}"
    rows, counts = [], {}
    for table in soup.select("table.wikitable"):
        heading = table.find_previous("h3")
        year_text = heading.get_text(strip=True) if heading else ""
        if not re.fullmatch(r"20\d\d", year_text):
            continue
        year, count = int(year_text), 0
        for cells in list(expanded_rows(table))[1:]:
            if len(cells) < 3:
                raise ValueError("Incomplete Wikipedia row")
            name = text_without_references(cells[0])
            references = []
            for cell in cells:
                for ref in cell.select('sup.reference a[href^="#"]'):
                    target = soup.find(id=ref["href"][1:])
                    if target:
                        references.extend(link["href"] for link in target.select("a.external[href]")
                                          if link["href"].startswith("http"))
            article = next((a["href"] for a in cells[0].select("a[href]")
                            if not a["href"].startswith("#")), None)
            count += 1
            rows.append({
                "entry_id": f"wiki-{year}-{count:02d}",
                "listed_name": name,
                "listed_year": year,
                "listed_date": text_without_references(cells[1]) or None,
                "listed_developer": text_without_references(cells[2]) or None,
                "source_url": revision_url + f"#{year}",
                "article_url": urljoin(URL, article) if article else None,
                "reference_urls": list(dict.fromkeys(references)),
            })
        counts[year_text] = count
    if not rows or len({r["entry_id"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate Wikipedia inventory")
    return {"source": {"url": URL, "revision_id": int(revision[1]), "revision_url": revision_url,
                       "html_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                       "attribution": "Wikipedia contributors, List of large language models; factual field extraction.",
                       "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                       "row_unit": "One source table row; a row can group multiple sizes or named models.",
                       "rows_by_year": counts}, "entries": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/model-tokenizer-statistics/wikipedia-inventory.json")
    args = parser.parse_args()
    result = extract_inventory(args.html.read_text())
    result["source"]["retrieved_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"entries": len(result["entries"]), "rows_by_year": result["source"]["rows_by_year"]}))


if __name__ == "__main__":
    main()
