"""Preserve the instructor's PPTX as vector pages in the course Reveal viewer.

Run with uv run --with pymupdf==1.26.7 python slides/lecture-11/import_source.py
    SOURCE.pptx POWERPOINT_EXPORT.pdf
Export the PDF with Microsoft PowerPoint first, including every slide.
"""

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import re
import textwrap
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pymupdf


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def paragraph_text(element):
    return "".join(node.text or "" for node in element.iter()
                   if node.tag in {f"{{{NS['a']}}}t", f"{{{NS['m']}}}t"})


def wrap_image_data(match):
    # Short base64 lines avoid false access-token matches in generated images.
    # Whitespace is ignored by the data-URL decoder; the PNG/JPEG bytes stay exact.
    encoded = "".join(match[2].split())
    wrapped = "\n".join(textwrap.wrap(encoded, 48))
    assert base64.b64decode(wrapped) == base64.b64decode(encoded)
    return match[1] + wrapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pptx", type=Path)
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    (root / "assets/slides").mkdir(parents=True, exist_ok=True)
    (root / "assets/media").mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_filename": args.pptx.name,
        "source_sha256": sha(args.pptx.read_bytes()),
        "pdf_sha256": sha(args.pdf.read_bytes()),
        "renderer": "Microsoft PowerPoint for Mac PDF export; PyMuPDF 1.26.7 SVG",
        "width": 1280, "height": 720,
        "slides": [], "media": [],
    }
    markdown = []
    with ZipFile(args.pptx) as archive, pymupdf.open(args.pdf) as pdf:
        names = sorted((n for n in archive.namelist()
                        if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                       key=lambda n: int(re.search(r"(\d+)\.xml", n)[1]))
        assert len(pdf) == len(names), "Export all source slides in order."
        for index, (name, page) in enumerate(zip(names, pdf), 1):
            assert abs(page.rect.width / page.rect.height - 16 / 9) < 0.001
            xml = ET.fromstring(archive.read(name))
            paragraphs = [paragraph_text(p) for p in xml.findall(".//a:p", NS)]
            paragraphs = [text for text in paragraphs if text.strip()]
            title = next(text for text in paragraphs if not text.strip().isdigit())
            slide_id = "references" if index == len(pdf) else f"source-{index:03d}"
            asset = f"assets/slides/slide-{index:03d}.svg"
            svg = re.sub(r'(data:image/(?:png|jpeg);base64,)([^\"]+)',
                         wrap_image_data, page.get_svg_image(text_as_path=True)).encode()
            (root / asset).write_bytes(svg)
            links = []
            for link in page.get_links():
                if link.get("uri"):
                    rect = link["from"]
                    links.append({"uri": link["uri"], "rect": [
                        round(value * 4 / 3, 5) for value in
                        (rect.x0, rect.y0, rect.width, rect.height)]})
            # PowerPoint's PDF exporter omits hyperlinks attached to pictures.
            if index == 44:
                links.append({"uri": "https://aclanthology.org/K16-1002.pdf", "rect": [
                    round(value / 9525, 5) for value in
                    (8701536, 2738773, 3207841, 1276015)]})
            manifest["slides"].append({
                "id": slide_id, "number": index, "title": title,
                "asset": asset, "sha256": sha(svg),
                "source_xml": name, "source_xml_sha256": sha(archive.read(name)),
                "paragraphs": paragraphs, "links": links,
            })
            transcript = "\n\n".join(html.escape(text) for text in paragraphs)
            markdown.append(
                f'<!-- .slide: id="{slide_id}" class="source-slide" -->\n\n'
                f'<h2 class="source-slide-title">{html.escape(title)}</h2>\n\n'
                f'<img class="source-slide-image" src="{asset}" '
                f'alt="{html.escape(title, quote=True)} — original slide {index}">\n\n'
                f'Note:\n<pre>{transcript}</pre>\n')
        for name in archive.namelist():
            if name.startswith("ppt/media/") and (name.endswith(".gif") or name == "ppt/media/image3.png"):
                data = archive.read(name)
                target = f"assets/media/{Path(name).name}"
                (root / target).write_bytes(data)
                manifest["media"].append({"asset": target, "sha256": sha(data)})
    (root / "slides.md").write_text("\n---\n\n".join(markdown))
    (root / "assets/source-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Preserved {len(manifest['slides'])} slides and {len(manifest['media'])} interactive media assets.")


if __name__ == "__main__":
    main()
