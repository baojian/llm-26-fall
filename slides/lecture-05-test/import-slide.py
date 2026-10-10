"""Extract a source slide's artwork and click sequence without changing it.

Run with the instructor's original PPTX as the only argument. No third-party
Python packages are needed. The browser uses the course template around the
original diagram; source-slideNN.pptx retains the original slide and animations.
"""

import argparse
import base64
import hashlib
import json
import posixpath
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
}
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
UNIT = 9525  # EMU per CSS pixel at the source's 1280 x 720 aspect ratio.
LABELS = {24: [
    "Compare the query embedding with all four embeddings",
    "Show the four dot-product scores",
    "Normalize the scores with softmax",
    "Show the four attention weights",
    "Bring in the embeddings to be combined",
    "Multiply each embedding by its weight",
    "Sum the weighted embeddings to obtain the third output",
    "Repeat for the other three outputs",
], 29: [
    "Introduce the trainable query matrix",
    "Introduce the trainable key matrix",
    "Introduce the trainable value matrix",
    "Emphasize that all three projection matrices are learned",
    "Write the query, key, and value projections in matrix form",
]}


def sub(parent, tag, **attrs):
    return ET.SubElement(parent, f"{{{SVG}}}{tag}", {k: str(v) for k, v in attrs.items()})


def box(shape):
    transform = shape.find("p:spPr/a:xfrm", NS)
    if transform is None:
        transform = shape.find("p:xfrm", NS)
    offset, extent = transform.find("a:off", NS), transform.find("a:ext", NS)
    return tuple(int(v) / UNIT for v in (offset.get("x"), offset.get("y"), extent.get("cx"), extent.get("cy")))


def source_color(element, default):
    if element is None:
        return default
    srgb = element.find("a:solidFill/a:srgbClr", NS)
    if srgb is not None:
        return "#" + srgb.get("val")
    scheme = element.find("a:solidFill/a:schemeClr", NS)
    if scheme is not None:
        if scheme.find("a:lumOff", NS) is not None:
            return "#808080"
        return {"tx1": "#000000", "accent1": "#4472C4"}.get(scheme.get("val"), default)
    return default


def relationship_path(part):
    directory, name = posixpath.split(part)
    return posixpath.join(directory, "_rels", name + ".rels")


def extract_native_slide(archive, chosen_id, output):
    """Keep the selected slide and its dependency closure, including its timing."""
    original = archive.read("ppt/presentation.xml").decode()
    slide_tag = re.search(r'<p:sldId\b[^>]*r:id="' + re.escape(chosen_id) + r'"[^>]*/>', original).group()
    presentation_bytes = re.sub(r"<p:sldIdLst>.*?</p:sldIdLst>", "<p:sldIdLst>" + slide_tag + "</p:sldIdLst>", original, flags=re.S).encode()
    relpart = "ppt/_rels/presentation.xml.rels"
    rels = ET.fromstring(archive.read(relpart))
    for rel in list(rels):
        if rel.get("Type").endswith("/slide") and rel.get("Id") != chosen_id:
            rels.remove(rel)
    replacements = {"ppt/presentation.xml": presentation_bytes, relpart: ET.tostring(rels, encoding="utf-8", xml_declaration=True)}
    # Do not retain the full deck's thumbnail or personal document properties.
    root_rels = ET.fromstring(archive.read("_rels/.rels"))
    for rel in list(root_rels):
        if not rel.get("Type").endswith("/officeDocument"):
            root_rels.remove(rel)
    replacements["_rels/.rels"] = ET.tostring(root_rels, encoding="utf-8", xml_declaration=True)
    keep, pending = {"[Content_Types].xml", "_rels/.rels"}, [""]
    while pending:
        part = pending.pop()
        relname = relationship_path(part) if part else "_rels/.rels"
        if relname not in archive.namelist():
            continue
        keep.add(relname)
        relroot = ET.fromstring(replacements.get(relname, archive.read(relname)))
        for rel in relroot:
            if rel.get("TargetMode") == "External":
                continue
            target = posixpath.normpath(posixpath.join(posixpath.dirname(part), rel.get("Target")))
            if target not in keep:
                keep.add(target)
                pending.append(target)
    types = ET.fromstring(archive.read("[Content_Types].xml"))
    for item in list(types):
        if item.tag.endswith("Override") and item.get("PartName").lstrip("/") not in keep:
            types.remove(item)
    replacements["[Content_Types].xml"] = ET.tostring(types, encoding="utf-8", xml_declaration=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as out:
        for name in sorted(keep):
            out.writestr(name, replacements.get(name, archive.read(name)))


def main(source, number=24):
    destination = Path(__file__).resolve().parent / "assets"
    destination.mkdir(exist_ok=True)
    with ZipFile(source) as archive:
        presentation = ET.fromstring(archive.read("ppt/presentation.xml"))
        chosen = presentation.find("p:sldIdLst", NS)[number - 1]
        chosen_id = chosen.get(f"{{{NS['r']}}}id")
        presentation_rels = ET.fromstring(archive.read("ppt/_rels/presentation.xml.rels"))
        slide_part = "ppt/" + next(r.get("Target") for r in presentation_rels if r.get("Id") == chosen_id)
        raw_slide = archive.read(slide_part)
        slide = ET.fromstring(raw_slide)
        rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(archive.read(relationship_path(slide_part)))}
        steps = []
        for timing in slide.findall(".//p:cTn", NS):
            if timing.get("presetClass") is None:
                continue
            assert timing.get("presetClass") == "entr" and timing.get("presetID") == "1", "Only the source's Appear effects are supported."
            assert timing.get("nodeType") in ("clickEffect", "withEffect")
            if timing.get("nodeType") == "clickEffect":
                steps.append([])
            steps[-1].append(timing.find(".//p:spTgt", NS).get("spid"))
        labels = LABELS[number]
        assert len(steps) == len(labels)
        suffix = "" if number == 24 else f"-{number}"
        prefix = f"source-{number}"
        stage_for = {sid: i for i, step in enumerate(steps) for sid in step}
        root = ET.Element(f"{{{SVG}}}svg", {
            "viewBox": "0 52 1280 668", "width": "1152", "height": "552",
            "role": "img", "aria-labelledby": f"{prefix}-title {prefix}-description",
        })
        sub(root, "title", id=f"{prefix}-title").text = f"Self-attention, source slide {number}"
        sub(root, "desc", id=f"{prefix}-description").text = "; ".join(labels)
        defs = sub(root, "defs")
        arrow_colors = {"#000000": "black", "#808080": "gray", "#00B050": "green", "#C00000": "red", "#7030A0": "purple"}
        for color, name in arrow_colors.items():
            marker = sub(defs, "marker", id=f"{prefix}-arrow-{name}", viewBox="0 0 8 8", refX="7", refY="4", markerWidth="5", markerHeight="5", orient="auto-start-reverse", markerUnits="strokeWidth")
            sub(marker, "path", d="M 1 1 L 7 4 L 1 7", fill="none", stroke=color, **{"stroke-width": "1.5", "stroke-linecap": "round", "stroke-linejoin": "round"})
        rendered = []
        for shape in slide.find("p:cSld/p:spTree", NS):
            nv = shape.find(".//p:cNvPr", NS)
            if nv is None or nv.get("id") in ({"1", "2", "4", "6"} if number == 24 else {"1", "2", "4", "5"}):
                continue  # The title, slide number and attribution use the course shell.
            sid = nv.get("id")
            if sid in stage_for:
                outer = sub(root, "g", **{"class": "fragment custom", "data-fragment-index": stage_for[sid], "data-source-shape": sid})
                group = sub(outer, "g", **{"data-appear": "", "visibility": "visible"})
                rendered.append(sid)
            else:
                group = sub(root, "g", **{"data-source-static": sid})
            if shape.tag.endswith("AlternateContent"):
                fallback = shape.find("mc:Fallback/p:sp", NS)
                x, y, w, h = box(fallback)
                image_fill = fallback.find("p:spPr/a:blipFill", NS)
                rid = image_fill.find("a:blip", NS).get(f"{{{NS['r']}}}embed")
                media = posixpath.normpath(posixpath.join("ppt/slides", rels[rid]))
                fill = image_fill.find("a:stretch/a:fillRect", NS)
                left, top, right, bottom = (int(fill.get(k, "0")) / 100000 for k in ("l", "t", "r", "b"))
                # PowerPoint's fallback artwork includes transparent margins. Keep
                # the source fill rectangle, including its negative edge offsets.
                sub(group, "image", x=x + left*w, y=y + top*h, width=w*(1-left-right), height=h*(1-top-bottom), preserveAspectRatio="none", href="data:image/png;base64," + base64.b64encode(archive.read(media)).decode())
                continue
            x, y, w, h = box(shape)
            if shape.tag.endswith("graphicFrame"):
                table = shape.find(".//a:tbl", NS)
                columns = table.findall("a:tblGrid/a:gridCol", NS)
                row_y = y
                for row in table.findall("a:tr", NS):
                    cursor = x
                    row_height = int(row.get("h")) / UNIT
                    for column in columns:
                        width = int(column.get("w")) / UNIT
                        sub(group, "rect", x=cursor, y=row_y, width=width, height=row_height, fill="#4472C4", stroke="#FFFFFF", **{"stroke-width": 1})
                        cursor += width
                    row_y += row_height
                continue
            properties = shape.find("p:spPr", NS)
            if shape.tag.endswith("cxnSp"):
                transform = properties.find("a:xfrm", NS)
                x1, x2 = (x+w, x) if transform.get("flipH") == "1" else (x, x+w)
                y1, y2 = (y+h, y) if transform.get("flipV") == "1" else (y, y+h)
                line = properties.find("a:ln", NS)
                color = source_color(line, "#000000")
                attrs = {"stroke-width": int(line.get("w", "12700"))/UNIT}
                dash = line.find("a:prstDash", NS)
                if dash is not None and dash.get("val") == "dash":
                    attrs["stroke-dasharray"] = f"{4*attrs['stroke-width']} {3*attrs['stroke-width']}"
                for tag, end in (("headEnd", "start"), ("tailEnd", "end")):
                    arrow = line.find("a:" + tag, NS)
                    if arrow is not None and arrow.get("type") == "arrow":
                        attrs["marker-"+end] = f"url(#{prefix}-arrow-{arrow_colors[color]})"
                sub(group, "line", x1=x1, y1=y1, x2=x2, y2=y2, stroke=color, **attrs)
                continue
            custom = properties.find("a:custGeom", NS)
            if custom is not None:
                transform = properties.find("a:xfrm", NS)
                rotation = int(transform.get("rot", "0")) / 60000
                flip_x = -1 if transform.get("flipH") == "1" else 1
                flip_y = -1 if transform.get("flipV") == "1" else 1
                custom_group = sub(group, "g", transform=f"translate({x+w/2} {y+h/2}) rotate({rotation}) scale({flip_x} {flip_y}) translate({-w/2} {-h/2})")
                outline = properties.find("a:ln", NS)
                color = source_color(outline, "#000000")
                for path in custom.findall("a:pathLst/a:path", NS):
                    scale_x, scale_y = w / int(path.get("w")), h / int(path.get("h"))
                    commands = []
                    for command in path:
                        tag = command.tag.rsplit("}", 1)[-1]
                        assert tag in ("moveTo", "lnTo", "cubicBezTo", "close"), f"Unsupported custom path: {tag}"
                        points = " ".join(f"{int(point.get('x'))*scale_x},{int(point.get('y'))*scale_y}" for point in command)
                        commands.append({"moveTo": "M", "lnTo": "L", "cubicBezTo": "C", "close": "Z"}[tag] + points)
                    attrs = {"stroke-width": int(outline.get("w", "12700")) / UNIT}
                    for tag, end in (("headEnd", "start"), ("tailEnd", "end")):
                        arrow = outline.find("a:" + tag, NS)
                        if arrow is not None and arrow.get("type") == "arrow":
                            attrs["marker-" + end] = f"url(#{prefix}-arrow-{arrow_colors[color]})"
                    sub(custom_group, "path", d=" ".join(commands), fill="none", stroke=color, **attrs)
                continue
            geometry = properties.find("a:prstGeom", NS).get("prst")
            color = source_color(properties.find("a:ln", NS), "#325490")
            if geometry == "diamond":
                sub(group, "polygon", points=f"{x+w/2},{y} {x+w},{y+h/2} {x+w/2},{y+h} {x},{y+h/2}", fill="none", stroke=color, **{"stroke-width": 4/3})
            else:
                assert geometry == "roundRect"
                radius = min(w, h) * 0.0442
                sub(group, "rect", x=x, y=y, width=w, height=h, rx=radius, fill="none", stroke=color, **{"stroke-width": 4/3})
                text = "".join(t.text or "" for t in shape.findall(".//a:t", NS))
                # Use the shared course font for the three native box labels.
                at_top = shape.find("p:txBody/a:bodyPr", NS).get("anchor") == "t"
                sub(group, "text", x=x+9.6 if at_top else x+w/2, y=y+30 if at_top else y+h/2, fill="#00B050", **{"font-size": "32", "text-anchor": "start" if at_top else "middle", "dominant-baseline": "alphabetic" if at_top else "central"}).text = text
        assert set(rendered) == set(stage_for), "Every animated source object must be preserved."
        ET.indent(root)
        (destination / f"self-attention{suffix}.svg").write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))
        manifest = {
            "source": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "slide": number, "slide_part": slide_part, "slide_sha256": hashlib.sha256(raw_slide).hexdigest(),
            "effect": "Appear", "steps": [{"click": i+1, "label": labels[i], "shape_ids": targets} for i, targets in enumerate(steps)],
            "attribution": "https://www.youtube.com/watch?v=tIvKXrEDMhk",
        }
        (destination / f"animation{suffix}.json").write_text(json.dumps(manifest, indent=2) + "\n")
        extract_native_slide(archive, chosen_id, destination / f"source-slide{number}.pptx")
        print(f"Extracted slide {number}: {len(rendered)} animated objects, {len(steps)} clicks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--slide", type=int, choices=LABELS, default=24)
    args = parser.parse_args()
    main(args.source, args.slide)
