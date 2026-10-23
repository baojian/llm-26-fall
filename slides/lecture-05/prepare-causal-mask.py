"""Build matching editable and SVG views of the Noa prefix's causal mask."""

from html import escape
import json
from pathlib import Path


ASSETS = Path(__file__).resolve().parent / "assets"
TOKENS = ["Noa", "can", "be", "annoying", "but", "she"]
TARGETS = ["can", "be", "annoying", "but", "she", "is"]
INK, BLUE, GREEN, MUTED = "#142e4b", "#20578c", "#147d73", "#64748b"
elements = []
artwork = []


def base(kind, x, y, width, height, stroke=INK, fill="transparent"):
    index = len(elements) + 1
    element = {
        "id": f"noa-causal-{index}", "type": kind, "x": x, "y": y,
        "width": width, "height": height, "angle": 0, "strokeColor": stroke,
        "backgroundColor": fill, "fillStyle": "solid", "strokeWidth": 2,
        "strokeStyle": "solid", "roughness": 0, "opacity": 100,
        "groupIds": [], "frameId": None, "roundness": None,
        "seed": index, "version": 1, "versionNonce": index,
        "isDeleted": False, "boundElements": None, "updated": 0,
        "link": None, "locked": False,
    }
    elements.append(element)
    return element


def text(x, y, label, color=INK, width=180, anchor="start"):
    left = x - width / 2 if anchor == "middle" else x
    element = base("text", left, y - 29, width, 36, color)
    element.update({"text": label, "originalText": label, "fontSize": 30,
                    "fontFamily": 2, "textAlign": "center" if anchor == "middle" else "left",
                    "verticalAlign": "top", "containerId": None,
                    "lineHeight": 1.2, "autoResize": False})
    artwork.append(f'<text x="{x}" y="{y}" font-size="30" fill="{color}" '
                   f'text-anchor="{anchor}">{escape(label)}</text>')


def rectangle(x, y, width, height, fill, stroke):
    base("rectangle", x, y, width, height, stroke, fill)
    artwork.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
                   f'rx="5" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')


text(10, 31, "Input query")
text(220, 31, "Allowed input keys", width=340)
text(957, 31, "Next target", GREEN, width=190)
for key in range(6):
    text(254 + key * 110, 76, str(key), width=80, anchor="middle")
for query, (token, target) in enumerate(zip(TOKENS, TARGETS)):
    y = 99 + 51 * query
    text(10, y + 29, f"{query}  {token}", width=195)
    for key in range(6):
        allowed = key <= query
        rectangle(213 + key * 110, y, 82, 40,
                  "#e8f0f7" if allowed else "#f1f5f9", BLUE if allowed else "#d5dee7")
        text(254 + key * 110, y + 29, "yes" if allowed else "—",
             BLUE if allowed else MUTED, width=82, anchor="middle")
    arrow = base("arrow", 890, y + 20, 47, 0, GREEN)
    arrow.update({"points": [[0, 0], [47, 0]], "lastCommittedPoint": None,
                  "startBinding": None, "endBinding": None,
                  "startArrowhead": None, "endArrowhead": "arrow"})
    artwork.append(f'<path d="M 890 {y + 20} H 937" fill="none" stroke="{GREEN}" '
                   'stroke-width="2.5" marker-end="url(#noa-causal-arrow)"/>')
    text(965, y + 29, target, GREEN, width=180)

svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="1152" height="410" '
       'viewBox="0 0 1152 410" role="img" aria-labelledby="noa-mask-title noa-mask-description" '
       'font-family="Arial, Helvetica, sans-serif">'
       '<title id="noa-mask-title">Causal visibility in the Noa prefix</title>'
       '<desc id="noa-mask-description">Six input rows Noa, can, be, annoying, but, she. '
       'Each row can use itself and all earlier keys. The next targets are can, be, '
       'annoying, but, she, is. The complete training sequence has ten input positions.</desc>'
       '<defs><marker id="noa-causal-arrow" viewBox="0 0 10 10" refX="8" refY="5" '
       'markerWidth="5" markerHeight="5" orient="auto">'
       f'<path d="M 1 1 L 8 5 L 1 9" fill="none" stroke="{GREEN}" stroke-width="1.8"/>'
       '</marker></defs>' + ''.join(artwork) + '</svg>\n')
scene = {"type": "excalidraw", "version": 2,
         "source": "https://github.com/baojian/llm-26-fall",
         "elements": elements, "appState": {"viewBackgroundColor": "#fbfbf9"}, "files": {}}
(ASSETS / "causal-mask.svg").write_text(svg)
(ASSETS / "causal-mask.excalidraw").write_text(json.dumps(scene, indent=2) + "\n")
print("Built the six-position Noa causal mask in SVG and editable Excalidraw formats.")
