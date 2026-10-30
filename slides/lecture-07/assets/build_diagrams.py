"""Build original Lecture 07 diagrams with matching SVG/Excalidraw geometry.

Run with: uv run python slides/lecture-07/assets/build_diagrams.py
The scene/export convention follows Lecture 06's asset builder.
"""

from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAVY, BLUE, GREEN, PURPLE = "#142e4b", "#20578c", "#28673f", "#634185"
MUTED, PALE, ORANGE = "#566471", "#eef2f5", "#ba6419"


class Drawing:
    def __init__(self, name, title):
        self.name, self.title = name, title
        self.elements, self.svg = [], []

    def element(self, kind, x, y, width, height, color, fill="transparent"):
        index = len(self.elements)
        element = dict(
            id=f"{self.name}-{index}", type=kind, x=x, y=y,
            width=width, height=height, angle=0, strokeColor=color,
            backgroundColor=fill, fillStyle="solid", strokeWidth=2,
            strokeStyle="solid", roughness=0, opacity=100, groupIds=[],
            frameId=None, roundness=None, seed=700 + index,
            version=1, versionNonce=900 + index, isDeleted=False,
            boundElements=None, updated=1, link=None, locked=False,
        )
        self.elements.append(element)
        return element

    def text(self, x, y, value, width, size=32, color=NAVY, align="center"):
        lines = value.split("\n")
        self.element("text", x, y, width, len(lines) * size * 1.25, color).update(
            fontSize=size, fontFamily=2, text=value, originalText=value,
            textAlign=align, verticalAlign="top", containerId=None,
            autoResize=False, lineHeight=1.25,
        )
        anchor = "middle" if align == "center" else "start"
        xx = x + width / 2 if align == "center" else x
        for index, line in enumerate(lines):
            self.svg.append(
                f'<text x="{xx}" y="{y + size + index * size * 1.25}" '
                f'font-size="{size}" text-anchor="{anchor}" fill="{color}">'
                f'{escape(line)}</text>'
            )

    def rect(self, x, y, width, height, color=BLUE, fill=PALE):
        self.element("rectangle", x, y, width, height, color, fill)
        self.svg.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'fill="{fill}" stroke="{color}" stroke-width="2"/>'
        )

    def box(self, x, y, width, height, label, color=BLUE, size=32):
        self.rect(x, y, width, height, color)
        label_height = len(label.split("\n")) * size * 1.25
        self.text(x + 8, y + (height - label_height) / 2 - 2,
                  label, width - 16, size, color)

    def arrow(self, x1, y, x2):
        self.element("arrow", x1, y, x2 - x1, 0, BLUE).update(
            points=[[0, 0], [x2 - x1, 0]], lastCommittedPoint=None,
            startBinding=None, endBinding=None, startArrowhead=None,
            endArrowhead="arrow", elbowed=False,
        )
        self.svg.append(
            f'<path d="M{x1},{y} L{x2},{y}" fill="none" stroke="{BLUE}" '
            'stroke-width="3" marker-end="url(#arrow)"/>'
        )

    def save(self):
        scene = dict(
            type="excalidraw", version=2, source="https://excalidraw.com",
            elements=self.elements,
            appState={"viewBackgroundColor": "#fbfbf9", "gridSize": None},
            files={},
        )
        (ROOT / f"{self.name}.excalidraw").write_text(
            json.dumps(scene, indent=2) + "\n", encoding="utf-8",
        )
        (ROOT / f"{self.name}.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="1152" height="410" '
            f'viewBox="0 0 1152 410" role="img"><title>{escape(self.title)}</title>'
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto">'
            f'<path d="M0 0L10 5L0 10" fill="{BLUE}"/></marker></defs>'
            '<g font-family="Arial, Helvetica, sans-serif">'
            + "".join(self.svg) + "</g></svg>\n", encoding="utf-8",
        )


def main():
    drawing = Drawing("training-matmul", "One forward and two backward matrix products")
    drawing.text(0, 0, "m token positions · input width d · output width h", 1152)
    products = [
        ("Forward", "X\nm × d", "W\nd × h", "Y\nm × h", BLUE),
        ("Input\ngradient", "G\nm × h", "Wᵀ\nh × d", "∂L/∂X\nm × d", GREEN),
        ("Weight\ngradient", "Xᵀ\nd × m", "G\nm × h", "∂L/∂W\nd × h", PURPLE),
    ]
    for index, (label, left, right, output, color) in enumerate(products):
        y = 62 + index * 116
        drawing.text(0, y + (25 if index == 0 else 5), label, 180, color=color)
        drawing.box(205, y, 175, 100, left, color)
        drawing.text(390, y + 27, "×", 40, 36, color)
        drawing.box(440, y, 175, 100, right, color)
        drawing.arrow(628, y + 50, 670)
        drawing.box(685, y, 180, 100, output, color)
        drawing.text(885, y + 12, "≈ 2mdh\nFLOPs", 250, color=color)
    drawing.save()

    drawing = Drawing("warmup-fraction", "Equal warmup steps occupy different fractions of each run")
    bar_x, bar_width = 35, 1080
    for index, steps in enumerate([1000, 10000]):
        y = 25 + index * 150
        fraction = 200 / steps
        drawing.text(35, y, f"{steps:,} updates · 200 warmup · {fraction:.0%} in warmup",
                     1080, align="left")
        drawing.rect(bar_x, y + 60, bar_width, 44, MUTED)
        drawing.rect(bar_x, y + 60, bar_width * fraction, 44, ORANGE, ORANGE)
    drawing.rect(35, 340, 30, 30, ORANGE, ORANGE)
    drawing.text(80, 332, "Warmup", 200, align="left", color=ORANGE)
    drawing.rect(335, 340, 30, 30, MUTED)
    drawing.text(380, 332, "Remaining updates", 385, align="left", color=MUTED)
    drawing.text(780, 332, "Each bar = one full run", 350, size=30)
    drawing.save()

    drawing = Drawing("prediction-check", "Fit on small runs, freeze, predict, then check a reserved run")
    labels = [
        ("1  Small runs", "Fit the curve\nTune the recipe", BLUE),
        ("2  Freeze", "Recipe + curve\nError tolerance", PURPLE),
        ("3  Predict", "Write the loss\nat a larger budget", BLUE),
        ("4  Check", "Reveal the run\nCompare the error", GREEN),
    ]
    for index, (heading, label, color) in enumerate(labels):
        x = 6 + index * 290
        drawing.text(x, 55, heading, 270, color=color)
        drawing.box(x, 120, 270, 140, label, color, size=30)
        if index < 3:
            drawing.arrow(x + 273, 190, x + 287)
    drawing.text(10, 318, "The reserved larger run stays outside fitting and recipe selection.",
                 1130, color=NAVY)
    drawing.save()


if __name__ == "__main__":
    main()
