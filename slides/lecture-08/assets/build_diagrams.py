"""Build original, editable Lecture 08 diagrams using the course SVG convention.

Each SVG and its Excalidraw source share positions and text. All examples are
course-authored. Run with uv run python slides/lecture-08/assets/build_diagrams.py.
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

    def edge(self, x1, y, x2):
        self.element("line", x1, y, x2 - x1, 0, BLUE).update(
            points=[[0, 0], [x2 - x1, 0]], lastCommittedPoint=None,
            startBinding=None, endBinding=None,
        )
        self.svg.append(
            f'<path d="M{x1},{y} L{x2},{y}" fill="none" stroke="{BLUE}" stroke-width="3"/>'
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
    drawing = Drawing("data-lineage", "A source record remains traceable through the training pipeline")
    for index, (title, detail, color) in enumerate([
        ("1  Extract", "Source + revision\nDocument ID", BLUE),
        ("2  Curate", "Normalize + filter\nCandidate matches", PURPLE),
        ("3  Split", "Family + source ID\nTrain / dev / test", GREEN),
        ("4  Sample", "Token weights\nProcessed budget", BLUE),
    ]):
        x = 8 + index * 287
        drawing.text(x, 40, title, 270, color=color)
        drawing.box(x, 115, 270, 140, detail, color, size=30)
        if index < 3:
            drawing.arrow(x + 272, 185, x + 284)
    drawing.text(10, 310, "Keep provenance and audit counts at every stage.", 1130)
    drawing.text(10, 355, "External benchmarks keep their published split.", 1130, size=30, color=MUTED)
    drawing.save()

    drawing = Drawing("text-extraction", "Extract content without losing code, headings, or meaning")
    drawing.text(0, 10, "Page text", 530, color=BLUE)
    drawing.text(620, 10, "Intended content", 530, color=GREEN)
    drawing.box(0, 75, 530, 230,
                "HOME   MENU   SUBSCRIBE\nNoa reads at the café.\nReturn the book tomorrow.\nCOOKIE SETTINGS", BLUE, size=30)
    drawing.arrow(542, 188, 606)
    drawing.box(620, 75, 530, 230,
                "Noa reads at the café.\nReturn the book tomorrow.", GREEN, size=30)
    drawing.text(0, 340, "Inspect extraction errors on prose, code, tables, and each language.", 1152, size=30)
    drawing.save()

    drawing = Drawing("minhash-orders", "MinHash keeps the earliest shingle under each shared permutation")
    drawing.text(0, 0, "A = {a, b, c}     B = {b, c, d}     J(A, B) = 2 / 4", 1152)
    for x, label, width in [(20, "Shared order", 440), (480, "Min A", 180),
                            (700, "Min B", 180), (910, "Match?", 220)]:
        drawing.text(x, 65, label, width, color=BLUE)
    for index, (order, a, b) in enumerate([
        ("a  b  c  d", "a", "b"), ("b  a  d  c", "b", "b"),
        ("c  d  a  b", "c", "c"), ("d  c  b  a", "c", "d"),
    ]):
        y = 120 + index * 55
        for x, label, width in [(20, order, 440), (480, a, 180), (700, b, 180),
                                (910, "Yes" if a == b else "No", 220)]:
            drawing.text(x, y, label, width, color=GREEN if a == b else MUTED)
    drawing.text(0, 355, "Two matching positions out of four: estimated similarity = 0.5", 1152, size=30)
    drawing.save()

    drawing = Drawing("duplicate-families", "Connected match families stay within a split without deleting every record")
    drawing.text(0, 0, "Match threshold: J ≥ 0.75", 1152)
    for x, name, values in [(70, "A", "{1, 2, 3, 4}"), (445, "B", "{1, 2, 3, 4, 5}"),
                            (820, "C", "{2, 3, 4, 5}")]:
        drawing.box(x, 100, 260, 130, name + "\n" + values, BLUE, size=30)
    for x1, x2 in [(330, 445), (705, 820)]:
        drawing.edge(x1 + 5, 165, x2 - 5)
        drawing.text(x1, 55, "J = 0.8", x2-x1, size=28, color=GREEN)
    drawing.text(0, 260, "J(A, C) = 0.6: there is no direct A–C match.", 1152, size=30, color=MUTED)
    drawing.box(70, 330, 1010, 65, "One connected split group; inspect retention within the group.", GREEN, size=30)
    drawing.save()

    drawing = Drawing("mixture-exposure", "Changing source probabilities changes expected repetition")
    drawing.text(0, 0, "Available tokens: English 9M; Chinese 1M. Training budget: 60M.", 1152, size=30)
    for index, (label, p, detail) in enumerate([
        ("Natural weights", .9, "54M / 6M processed; 6 / 6 expected passes"),
        ("Square-root weights", .75, "45M / 15M processed; 5 / 15 expected passes"),
    ]):
        y = 70 + 155 * index
        drawing.text(20, y, label, 1100, size=30, align="left")
        drawing.rect(20, y + 52, 1100 * p, 44, BLUE, BLUE)
        drawing.rect(20 + 1100 * p, y + 52, 1100 * (1-p), 44, ORANGE, ORANGE)
        drawing.text(20, y + 53, f"{p:.0%}", 1100 * p, size=30, color="#ffffff")
        drawing.text(20 + 1100 * p, y + 53, f"{1-p:.0%}", 1100 * (1-p), size=30, color="#ffffff")
        drawing.text(20, y + 101, detail, 1100, size=30, align="left")
    drawing.save()


if __name__ == "__main__":
    main()
