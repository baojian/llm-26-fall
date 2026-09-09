"""Write matching editable Excalidraw scenes and self-contained SVG lecture diagrams.

Run from anywhere with: uv run python slides/lecture-01/assets/build-diagrams.py
Coordinates and labels are shared by both representations; Arial is a system font.
These are conceptual schematics, not measured model outputs.
"""
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAVY = '#142e4b'
BLUE = '#20578c'
MUTED = '#566471'


def diagram(name, rows, note, *, note_spans=None):
    elements = []
    svg = []

    def base(kind, x, y, width, height, color=NAVY):
        return {'id': f'{name}-{len(elements)}', 'type': kind, 'x': x, 'y': y,
                'width': width, 'height': height, 'angle': 0,
                'strokeColor': color, 'backgroundColor': 'transparent',
                'fillStyle': 'solid', 'strokeWidth': 2, 'strokeStyle': 'solid',
                'roughness': 0, 'opacity': 100, 'groupIds': [], 'frameId': None,
                'roundness': None, 'seed': 1200 + len(elements), 'version': 1,
                'versionNonce': 7000 + len(elements), 'isDeleted': False,
                'boundElements': None, 'updated': 1, 'link': None, 'locked': False}

    def label(x, y, text, width, size=32, color=NAVY, align='center'):
        lines = text.split('\n')
        el = base('text', x, y, width, len(lines) * size * 1.25, color)
        el.update({'fontSize': size, 'fontFamily': 2, 'text': text,
                   'originalText': text, 'textAlign': align, 'verticalAlign': 'top',
                   'containerId': None, 'autoResize': False, 'lineHeight': 1.25})
        elements.append(el)
        for j, line in enumerate(lines):
            text_x = x + width / 2 if align == 'center' else x
            anchor = 'middle' if align == 'center' else 'start'
            svg.append(f'<text x="{text_x}" y="{y + size + j * size * 1.25}" '
                       f'font-size="{size}" text-anchor="{anchor}" fill="{color}">{escape(line)}</text>')

    def box(x, y, width, height):
        el = base('rectangle', x, y, width, height)
        el['backgroundColor'] = '#eef2f5'
        elements.append(el)
        svg.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
                   'fill="#eef2f5" stroke="#9aafc2" stroke-width="2"/>')

    def arrow(x1, y, x2):
        el = base('arrow', x1, y, x2 - x1, 0, BLUE)
        el.update({'points': [[0, 0], [x2 - x1, 0]], 'lastCommittedPoint': None,
                   'startBinding': None, 'endBinding': None,
                   'startArrowhead': None, 'endArrowhead': 'arrow', 'elbowed': False})
        elements.append(el)
        svg.append(f'<path d="M{x1} {y}H{x2}m-12 -7l12 7l-12 7" '
                   f'fill="none" stroke="{BLUE}" stroke-width="3"/>')

    for y, entries in rows:
        width = (1120 - (len(entries) - 1) * 52) / len(entries)
        for i, (heading, value, detail) in enumerate(entries):
            x = 20 + i * (width + 52)
            label(x, y, heading, width, 30, BLUE)
            box(x, y + 54, width, 114)
            label(x + 6, y + 79, value, width - 12, 32)
            if detail:
                label(x - 4, y + 190, detail, width + 8, 28, MUTED)
            if i < len(entries) - 1:
                arrow(x + width + 7, y + 111, x + width + 45)
    if note_spans:
        assert ''.join(text for text, _, _ in note_spans) == note
        x = (1160 - sum(width for _, width, _ in note_spans)) / 2
        for text, width, color in note_spans:
            label(x, 320, text, width, 30, color, align='left')
            x += width
    else:
        label(20, 320, note, 1120, 30, NAVY)
    scene = {'type': 'excalidraw', 'version': 2, 'source': 'https://excalidraw.com',
             'elements': elements, 'appState': {'viewBackgroundColor': '#fbfbf9', 'gridSize': None},
             'files': {}}
    (ROOT / f'{name}.excalidraw').write_text(json.dumps(scene, indent=2) + '\n')
    (ROOT / f'{name}.svg').write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="410" '
        'viewBox="0 0 1160 410" role="img">\n'
        f'<title>{escape(name.replace("-", " "))}</title>\n'
        '<g font-family="Arial, Helvetica, sans-serif">\n' + '\n'.join(svg) + '\n</g></svg>\n')


def main():
    diagram('text-pipeline', [(35, [
        ('Decode', 'Raw bytes\n→ text', 'Choose an encoding'),
        ('Normalize?', 'Text\n→ text', 'Record changes'),
        ('Split', 'Text\n→ chunks', 'Define boundaries'),
        ('Encode', 'Chunks\n→ token IDs', 'Use a vocabulary'),
    ])], 'Lossless encoding can only recover what earlier stages preserve.')
    diagram('tokenizer-interface', [(35, [
        ('Input text', 'lowest', 'Exact string'),
        ('Pieces', 'low · e · s · t', 'Stored byte strings'),
        ('Token IDs', '257 · 101\n115 · 116', 'Vocabulary indices'),
        ('Model input', 'Embedding\nvectors', 'One lookup per ID'),
    ])], 'Our two-merge toy vocabulary: token 257 stores the bytes for low.', note_spans=[
        # Arial at 30px; shared positions keep the SVG and editable text aligned.
        ('Our two-merge toy vocabulary: token 257 stores the bytes for ', 824, NAVY),
        ('low', 46, BLUE),
        ('.', 8, NAVY),
    ])
    diagram('instruction-tuning', [(35, [
        ('Pretraining', 'Predict text', 'Learn from a corpus'),
        ('Demonstrations', 'Imitate good\nresponses', 'Supervised fine-tuning'),
        ('Preferences', 'Optimize a\nreward signal', 'Comparisons + RL'),
    ])], 'Evaluate the resulting assistant on separate, checkable tasks.')


if __name__ == '__main__':
    main()
