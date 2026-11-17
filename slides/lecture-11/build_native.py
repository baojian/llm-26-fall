"""Build editable Lecture 11 HTML from the original DrawingML and OMML.

uv run --with pillow==12.1.0 python slides/lecture-11/build_native.py SOURCE.pptx
All generated text remains directly editable in slides.md. Rebuilding is optional.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path, PurePosixPath
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile
from PIL import Image

from omml_to_latex import to_latex

NS = {
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'mc': 'http://schemas.openxmlformats.org/markup-compatibility/2006',
}
ROOT = Path(__file__).resolve().parent
EMU = 9525


def tag(el):
    return el.tag.rsplit('}', 1)[-1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def find(el, query):
    return el.find(query, NS) if el is not None else None


def children(tree):
    for child in tree:
        if tag(child) == 'AlternateContent':
            yield from children(find(child, 'mc:Choice'))
        else:
            yield child


def ph(shape):
    return find(shape, 'p:nvSpPr/p:nvPr/p:ph')


def placeholder(shape, tree):
    wanted = ph(shape)
    if wanted is None:
        return None
    candidates = [s for s in children(find(tree, 'p:cSld/p:spTree')) if ph(s) is not None]
    for candidate in candidates:
        if ph(candidate).get('type', 'body') == wanted.get('type', 'body') and ph(candidate).get('idx', '0') == wanted.get('idx', '0'):
            return candidate
    return next((s for s in candidates if ph(s).get('type', 'body') == wanted.get('type', 'body')), None)


def color(element, fallback='#000000'):
    if element is None:
        return fallback
    direct = find(element, 'a:srgbClr')
    if direct is not None:
        return '#' + direct.get('val')
    scheme = find(element, 'a:schemeClr')
    result = {'tx1': '#000000', 'bg1': '#ffffff', 'tx2': '#1f497d',
              'hlink': '#0563c1', 'accent1': '#4472c4'}.get(scheme.get('val'), fallback) if scheme is not None else fallback
    tint = find(scheme, 'a:tint')
    if tint is not None and result.startswith('#'):
        keep = float(tint.get('val')) / 100000
        channels = [int(result[i:i+2],16)/255 for i in [1,3,5]]
        channels = [round(255 * (keep * value**2.2 + 1-keep)**(1/2.2)) for value in channels]
        result = '#' + ''.join(f'{c:02x}' for c in channels)
    return result


def merge_properties(nodes):
    attrs, elements = {}, {}
    for node in nodes:
        if node is not None:
            attrs.update(node.attrib)
            for child in node:
                elements[tag(child)] = child
                if tag(child) in ['buNone', 'buChar']:
                    elements.pop('buChar' if tag(child) == 'buNone' else 'buNone', None)
    return attrs, elements


def run_style(nodes, scale=1):
    attrs, elements = merge_properties(nodes)
    size = float(attrs.get('sz', '1800')) / 100 * 4 / 3 * scale
    face = elements.get('latin')
    family = face.get('typeface', 'Tahoma') if face is not None else 'Tahoma'
    if family.startswith('+'):
        family = 'Tahoma'
    # These source overrides are resolved to these families by PowerPoint for Mac.
    if family == '-apple-system':
        family = 'Calibri'
    font = {'Tahoma': 'Tahoma, Verdana, Arial, sans-serif',
            'Calibri': 'Calibri, Carlito, Arial, sans-serif',
            'Roboto': 'Roboto, Arial, sans-serif'}.get(family, 'Tahoma, Arial, sans-serif')
    rules = {'font-size': f'{size:.4f}px', 'font-family': font,
             'font-weight': '700' if attrs.get('b') == '1' else '400',
             'font-style': 'italic' if attrs.get('i') == '1' else 'normal',
             'color': color(elements.get('solidFill'))}
    if attrs.get('u', 'none') != 'none':
        rules['text-decoration'] = 'underline'
    if attrs.get('baseline'):
        rules['vertical-align'] = f"{float(attrs['baseline']) / 1000}%"
    return rules, size


class Builder:
    def __init__(self, archive):
        self.archive = archive
        self.css = []
        self.styles = {}

    def style(self, rules):
        content = '; '.join(f'{k}: {v}' for k, v in rules.items())
        if content not in self.styles:
            name = f'n{len(self.styles) + 1:03d}'
            self.styles[content] = name
            self.css.append(f'.reveal .native-slide .{name} {{ {content}; }}')
        return self.styles[content]

    def rect(self, shape, chain):
        xfrm = next((find(s, 'p:spPr/a:xfrm') for s in reversed(chain) if find(s, 'p:spPr/a:xfrm') is not None), None)
        if xfrm is None:
            raise ValueError('Missing shape geometry')
        off, ext = find(xfrm, 'a:off'), find(xfrm, 'a:ext')
        return [float(off.get('x')) / EMU, float(off.get('y')) / EMU,
                float(ext.get('cx')) / EMU, float(ext.get('cy')) / EMU]

    def geometry(self, rect, **extra):
        x, y, w, h = rect
        return self.style({'left': f'{x:.4f}px', 'top': f'{y:.4f}px',
                           'width': f'{w:.4f}px', 'height': f'{h:.4f}px', **extra})

    def rels(self, name):
        directory, filename = name.rsplit('/', 1)
        root = ET.fromstring(self.archive.read(f'{directory}/_rels/{filename}.rels'))
        return {r.get('Id'): r.attrib for r in root}

    def asset(self, media_name):
        target = f'assets/media/{PurePosixPath(media_name).name}'
        data = self.archive.read(media_name)
        (ROOT / target).write_bytes(data)
        return target, sha(data)

    def picture(self, shape, number, rels):
        rect = self.rect(shape, [shape])
        rid = find(shape, 'p:blipFill/a:blip').get(f"{{{NS['r']}}}embed")
        media_name = 'ppt/' + rels[rid]['Target'].removeprefix('../')
        target, digest = self.asset(media_name)
        identifier = find(shape, 'p:nvPicPr/p:cNvPr').get('id')
        boxid = f's{number:03d}-pic{identifier}'
        crop = find(shape, 'p:blipFill/a:srcRect')
        l, t, r, b = [float(crop.get(k, 0)) / 100000 if crop is not None else 0 for k in ['l','t','r','b']]
        rules = {'width': f'{100 / (1-l-r):.6f}%', 'height': f'{100 / (1-t-b):.6f}%',
                 'left': f'{-100*l/(1-l-r):.6f}%', 'top': f'{-100*t/(1-t-b):.6f}%'}
        imgclass = self.style(rules)
        title = self.title
        extra = ' fragment' if number == 6 and identifier == '6' else ''
        anim = ' native-gif' if target.endswith('.gif') else ''
        poster = target.replace('.gif', '-poster.png') if anim else None
        if poster:
            with Image.open(ROOT/target) as image:
                image.seek(0)
                image.convert('RGBA').save(ROOT/poster)
        attr = f' data-gif-source="{target}" data-poster="{poster}"' if anim else ''
        source = f'src="{poster}"' if anim else f'data-src="{target}"'
        markup = (f'<div id="{boxid}" class="native-figure {self.geometry(rect)}{extra}">'
                  f'<img class="{imgclass}{anim}" {source}{attr} '
                  f'alt="{html.escape(title, quote=True)} — original illustration {identifier}"></div>')
        link = find(shape, 'p:nvPicPr/p:cNvPr/a:hlinkClick')
        if link is not None:
            uri = rels[link.get(f"{{{NS['r']}}}id")]['Target']
            markup = markup.replace('</div>', f'<a class="native-picture-link" href="{html.escape(uri, quote=True)}" aria-label="{html.escape(uri, quote=True)}"></a></div>')
        return markup, {'asset': target, 'sha256': digest, 'id': boxid, 'rect': rect, 'crop': [l,t,r,b]}

    def text(self, shape, chain, master, number, rels):
        textbox = find(shape, 'p:txBody')
        paragraphs = textbox.findall('a:p', NS)
        if not any(''.join(p.itertext()).strip() for p in paragraphs):
            return None
        identifier = find(shape, 'p:nvSpPr/p:cNvPr').get('id')
        boxid = f's{number:03d}-text{identifier}'
        rect = self.rect(shape, chain)
        bodyattrs, bodyels = merge_properties([find(s, 'p:txBody/a:bodyPr') for s in chain])
        scale = float(bodyels.get('normAutofit').get('fontScale', '100000')) / 100000 if 'normAutofit' in bodyels else 1
        kind = ph(shape).get('type', 'body') if ph(shape) is not None else 'other'
        stylekind = 'titleStyle' if kind in ['title','ctrTitle'] else 'bodyStyle' if kind == 'body' else 'otherStyle'
        inset = {side: float(bodyattrs.get(key, default)) / EMU for side,key,default in
                 [('left','lIns',91440),('right','rIns',91440),('top','tIns',45720),('bottom','bIns',45720)]}
        anchor = {'ctr': 'center', 'b': 'flex-end'}.get(bodyattrs.get('anchor'), 'flex-start')
        geometry = self.geometry(rect, **{f'padding-{s}': f'{v:.4f}px' for s,v in inset.items()}, **{'justify-content': anchor})
        out, texts, originals, sizes = [], [], [], set()
        for index, paragraph in enumerate(paragraphs):
            original = ''.join(x.text or '' for x in paragraph.iter() if tag(x) in ['t','br'])
            if not original.strip():
                continue
            level = int(find(paragraph, 'a:pPr').get('lvl', '0')) if find(paragraph, 'a:pPr') is not None else 0
            propnodes = [find(master, f'p:txStyles/p:{stylekind}/a:lvl{level+1}pPr')]
            propnodes += [find(s, f'p:txBody/a:lstStyle/a:lvl{level+1}pPr') for s in chain]
            propnodes += [find(paragraph, 'a:pPr')]
            attrs, props = merge_properties(propnodes)
            defaults = [find(p, 'a:defRPr') for p in propnodes]
            # Number placeholders have their own master paragraph properties.
            if kind == 'sldNum':
                defaults += [find(s, 'p:txBody/a:p/a:pPr/a:defRPr') for s in chain]
                attrs, props = merge_properties(propnodes + [find(s, 'p:txBody/a:p/a:pPr') for s in chain])
            first_run = next((find(c, 'a:rPr') for c in paragraph if tag(c) in ['r','fld'] and ''.join(c.itertext()).strip()), None)
            if first_run is None:
                first_run = find(paragraph, './/m:r/a:rPr')
            normal, size = run_style(defaults + [first_run], scale)
            sizes.add(round(size,4))
            runhtml, plaintext = [], []
            for child in paragraph:
                if tag(child) in ['r', 'fld']:
                    text = find(child, 'a:t').text or ''
                    rstyle, rsize = run_style(defaults + [find(child, 'a:rPr')], scale)
                    sizes.add(round(rsize,4))
                    content = html.escape(text)
                    link = find(child, 'a:rPr/a:hlinkClick')
                    if link is not None:
                        uri = rels[link.get(f"{{{NS['r']}}}id")]['Target']
                        content = f'<a href="{html.escape(uri,quote=True)}">{content}</a>'
                        rstyle['color'] = '#0563c1'
                    runhtml.append(f'<span class="{self.style(rstyle)}">{content}</span>')
                    plaintext.append(text)
                elif tag(child) == 'br':
                    runhtml.append('<br>'); plaintext.append('\n')
                elif tag(child) == 'm':
                    math = find(child, './/m:oMath')
                    if math is None:
                        raise ValueError(f'{boxid}: unhandled mathematical structure')
                    latex = to_latex(math)
                    # Separate nested TeX braces from the course's template syntax.
                    latex = re.sub(r'(?<=\{)(?=\{)|(?<=\})(?=\})', ' ', latex)
                    if not any(tag(c) in ['r','fld'] and ''.join(c.itertext()).strip() for c in paragraph):
                        latex = r'\displaystyle ' + latex
                    runhtml.append(f'<span class="native-math">${html.escape(latex)}$</span>')
                    plaintext.append(f'${latex}$')
            # Exact PowerPoint point sizes are retained; no fit-to-text scaling.
            pct = find(props.get('lnSpc'), 'a:spcPct')
            leading = float(pct.get('val')) / 100000 if pct is not None else 1
            if 'normAutofit' in bodyels:
                leading *= 1 - float(bodyels['normAutofit'].get('lnSpcReduction', 0)) / 100000
            normal['line-height'] = f'{1.2*leading:.4f}'
            normal['text-align'] = {'ctr':'center','r':'right','just':'justify'}.get(attrs.get('algn'),'left')
            normal['margin-left'] = f"{float(attrs.get('marL',0))/EMU:.4f}px"
            bullet = 'buChar' in props
            if not bullet:
                normal['text-indent'] = f"{float(attrs.get('indent',0))/EMU:.4f}px"
            def space(prop):
                pts = find(props.get(prop), 'a:spcPts')
                return float(pts.get('val')) / 100 * 4 / 3 if pts is not None else 0
            normal['margin-top'] = f'{space("spcBef") if index else 0:.4f}px'
            normal['margin-bottom'] = f'{space("spcAft"):.4f}px'
            pclass = self.style(normal)
            content = ''.join(runhtml)
            # Consolidate PowerPoint's per-word runs for comfortable hand editing.
            for pattern, replacement in [
                (r'<span class="(n\d+)">((?:(?!</span>).)*)</span><span class="\1">', r'<span class="\1">\2'),
                (r'<a href="([^"]+)">([^<]*)</a><a href="\1">', r'<a href="\1">\2'),
            ]:
                while True:
                    content, count = re.subn(pattern, replacement, content, flags=re.S)
                    if not count:
                        break
            heading = kind in ['title','ctrTitle'] or original.strip() == self.title.strip()
            element = 'h1' if number == 1 and heading else 'h2' if heading else 'p'
            classes = f'native-paragraph {pclass}' + (' native-bullet' if bullet else '')
            if all(tag(c) not in ['r','fld'] or not ''.join(c.itertext()).strip() for c in paragraph):
                classes += ' native-display-math'
            out.append(f'<{element} class="{classes}">{content}</{element}>')
            texts.append(''.join(plaintext)); originals.append(original)
        return (f'<div class="native-text {geometry}" data-source-text-id="{boxid}">\n' + '\n'.join(out) + '\n</div>',
                {'id': boxid, 'rect': rect, 'paragraphs': texts, 'original_paragraphs': originals, 'font_sizes': sorted(sizes)})

    def decoration(self, shape):
        rect = self.rect(shape, [shape])
        fill = find(shape, 'p:spPr/a:solidFill')
        line = find(shape, 'p:spPr/a:ln')
        rules = {'background': color(fill, 'transparent')}
        if line is not None and find(line,'a:noFill') is None:
            rules['outline'] = f'{float(line.get("w",12700))/EMU:.4f}px solid {color(find(line,"a:solidFill"))}'
        return f'<div aria-hidden="true" class="native-decoration {self.geometry(rect, **rules)}"></div>'

    def build(self):
        previous = json.loads((ROOT / 'assets/source-manifest.json').read_text())
        manifest = {k:v for k,v in previous.items() if k not in ['slides','media','renderer']}
        manifest.update(mode='native', renderer='DrawingML → HTML; OMML → KaTeX; original picture bytes', slides=[])
        slides = []
        for number in range(1,70):
            name = f'ppt/slides/slide{number}.xml'
            xml = ET.fromstring(self.archive.read(name)); rels = self.rels(name)
            layoutname = next(r['Target'] for r in rels.values() if r['Type'].endswith('/slideLayout'))
            layoutname = 'ppt/' + layoutname.removeprefix('../')
            layout = ET.fromstring(self.archive.read(layoutname))
            mastername = next(r['Target'] for r in self.rels(layoutname).values() if r['Type'].endswith('/slideMaster'))
            master = ET.fromstring(self.archive.read('ppt/' + mastername.removeprefix('../')))
            self.title = previous['slides'][number-1]['title']
            sid = 'references' if number == 69 else f'source-{number:03d}'
            page = {'id':sid,'number':number,'title':self.title,'source_xml':name,
                    'source_xml_sha256':sha(self.archive.read(name)),'text_boxes':[],'illustrations':[]}
            pieces = [f'<!-- .slide: id="{sid}" class="native-slide" -->\n']
            for s in children(find(layout,'p:cSld/p:spTree')):
                if tag(s) == 'sp' and ph(s) is None:
                    pieces.append(self.decoration(s))
            for s in children(find(xml,'p:cSld/p:spTree')):
                if tag(s) == 'pic':
                    markup, record = self.picture(s,number,rels)
                    pieces.append(markup); page['illustrations'].append(record)
                elif tag(s) == 'sp':
                    chain = [placeholder(s,master), placeholder(s,layout), s]
                    result = self.text(s,chain,master,number,rels)
                    if result:
                        markup, record = result;pieces.append(markup);page['text_boxes'].append(record)
                    elif find(s,'p:spPr/a:xfrm') is not None:
                        pieces.append(self.decoration(s))
            slides.append('\n'.join(pieces));manifest['slides'].append(page)
        (ROOT/'slides.md').write_text('\n\n---\n\n'.join(slides)+'\n')
        (ROOT/'native-layout.css').write_text('/* Source-derived positions and typography; edit these alongside slides.md. */\n'+'\n'.join(self.css)+'\n')
        (ROOT/'assets/source-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        print('Native text boxes:',sum(len(s['text_boxes']) for s in manifest['slides']))
        print('Original illustrations:',sum(len(s['illustrations']) for s in manifest['slides']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pptx',type=Path)
    args=parser.parse_args()
    recorded = json.loads((ROOT/'assets/source-manifest.json').read_text())
    if sha(args.pptx.read_bytes()) != recorded['source_sha256']:
        raise ValueError('The supplied PPTX differs from the recorded source baseline.')
    with ZipFile(args.pptx) as archive:
        Builder(archive).build()
