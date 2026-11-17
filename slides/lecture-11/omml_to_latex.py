"""Convert the mathematical structures in the Lecture 11 PPTX to editable TeX.

This intentionally rejects unhandled structures instead of flattening their text.
DrawingML font metadata is omitted; mathematical alphabet styles, run colors,
delimiters, scripts, fractions, limits, and accents are retained for KaTeX.
"""

from __future__ import annotations

import re
import unicodedata
from xml.etree.ElementTree import Element

MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS = {"m": MATH_NS, "a": DRAWING_NS}

_SYMBOLS = {
    "α": r"\alpha ", "β": r"\beta ", "γ": r"\gamma ",
    "δ": r"\delta ", "ε": r"\varepsilon ", "ϵ": r"\epsilon ",
    "ζ": r"\zeta ", "η": r"\eta ", "θ": r"\theta ",
    "ϑ": r"\vartheta ", "ι": r"\iota ", "κ": r"\kappa ",
    "λ": r"\lambda ", "μ": r"\mu ", "ν": r"\nu ",
    "ξ": r"\xi ", "π": r"\pi ", "ϖ": r"\varpi ",
    "ρ": r"\rho ", "ϱ": r"\varrho ", "σ": r"\sigma ",
    "ς": r"\varsigma ", "τ": r"\tau ", "υ": r"\upsilon ",
    "φ": r"\varphi ", "ϕ": r"\phi ", "χ": r"\chi ",
    "ψ": r"\psi ", "ω": r"\omega ", "Γ": r"\Gamma ",
    "Δ": r"\Delta ", "Θ": r"\Theta ", "Λ": r"\Lambda ",
    "Ξ": r"\Xi ", "Π": r"\Pi ", "Σ": r"\Sigma ",
    "Υ": r"\Upsilon ", "Φ": r"\Phi ", "Ψ": r"\Psi ",
    "Ω": r"\Omega ", "×": r"\times ", "⋅": r"\cdot ",
    "−": "-", "∈": r"\in ", "∣": r"\mid ", "∼": r"\sim ",
    "≈": r"\approx ", "…": r"\ldots ", "′": r"\prime ",
    "∞": r"\infty ", "≤": r"\leq ", "≥": r"\geq ",
    "≠": r"\neq ", "∂": r"\partial ", "∇": r"\nabla ",
    "±": r"\pm ", "→": r"\to ", "←": r"\leftarrow ",
    "∑": r"\sum ", "∏": r"\prod ", "∫": r"\int ",
    "{": r"\{", "}": r"\}", "_": r"\_", "%": r"\%",
    "#": r"\#", "&": r"\&", "$": r"\$", "\\": r"\backslash ",
    "\u00a0": r"\ ",
}


def _local(element: Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _value(element: Element | None, default: str = "") -> str:
    return default if element is None else element.get(f"{{{MATH_NS}}}val", default)


def _color(element: Element, path: str) -> str | None:
    color = element.find(path + "/a:solidFill/a:srgbClr", NS)
    if color is None:
        return None
    value = color.get("val", "")
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", value):
        raise ValueError(f"Invalid mathematical run color: {value!r}")
    return value.upper()


def _colored(tex: str, color: str | None) -> str:
    return rf"\textcolor{{#{color}}}{{{tex}}}" if color else tex


def _character(char: str) -> str:
    name = unicodedata.name(char, "")
    if not name.startswith("MATHEMATICAL "):
        return _SYMBOLS.get(char, char)

    # NFKC would merge phi-symbol and small-phi shapes. Preserve the source
    # glyph distinction before reducing the mathematical alphabet to TeX.
    variants = {
        "PHI SYMBOL": r"\phi ", "EPSILON SYMBOL": r"\epsilon ",
        "THETA SYMBOL": r"\vartheta ", "RHO SYMBOL": r"\varrho ",
        "PI SYMBOL": r"\varpi ", "KAPPA SYMBOL": r"\varkappa ",
    }
    base = next((value for suffix, value in variants.items() if name.endswith(suffix)), None)
    if base is None:
        normalized = unicodedata.normalize("NFKC", char)
        if normalized == char:
            raise ValueError(f"Unsupported mathematical alphabet character: {char!r}")
        base = "".join(_SYMBOLS.get(c, c) for c in normalized)
    if "BOLD ITALIC" in name:
        return rf"\boldsymbol{{{base}}}"
    if "BOLD" in name:
        return rf"\mathbf{{{base}}}"
    if "DOUBLE-STRUCK" in name:
        return rf"\mathbb{{{base}}}"
    if "SCRIPT" in name:
        return rf"\mathcal{{{base}}}"
    if "FRAKTUR" in name:
        return rf"\mathfrak{{{base}}}"
    if "MONOSPACE" in name:
        return rf"\mathtt{{{base}}}"
    if "SANS-SERIF" in name:
        return rf"\mathsf{{{base}}}"
    return base


def _run(element: Element) -> str:
    text = "".join(child.text or "" for child in element.findall("m:t", NS))
    tex = "".join(_character(char) for char in text)
    style = _value(element.find("m:rPr/m:sty", NS))
    properties = element.find("a:rPr", NS)
    # Mathematical Unicode already encodes its weight and slant. Ordinary
    # letters (notably the source's upright 'data' subscript) use run metadata.
    encoded_style = any(unicodedata.name(char, "").startswith("MATHEMATICAL ") for char in text)
    if not encoded_style:
        bold = style in {"b", "bi"} or (properties is not None and properties.get("b") == "1")
        upright = style in {"p", "b"} or (properties is not None and properties.get("i") == "0")
        if bold:
            tex = rf"\mathbf{{{tex}}}" if upright else rf"\boldsymbol{{{tex}}}"
        elif upright and any(char.isalpha() for char in text):
            tex = rf"\mathrm{{{tex}}}"
    return _colored(tex, _color(element, "a:rPr"))


def _delimiter(char: str) -> str:
    delimiters = {
        "": ".", "(": "(", ")": ")", "[": "[", "]": "]",
        "{": r"\{", "}": r"\}", "|": "|", "‖": r"\Vert ",
        "∣": "|", "〈": r"\langle ", "〉": r"\rangle ",
        "⟨": r"\langle ", "⟩": r"\rangle ",
    }
    if char not in delimiters:
        raise ValueError(f"Unsupported mathematical delimiter: {char!r}")
    return delimiters[char]


def to_latex(element: Element | None) -> str:
    """Return delimiter-free TeX for an OMML node; reject unsupported content.

    Both ElementTree and lxml elements work. Callers choose inline/display math
    and HTML escaping. No source symbol is silently dropped or corrected.
    """
    if element is None:
        return ""
    tag = _local(element)
    if tag.endswith("Pr"):
        return ""
    if tag in {"oMath", "oMathPara", "e", "sub", "sup", "num", "den", "deg"}:
        return "".join(to_latex(child) for child in element)
    if tag == "r":
        return _run(element)
    if tag == "t":
        return "".join(_character(char) for char in element.text or "")

    def child(name: str) -> str:
        return to_latex(element.find(f"m:{name}", NS))

    if tag == "sSub":
        tex = "{" + child("e") + "}_{" + child("sub") + "}"
    elif tag == "sSup":
        tex = "{" + child("e") + "}^{" + child("sup") + "}"
    elif tag == "sSubSup":
        tex = "{" + child("e") + "}_{" + child("sub") + "}^{" + child("sup") + "}"
    elif tag == "f":
        kind = _value(element.find("m:fPr/m:type", NS), "bar")
        if kind != "bar":
            raise ValueError(f"Unsupported fraction type: {kind}")
        tex = rf"\frac{{{child('num')}}}{{{child('den')}}}"
    elif tag == "d":
        left = _delimiter(_value(element.find("m:dPr/m:begChr", NS), "("))
        right = _delimiter(_value(element.find("m:dPr/m:endChr", NS), ")"))
        separator = _value(element.find("m:dPr/m:sepChr", NS), "|")
        # The absent sepChr default matters: the source uses it to represent
        # q_phi(z|x), with z and x in separate <m:e> arguments.
        separator_tex = rf"\middle{_delimiter(separator)}" if separator else ""
        body = separator_tex.join(to_latex(arg) for arg in element.findall("m:e", NS))
        tex = rf"\left{left}{body}\right{right}"
    elif tag == "nary":
        symbol = _value(element.find("m:naryPr/m:chr", NS), "∫")
        operators = {"∑": r"\sum", "∏": r"\prod", "∫": r"\int", "∮": r"\oint", "⋃": r"\bigcup", "⋂": r"\bigcap"}
        if symbol not in operators:
            raise ValueError(f"Unsupported n-ary operator: {symbol!r}")
        placement = _value(element.find("m:naryPr/m:limLoc", NS), "undOvr" if symbol in {"∑", "∏", "⋃", "⋂"} else "subSup")
        tex = operators[symbol] + (r"\limits" if placement == "undOvr" else r"\nolimits")
        for name, mark in (("sub", "_"), ("sup", "^")):
            hidden = _value(element.find(f"m:naryPr/m:{name}Hide", NS), "0") in {"1", "true", "on"}
            if child(name) and not hidden:
                tex += mark + "{" + child(name) + "}"
        tex += "{" + child("e") + "}"
    elif tag == "acc":
        accent = _value(element.find("m:accPr/m:chr", NS), "̂")
        accents = {"̃": r"\tilde", "~": r"\tilde", "̂": r"\hat", "ˆ": r"\hat", "̅": r"\bar", "⃗": r"\vec", "̇": r"\dot", "̈": r"\ddot"}
        if accent not in accents:
            raise ValueError(f"Unsupported mathematical accent: {accent!r}")
        tex = accents[accent] + "{" + child("e") + "}"
    elif tag == "rad":
        degree_hidden = _value(element.find("m:radPr/m:degHide", NS), "0") in {"1", "true", "on"}
        degree = "" if degree_hidden else child("deg")
        tex = r"\sqrt" + ("[" + degree + "]" if degree else "") + "{" + child("e") + "}"
    else:
        raise ValueError(f"Unsupported OMML structure: {tag}")
    return _colored(tex, _color(element, f"m:{tag}Pr/m:ctrlPr/a:rPr"))
