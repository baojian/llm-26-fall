"""Guard against losing mathematical meaning when importing Lecture 11 OMML."""

import importlib.util
from pathlib import Path
from xml.etree import ElementTree

import pytest

DECK = Path(__file__).resolve().parents[1] / "slides/lecture-11"
spec = importlib.util.spec_from_file_location("lecture11_omml", DECK / "omml_to_latex.py")
omml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(omml)


def convert(contents):
    return omml.to_latex(ElementTree.fromstring(
        f'<m:oMath xmlns:m="{omml.MATH_NS}" xmlns:a="{omml.DRAWING_NS}">{contents}</m:oMath>'
    ))


def test_conditional_probability_retains_default_separator():
    # OMML encodes the conditioning bar implicitly, rather than as a text run.
    tex = convert('''<m:sSub><m:e><m:r><m:t>𝑞</m:t></m:r></m:e>
        <m:sub><m:r><m:t>𝝓</m:t></m:r></m:sub></m:sSub>
        <m:d><m:dPr/><m:e><m:r><m:t>𝒛</m:t></m:r></m:e>
        <m:e><m:r><m:t>𝒙</m:t></m:r></m:e></m:d>''')
    assert tex == r"{q}_{\boldsymbol{\phi }}\left(\boldsymbol{z}\middle|\boldsymbol{x}\right)"


def test_denoising_input_keeps_tilde_color_and_sample_index():
    tex = convert('''<m:sSup><m:e><m:acc><m:accPr><m:chr m:val="̃"/>
        <m:ctrlPr><a:rPr><a:solidFill><a:srgbClr val="00B050"/></a:solidFill></a:rPr></m:ctrlPr>
        </m:accPr><m:e><m:r><m:t>𝒙</m:t></m:r></m:e></m:acc></m:e>
        <m:sup><m:r><m:t>(𝑖)</m:t></m:r></m:sup></m:sSup>''')
    assert tex == r"{\textcolor{#00B050}{\tilde{\boldsymbol{x}}}}^{(i)}"


def test_fraction_and_summation_keep_numerator_denominator_and_limits():
    tex = convert('''<m:f><m:num><m:r><m:t>1</m:t></m:r></m:num>
        <m:den><m:r><m:t>𝑛</m:t></m:r></m:den></m:f>
        <m:nary><m:naryPr><m:chr m:val="∑"/></m:naryPr>
        <m:sub><m:r><m:t>𝑖=1</m:t></m:r></m:sub>
        <m:sup><m:r><m:t>𝑛</m:t></m:r></m:sup><m:e>
        <m:sSup><m:e><m:r><m:t>𝒙</m:t></m:r></m:e>
        <m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSup>
        </m:e></m:nary>''')
    assert tex == r"\frac{1}{n}\sum\limits_{i=1}^{n}{{\boldsymbol{x}}^{2}}"


def test_math_alphabet_and_upright_data_preserve_source_distinctions():
    tex = convert('''<m:r><m:t>𝑅𝒙𝜙𝜽</m:t></m:r>
        <m:sSub><m:e><m:r><m:t>𝑝</m:t></m:r></m:e><m:sub>
        <m:r><m:rPr><m:sty m:val="p"/></m:rPr><m:t>data</m:t></m:r>
        </m:sub></m:sSub>''')
    assert tex == r"R\boldsymbol{x}\phi \boldsymbol{\theta }{p}_{\mathrm{data}}"
    assert r"\mathbb{R}" not in tex


def test_explicit_empty_delimiters_do_not_create_parentheses_or_bar():
    tex = convert('''<m:d><m:dPr><m:begChr m:val=""/><m:endChr m:val=""/>
        <m:sepChr m:val=""/></m:dPr><m:e><m:r><m:t>𝑥</m:t></m:r></m:e>
        <m:e><m:r><m:t>𝑦</m:t></m:r></m:e></m:d>''')
    assert tex == r"\left.xy\right."


def test_unknown_structures_fail_instead_of_silently_flattening():
    with pytest.raises(ValueError, match="Unsupported OMML structure: mystery"):
        convert('<m:mystery><m:r><m:t>𝑥</m:t></m:r></m:mystery>')
    with pytest.raises(ValueError, match="Unsupported mathematical accent"):
        convert('<m:acc><m:accPr><m:chr m:val="?"/></m:accPr><m:e><m:r><m:t>𝑥</m:t></m:r></m:e></m:acc>')
