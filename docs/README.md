# Course reading pages

The [Lecture 01 reader](lecture-01-pre-tokenization.html) explains text
preprocessing before tokenization. Its English and Chinese versions share
section links, code samples, citations, and three illustrated stages:

1. Prepare a corpus: collection, decoding, extraction, repair, annotation,
   quality filtering, deduplication, exclusions, and versioning.
2. Learn a tokenizer: representative sampling, text rules, vocabulary learning,
   evaluation, and freezing the artifact.
3. Encode text: source-dependent application processing, the frozen tokenizer,
   and the subsequent training or inference path.

The diagrams synthesize common practices described in the lecture's sources.
They do not assert a single mandatory order or an undisclosed company recipe.

## Source files

| File | Edit for |
| --- | --- |
| [English Markdown](lecture-01-pre-tokenization.md) | Canonical section structure, explanations, examples, citations |
| [Chinese Markdown](lecture-01-pre-tokenization.zh.md) | Human-readable Chinese translation with matching numbered sections |
| [Diagram sources](diagrams/pre-tokenization/README.md) | English and Chinese Mermaid flowcharts |
| [Reader builder](../scripts/build-notes.mjs) | Shared page shell and Markdown-to-HTML generation |
| [Reader styles](../assets/notes.css) and [behavior](../assets/notes.js) | Reading layout, diagram tabs, navigation, copying, and print |
| [Language controls](../assets/course-language.js) and [styles](../assets/course-language.css) | Shared EN / 中文 preference and interface translations |

Markdown is the editable source; HTML is the reading format. Commit both source
files and generated HTML/SVG assets so GitHub Pages can serve them directly.
The generated HTML includes both translations and all six SVG diagrams; it
does not download a renderer, fonts, or a translation service at runtime.
It also works when opened directly from disk. The separate Reveal.js slide
pages still require the published website or local preview server.

## Rebuild and check

From the repository root, install the pinned authoring dependencies once:

```sh
npm ci --prefix slides
npm --prefix slides run browser:install
```

After editing Markdown or Mermaid sources:

```sh
npm --prefix slides run build:notes
npm --prefix slides run check:notes
```

The builder uses Markdown-it and Mermaid with Playwright's Chromium to export
SVGs. These are authoring dependencies; students do not need Node or Chromium
installed separately to read the HTML. Mermaid uses strict security mode and
SVG text labels. Only repository-controlled sources enter this build.

The builder rejects mismatched section numbers, executable code blocks,
sample JSON, and reference definitions between translations. Keep the same
heading order and numbering. Code, commands, model identifiers, and source
URLs stay unchanged in translations.

The browser check exercises both articles and all six diagrams, section links,
keyboard navigation, copy buttons, language persistence, 320px/390px layouts,
print behavior, and direct-file viewing with external requests blocked.
It also checks the homepage, slide toolbar, and notebook launcher. Review the
generated images in ignored `slides/.checks/notes/`. Shared slide framework
changes additionally require the deck checks described in
[slides/README.md](../slides/README.md#checks-before-class).

## Language behavior

Every course-owned HTML shell uses the same EN / 中文 control, including the
slide template for future lectures. Third-party vendored HTML is unchanged.
The preference is saved in browser storage; an explicit `?lang=en` or
`?lang=zh` takes precedence and becomes the saved preference. Storage is scoped
to the website origin: public GitHub Pages and a local preview are separate.
If browser storage is unavailable, the current page still switches languages.

The reader translates prose, navigation, and diagram labels, preserves the
current numbered section when switching languages, and prints all three
diagrams in the selected language. Existing slide Markdown and companion
notebook content are English; their bilingual shells do not imply translated
lesson content. No machine translation runs in a student's browser.

## Why Mermaid for this reader

[Mermaid](https://mermaid.js.org/intro/) describes nodes and edges in text,
which makes pipeline diagrams easy to review alongside Markdown. Its
[flowcharts](https://mermaid.js.org/syntax/flowchart.html) support groups,
branches, and feedback edges. SVG exports remain sharp when resized or
printed, and the reader offers both an SVG download and editable source.

The existing slides retain their Excalidraw diagram convention. Mermaid is
the authoring choice for this new documentation reader; no slide drawing
assets were converted.
