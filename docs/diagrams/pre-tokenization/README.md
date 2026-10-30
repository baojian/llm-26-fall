# Text preprocessing pipeline diagrams

These original diagrams summarize the common design discussed in
[Lecture 01](../../lecture-01-pre-tokenization.md). They combine reported
corpus practices with the separation between tokenizer learning and encoding;
they are not copied company architecture diagrams.

| Stage | English source / SVG | Chinese source / SVG |
| --- | --- | --- |
| Corpus preparation | [Source](corpus.en.mmd) · [SVG](corpus.en.svg) | [Source](corpus.zh.mmd) · [SVG](corpus.zh.svg) |
| Tokenizer learning | [Source](learning.en.mmd) · [SVG](learning.en.svg) | [Source](learning.zh.mmd) · [SVG](learning.zh.svg) |
| Encoding | [Source](encoding.en.mmd) · [SVG](encoding.en.svg) | [Source](encoding.zh.mmd) · [SVG](encoding.zh.svg) |

Edit the `.mmd` files and run `npm --prefix slides run build:notes` from the
repository root. The build updates both the exported SVGs and the reader's
inline diagrams. Each diagram has an accessible title and description.
Keep the meanings of nodes, edges, and the revision loop aligned in both
languages. Source-specific extraction, repeated filtering passes, and
model-specific formatting are qualified in the accompanying captions.

For supporting evidence and model-specific exceptions, see Lecture 01 sections
2–5 and its [source guide](../../lecture-01-pre-tokenization.md#9-source-guide).
For the diagram language, see the official
[Mermaid flowchart guide](https://mermaid.js.org/syntax/flowchart.html).
