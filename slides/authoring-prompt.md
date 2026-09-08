# Prompt for a new lecture

Replace the bracketed fields before using this prompt.

```text
Prepare Lecture [number]: [title] for CS40008.01.

Read AGENTS.md, slides/AGENTS.md, and slides/README.md. Use the existing
Reveal.js framework and slides/example as the layout reference.

Audience: students with Python and introductory machine learning.
Teaching time: [minutes], including [quiz or other activity, if any].
Central question: [one question].
Learning objectives: [observable things students should be able to do].
Source materials: [verified papers, chapters, previous slides, and code].
Required content: [topics].
Practice: [two short checks and one guided experiment, adjusted to the lecture].

First write a concise outline with timing. Develop a small representative
sample before expanding the deck. Build the lecture in the same order as its
companion notebook. Include exercise IDs, concrete prompts, expected outputs,
and checked solution notes. Mark toy examples and cite sources.

Use the shared theme and layouts. Keep all visible text readable on a projector.
Use Plotly.js for interactive charts, Excalidraw for explanatory diagrams, and
Manim for mechanisms that benefit from animation. Choose these automatically
where useful. Follow the framework's asset conventions; retain editable sources
and prepare SVG exports and MP4 videos before publication. Give videos controls
and a summary poster for PDF readers.
Split overloaded slides. Put longer explanations in speaker notes. Implement
any demonstration in a separate local module with a visible initial example
and a reset button. The published lecture must work in a classroom browser.

Run the framework checks, inspect every slide image, and correct layout or
content problems. Report unresolved questions instead of inventing facts.
```
