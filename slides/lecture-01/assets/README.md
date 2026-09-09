# Lecture 01 media and diagrams

## New teaching visuals

| File | Content and source |
| --- | --- |
| `turing-test-rooms.png` | Simplified Turing test: separated human, machine, and judge, with anonymous text channels |
| `turing-test-terminal.png` | The judge's view: anonymous typed replies, with the participants hidden |
| `instruction-tuning.svg` + `.excalidraw` | Simplified pretraining, demonstrations, and preference-training workflow; Ouyang et al. (2022), Figure 2 |
| `text-pipeline.svg` + `.excalidraw` | Decoding, optional normalization, chunk boundaries, and token encoding |
| `tokenizer-interface.svg` + `.excalidraw` | The checked toy `lowest` example: pieces, IDs, and embedding lookup |

The Turing images were generated with the built-in `image_gen` tool on September
9, 2026. They are conceptual illustrations, not archival images. Their prompts
and the room diagram's readability refinement are in [image-prompts.md](image-prompts.md).
The three diagrams have matching editable Excalidraw scenes and self-contained
SVG exports, using the course palette and Arial. Regenerate both formats with
`uv run python slides/lecture-01/assets/build-diagrams.py` from the repository root.
The diagrams explain mechanisms; they do not show measured model results.

Turing source: [Computing Machinery and Intelligence, §§1–2](https://www.csee.umbc.edu/courses/471/papers/turing.pdf).
Instruction tuning: [Ouyang et al. (2022)](https://arxiv.org/abs/2203.02155).
Text handling: [Python Unicode HOWTO](https://docs.python.org/3/howto/unicode.html)
and [SentencePiece](https://aclanthology.org/D18-2012/).

## Instructor and Spring course media

`llm-26-fall-wechat.png` is the course-group QR image supplied by the instructor
on September 8, 2026. It is stored unchanged and appears on the right of
**About me**. Its embedded notice says the invitation is valid before September 15.

These teaching assets are copied from the instructor's
[Spring 2026 Lecture 01](https://baojian.github.io/llm-26/slides/lecture-01-slides/).
The original images and video clip are unchanged.

| Local file | Original file | Use |
| --- | --- | --- |
| `vision-big-data.png` | `media/ppt/media/image32.png` | Image understanding: Big Data illustration |
| `text-to-video-johannesburg.mp4` | `media/ppt/media/text-to-video-woman.mp4` | Generated-video example with the original Johannesburg prompt |
| `transformer-architecture.png` | `media/ppt/media/image12.png` | Original encoder–decoder Transformer, Vaswani et al. (2017), Figure 1 |
| `llm-timeline-2019-2024.png` | `media/ppt/media/image22.png` | Selected models from 2019–2024, Zhao et al. survey, v16 (March 11, 2025), Figure 3 |

Resolve original paths relative to the linked lecture directory. The original
deck does not identify the video-generation model, so the revised slides make
no model attribution. The clip plays locally with controls and no autoplay.

The Transformer figure comes from [Attention Is All You Need, Figure 1](https://arxiv.org/abs/1706.03762).
The [survey timeline](https://arxiv.org/html/2303.18223v16#S2.F3) is a historical
snapshot, not a current model inventory. Its yellow highlighting denotes
publicly available checkpoints, not necessarily available training code or data.
The ten history pages replace the dense NLP-history collage with readable
landmarks and worked explanations, correcting the LSTM paper's date to 1997. The original
Transformer-author panel image is from GTC 2024, so it is not used to illustrate
a 2017 event.

The `text-to-video-johannesburg-poster.jpg` file is a frame extracted at one second
from the clip using FFmpeg. It appears before playback and replaces the video
in PDF print layouts.

`ollama-terminal.png` comes from the original course's
`lecture-01-tokenization/assets/ollama-terminal.png` and accompanies the extended
tokenization notebook. The JSON chart specifications in this directory belong
to the existing E01–E06 classroom material; their toy data are described in the
slides and notebook.
