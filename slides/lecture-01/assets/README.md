# Lecture 01 example media

`llm-26-fall-wechat.png` is the course-group QR image supplied by the instructor
on September 8, 2026. It is stored unchanged and appears on the right of
**About me**. Its embedded notice says the invitation is valid before September 15.

These teaching assets are copied from the instructor's
[Spring 2026 Lecture 01](https://baojian.github.io/llm-26/slides/lecture-01-slides/).
The original images and video clips are unchanged.

| Local file | Original file | Use |
| --- | --- | --- |
| `vision-rainfall.jpg` | `media/ppt/media/image41.jpg` | Image understanding: precipitation chart; original image credits the National Climate Center and Xinhuanet |
| `vision-big-data.png` | `media/ppt/media/image32.png` | Image understanding: Big Data illustration |
| `text-to-video-antarctica.mp4` | `media/ppt/media/text-to-video.mp4` | Generated-video example with the original Antarctica prompt |
| `text-to-video-johannesburg.mp4` | `media/ppt/media/text-to-video-woman.mp4` | Generated-video example with the original Johannesburg prompt |
| `transformer-architecture.png` | `media/ppt/media/image12.png` | Original encoder–decoder Transformer, Vaswani et al. (2017), Figure 1 |
| `llm-timeline-2019-2024.png` | `media/ppt/media/image22.png` | Selected models from 2019–2024, Zhao et al. survey, v16 (March 11, 2025), Figure 3 |

Resolve original paths relative to the linked lecture directory. The original
deck does not identify the video-generation model, so the revised slides make
no model attribution. Both clips play locally with controls and no autoplay.

The Transformer figure comes from [Attention Is All You Need, Figure 1](https://arxiv.org/abs/1706.03762).
The [survey timeline](https://arxiv.org/html/2303.18223v16#S2.F3) is a historical
snapshot, not a current model inventory. Its yellow highlighting denotes
publicly available checkpoints, not necessarily available training code or data.
The old slide's dense NLP-history collage is replaced by an English table of
selected landmarks, correcting the LSTM paper's date to 1997. The original
Transformer-author panel image is from GTC 2024, so it is not used to illustrate
a 2017 event.

The two `text-to-video-*-poster.jpg` files are frames extracted at one second
from their corresponding clips using FFmpeg. They appear before playback and
replace the videos in PDF print layouts.

`ollama-terminal.png` comes from the original course's
`lecture-01-tokenization/assets/ollama-terminal.png` and accompanies the extended
tokenization notebook. The JSON chart specifications in this directory belong
to the existing E01–E06 classroom material; their toy data are described in the
slides and notebook.
