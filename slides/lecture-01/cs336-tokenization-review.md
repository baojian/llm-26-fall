# CS336 Lecture 1: Content and Tokenization Teaching Review

**Source:** Percy Liang, Stanford CS336, Spring **2026**,
[Lecture 1: Overview, Tokenization](https://www.youtube.com/watch?v=JuoVZkPBiKk).
The course schedule dates the lecture March 30, 2026; the recording is 79:22 long.
This review was prepared on September 8, 2026 from the complete English automatic
captions and the official executable lecture source. Captions can contain
transcription errors; code and numerical examples were checked separately.
Timestamps below are approximate topic starts in **minutes:seconds**.
[Course schedule](https://cs336.stanford.edu/#schedule) ·
[Executable lecture](https://cs336.stanford.edu/lectures/?trace=lecture_01)

**Main assessment:** Liang makes tokenization compelling by connecting it to
compute efficiency, then building BPE from the limitations of simpler choices.
For our class, keep that progression and give more time to **Unicode versus
bytes**, **BPE training versus encoding**, and **what compression does and does
not measure**. His detailed tokenization section lasts only about 14 minutes;
the encoding walkthrough is explicitly skipped for time. The recommendations
below are teaching judgments and proposed additions, separate from the summary
of what he presents.

## 1. Main content of the full lecture

| Video position | Main content | Takeaway for our course |
| --- | --- | --- |
| [00:04–03:23](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4s) | Staff introductions and changes in the third course offering. | Explain what students will learn by building. |
| [03:23–11:36](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=203s) | Motivation: abstractions hide mechanisms; small models teach mechanics and an efficiency mindset, while empirical intuitions may change with scale. | Organize the course around the best model achievable with limited resources. |
| [11:36–19:26](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=696s) | Language-model history, the open-model ecosystem, and increasing importance of inference. | Give historical context without requiring students to memorize model releases. |
| [19:26–27:17](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1166s) | Executable lecture format, workload, assignments, AI tutoring policy, and compute access. | Connect explanations to inspectable programs and correctness checks. |
| [27:17–35:53](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1637s) | Basics: tokenization, architecture, optimization, and training; balance expressiveness, stability, and efficiency. | Position tokenization within the complete training pipeline. |
| [35:53–45:12](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=2153s) | Systems: resource accounting, kernels, data movement, parallelism, and inference. | Explain why representation choices affect hardware work. |
| [45:12–53:29](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=2712s) | Scaling recipes and predicting larger runs from smaller experiments. | Distinguish measured evidence from extrapolation. |
| [53:29–60:20](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=3209s) | Evaluation, data collection, transformation, filtering, deduplication, and mixtures. | Data preparation is part of model building. |
| [60:20–65:07](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=3620s) | Alignment and reinforcement learning, followed by a return to the resource-budget theme. | Relate the course units through their shared constraints. |
| [65:07–79:15](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=3907s) | Detailed tokenization lesson, BPE implementation, and closing remarks. | This is the main segment to revisit for Lecture 01 preparation. |

The first hour is principally a course overview. Tokenization gets an early
preview at [28:05](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1685s), where Liang
connects shorter sequences and variable-size chunks to efficient computation.
He returns to that motivation before and after the detailed lesson.

## 2. How Liang presents tokenization

### The sequence of explanations

| Start | What he presents | Teaching function |
| --- | --- | --- |
| [65:22](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=3922s) | Raw Unicode text, integer token IDs, and an encode/decode interface. | Establish the input, output, and round-trip contract. |
| [65:50](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=3950s) | An intended interactive tokenizer demo; internet access fails, so he describes spaces, repeated words, and digit groups. | Show surprising behavior before introducing the algorithm. |
| [67:00](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4020s) | A real tokenizer encodes and decodes mixed English, Chinese, and emoji text; 20 bytes become 8 tokens. | Make the interface and bytes-per-token metric concrete. |
| [68:28](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4108s) | Character tokenizer using `ord` and `chr`; vocabulary coverage and rarity problems. | Test the most direct mapping from an existing string representation. |
| [69:44](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4184s) | UTF-8 byte tokenizer: 256 possible values, but longer sequences. | Exchange vocabulary complexity for sequence length. |
| [70:44](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4244s) | Word-like chunks; shorter sequences, many rare words, and unseen-word handling. | Show the opposite extreme and motivate reusable subword pieces. |
| [71:58](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4318s) | BPE intuition: learn frequent adjacent combinations, leaving rare material in smaller pieces. | Introduce a data-dependent compromise. |
| [73:16](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4396s) | Three training merges on `the cat in the hat`, with new IDs and a shrinking sequence. | Connect pair counts, vocabulary growth, and compression through code. |
| [75:21](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4521s) | Encode `the quick brown fox` using learned merges and decode it back. He skips stepping through the encoding code. | State the training/use distinction, but leave its mechanics mostly implicit. |
| [76:02](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4562s) | Slow implementation, relevant merges, special tokens, and pre-tokenization. | Bridge the toy algorithm to implementation work. |
| [77:29](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4649s) | Recap; future alternatives should also learn useful abstractions and allow variable computation. | Return from implementation details to the original motivation. |

### What works well

His progression gives students a reason to want BPE before seeing its loop.
Each baseline has a recognizable benefit and a cost. Reusing the same interface
also keeps the conceptual task stable while the implementation changes.

The executable lecture exposes intermediate state: IDs, pair counts, vocabulary
entries, and reconstructed text. This is especially useful at
[73:37–75:05](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4417s), where a merge
becomes an operation students can inspect rather than an abstract instruction.

The weakness is pacing for beginners. At
[68:30](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4110s) he announces a fast
walkthrough; at [75:44](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4544s) he
skips the encoding code. Our lesson should preserve the reasoning while adding
prediction pauses and a complete example on unseen text.

## 3. Where the explanation can be clearer

These are proposed improvements for our students, not claims that Liang's
lecture source lacks every relevant detail.

| Priority | Passage and possible confusion | Concrete improvement |
| --- | --- | --- |
| Highest | **Training versus encoding**, 73:16–75:58: students may recount frequent pairs in each new input. | Place two workflows side by side: training learns vocabulary and merge priority; encoding uses those fixed objects. Trace every merge on unseen text. |
| Highest | **Characters, bytes, and tokens**, 68:28–70:42: all appear as integers, making them easy to conflate. | Give one string three rows: Unicode code points, UTF-8 bytes, and tokenizer IDs. Include Chinese and emoji. Explain that IDs identify vocabulary entries; embeddings are learned later by the model. |
| Highest | **Compression**, 67:25–68:19: a larger number may sound like a universally better tokenizer or smaller file. | Label the metric “UTF-8 bytes per token.” Hold the text fixed; report vocabulary size and held-out results alongside it. Distinguish token count, runtime, storage size, and model quality. |
| High | **Merge mechanics**, 73:37–75:05: ties and overlapping pairs can be missed during code stepping. | State the tie rule, count adjacent pairs, apply non-overlapping replacements, then recount. Show `aaa` as a separate edge case. |
| High | **Pre-tokenization**, 76:53–77:09: it is introduced mainly as a speed improvement. | Draw allowed chunk boundaries and show that BPE cannot merge across them. Boundary rules change the learned vocabulary and segmentation as well as runtime. |
| High | **Round trips**, 65:37–65:47 and 67:14–67:23: students may think every token is independently readable text. | Display token bytes and concatenate them before UTF-8 decoding. State the contract for supported text without lossy normalization; arbitrary generated IDs need not form valid UTF-8. |
| Medium | **Variable computation**, 28:55–29:15 and 78:41–78:56: “interesting” chunks could sound like semantic understanding. | Explain that BPE uses corpus frequency, not a learned judgment of meaning or difficulty. Rare strings often require more token positions; this is a heuristic allocation of work. |
| Medium | **Tokenizer quirks**, 65:50–66:57: the intended live demo is unavailable. | Prepare an offline example with token IDs and byte pieces. Ask students to predict the effect of a leading space, an emoji, or a new word before revealing output. |

### A. Separate representation from meaning

Use Liang's exact mixed-text example, `Hello, 🌍! 你好!`, and compare:

| Representation | Sequence length | Meaning of an element |
| --- | ---: | --- |
| Unicode code points | 13 | One code point value |
| UTF-8 bytes | 20 | One integer from 0 to 255 |
| `o200k_base` tokens | 8 | One vocabulary entry representing a byte sequence |

The following is a **locally verified expansion** of the lecture example,
using `tiktoken` 0.14.0 and the explicit encoding name `o200k_base`:

```python
import tiktoken

text = "Hello, 🌍! 你好!"
encoding = tiktoken.get_encoding("o200k_base")
ids = encoding.encode(text)
pieces = [encoding.decode_single_token_bytes(i) for i in ids]
assert len(text) == 13 and len(text.encode("utf-8")) == 20
assert len(ids) == 8
assert b"".join(pieces) == text.encode("utf-8")
assert encoding.decode(ids) == text
```

The emoji is split across two token byte sequences: one ends with
`b'\xf0\x9f\x8c'` after a leading space, and the next is `b'\x8d'`.
Neither token can independently reconstruct that emoji. Meanwhile, `你好`
occupies one token here. A token can therefore be smaller than a code point or
contain several code points. These are measured properties of this encoding,
not universal rules for Chinese or emoji.

For a second small example, `é` and `e\u0301` can look alike but contain one and
two code points, respectively. A visible symbol is not a reliable counting unit.
Use the [Python Unicode HOWTO, Definitions and Encodings](https://docs.python.org/3/howto/unicode.html#definitions)
to distinguish code points from their byte representation.

### B. Finish the BPE walkthrough on both training and unseen text

This reconstruction uses Liang's toy corpus and the published algorithm.
It starts with **all 256 byte values**, has no special tokens or pre-tokenizer,
and resolves tied pair counts by their first encounter while scanning left to
right. Spaces below are actual bytes; `␠` is only a display label.
[Video walkthrough](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4396s) ·
[Published training function][training-source]

| Step | Selected pair | Pair count | New ID and byte content | Tokens in training text | Vocabulary size |
| --- | --- | ---: | --- | ---: | ---: |
| Initial | — | — | IDs 0–255 each represent one byte | 18 | 256 |
| 1 | `t` + `h` | 2 | 256 → `b'th'` | 16 | 257 |
| 2 | `th` + `e` | 2 | 257 → `b'the'` | 14 | 258 |
| 3 | `the` + `␠` | 2 | 258 → `b'the '` | 12 | 259 |

After three merges, the training-text ratio is `18 / 12 = 1.5` bytes per
token. Keep the original byte entries: merging adds an entry rather than
deleting the constituent tokens. The stored artifacts are a vocabulary mapping
IDs to bytes and an ordered list of merge rules.

Now **freeze those artifacts** and encode `the quick brown fox`:

```text
Start:  t | h | e | ␠ | q | u | i | c | k | ␠ | b | r | o | w | n | ␠ | f | o | x
Rule 1: th | e | ␠ | q | u | i | c | k | ␠ | b | r | o | w | n | ␠ | f | o | x
Rule 2: the | ␠ | q | u | i | c | k | ␠ | b | r | o | w | n | ␠ | f | o | x
Rule 3: the␠ | q | u | i | c | k | ␠ | b | r | o | w | n | ␠ | f | o | x
```

The result has 16 tokens for 19 bytes. The words `quick`, `brown`, and `fox`
were absent from the training string, but their bytes remain representable.
No unknown-word token or new vocabulary entry is needed. Decoding looks up
each ID's bytes, concatenates them, and decodes the complete byte sequence.
[Published encoder and decoder][encoder-source]

Have students explain three observations before running anything:

- Training selects pairs by their frequency in the current training sequences;
  encoding follows the learned priority without retraining on the new input.
- A merge rule can join tokens that already represent multiple bytes. BPE does
  not stop after creating two-byte pieces.
- An unseen word is encodable because the byte vocabulary is complete. That
  does not imply that the language model understands the word.

For overlap, `a a a` contains two adjacent `a a` pairs, but a left-to-right
replacement produces `aa a`, reducing three tokens to two. Pair frequency and
the number of replacements are different when occurrences overlap.

### C. Make the efficiency tradeoff measurable

For a fixed nonempty text, define

$$
C = \frac{\text{UTF-8 byte count}}{\text{token count}}.
$$

If `C` doubles on that same text, its token count halves. With fixed model
dimensions, the quadratic component of dense self-attention then involves
about one quarter as many position pairs. This is an explanation of one cost
component, not a promise of a fourfold end-to-end speedup: other computation,
memory traffic, vocabulary projection, and implementation choices also matter.
This calculation follows from the attention equation and complexity comparison
in [Vaswani et al., Sections 3.2.1 and 4](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1).

Vocabulary growth also has a concrete cost. An embedding table with `V` entries
of dimension `d` has `V × d` parameters. In a toy calculation with `d = 1024`,
increasing `V` from 32,000 to 128,000 adds 98,304,000 embedding parameters.
An untied output matrix adds its own parameters; weight tying changes that
accounting. More entries also spread observations over more distinct pieces.
See [Vaswani et al., Section 3.4](https://arxiv.org/html/1706.03762v7#S3.SS4)
for embeddings and weight sharing; the parameter totals here are our calculation.

For our class, compare tokenizers on the same **held-out** English and Chinese
texts. Record the encoding name, vocabulary size, bytes, tokens, and bytes per
token. Interpret results separately by language. Higher bytes per token alone
does not establish better semantic representations, lower storage size, or
better language-model predictions. Raw per-token perplexities also have
different units when the tokenizations differ.

## 4. Suggested adaptation for our Lecture 01

Use Liang's sequence as the conceptual backbone. A proposed **45-minute
tokenization block**, following the initial Unicode introduction, is:

| Minutes | Focus | Student action |
| --- | --- | --- |
| 0–5 | Interface and mixed-text example | Predict code point, byte, and token counts. |
| 5–12 | Character, byte, and word choices | Explain one cost of each using a comparison table. |
| 12–22 | BPE training | Count pairs, resolve a tie, trace merges, and explain overlaps. |
| 22–31 | Frozen vocabulary and encoding | Encode unseen text and reconstruct its bytes. |
| 31–36 | Boundaries and special tokens | Explain why a boundary forbids a possible merge; distinguish a reserved control ID from its displayed spelling. |
| 36–43 | Held-out comparison | Interpret English and Chinese token counts and identify a missing quality metric. |
| 43–45 | Exit check | Explain why encoding an unseen word does not train the tokenizer. |

For a boundary example, a pre-tokenizer might separate `the` and its following
space. In that case the toy merge `the + ␠` would be forbidden. Choose and show
an actual rule before comparing outputs; pre-tokenization conventions differ.
The original NLP BPE formulation applies merges within words with a boundary
symbol, which differs from Liang's unrestricted toy byte stream.
[Sennrich et al. (2016), Section 3.2](https://aclanthology.org/P16-1162.pdf#page=3)

The existing [teaching plan](teaching-plan.md) already reserves time for Unicode,
BPE training, fixed merge order, and held-out comparison. Its `low` × 5 and
`lower` × 2 example is suitable: keep its declared tie rule and explicitly
separate training strings. It need not reproduce Stanford's different toy
corpus or tie policy.

When implementing these recommendations in [slides.md](slides.md) and the
[exercise notebook](lecture-01-exercise.ipynb), synchronize exercise IDs and
solutions with the teaching plan. This review supplies preparation notes;
the proposed activities are not new published exercise assignments.

## 5. Accuracy notes when adapting the recording

- **Use the correct edition.** This video is Spring 2026. Searches often return
  the similarly titled 2025 lecture. Its timing and contents should not be
  substituted for this recording.
- **Name the encoding.** Liang labels the real example as a GPT-5 tokenizer;
  the source specifically loads `o200k_base`. Use that explicit name when
  reproducing this demonstration, since model labels alone are insufficient
  specifications. [Source function][encoding-source]
- **Correct the historical shortcut.** Around
  [72:19](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4339s), the spoken account
  identifies GPT-2 as the first use of BPE for language models. The 2018 GPT
  paper already describes BPE with 40,000 merges. Teach that GPT-2 used BPE,
  without the priority claim.
  [Radford et al. (2018), Section 4.1, Model specifications](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=5)
- **Avoid a timeless Unicode count.** Liang's approximate character count is
  background context. More fundamentally, raw code point IDs are not a dense
  vocabulary: `max(ids) + 1` measures the table size needed for direct indexing,
  not the number of distinct characters observed. Reindexing characters is a
  separate choice. [Python Unicode definitions](https://docs.python.org/3/howto/unicode.html#definitions)
- **Qualify the baseline verdict.** The lecture's efficiency argument concerns
  its model setting. Character and byte representations remain useful
  baselines; the closing discussion itself leaves room for architectures that
  learn how to group bytes.
  [Closing discussion](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4681s)

## 6. Verification and source details

The source links below are pinned to commit
`607a238629cf5332f71085d19c97dff41decf661`, the latest change to `lecture_01.py`
reported by the official repository when retrieved. This is the published
source consulted, not a claim that this commit was used on the recording date.
The relevant spoken examples were cross-checked against the captions.

Verification used the project Python environment with `uv run --no-sync python`.
The published toy functions were run without the lecture viewer. Checks covered
all three merges, vocabulary sizes, training and unseen-text lengths, round
trips, overlap behavior, the mixed-text `o200k_base` example, and the embedding
parameter calculation. The full caption transcript is not reproduced here.

[training-source]: https://github.com/stanford-cs336/lectures/blob/607a238629cf5332f71085d19c97dff41decf661/lecture_01.py#L729-L759
[encoder-source]: https://github.com/stanford-cs336/lectures/blob/607a238629cf5332f71085d19c97dff41decf661/lecture_01.py#L542-L564
[encoding-source]: https://github.com/stanford-cs336/lectures/blob/607a238629cf5332f71085d19c97dff41decf661/lecture_01.py#L574-L576
