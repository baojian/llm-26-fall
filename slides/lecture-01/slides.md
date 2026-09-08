<!-- .slide: class="title-slide" id="title" -->

<p class="eyebrow">CS40008.01</p>

# Introduction and Tokenization

<p class="subtitle">Lecture 01 – Introduction to NLP &amp; LLMs</p>

<p class="byline">Baojian Zhou<br>School of Data Science<br>Fudan University<br>September 9, 2026</p>

Note:
Ask students how they think a model represents a Chinese sentence.

---

<!-- .slide: class="outline-slide" id="outline-overview" -->

## Outline

<ul class="outline-topics">
<li aria-current="step">Course Overview</li>
<li>Development of NLP &amp; LLMs</li>
<li>Basics for Text Preprocessing</li>
<li>Text Tokenization</li>
</ul>

Note:
Period 1, minutes 0–25: course overview. Read the four topics once. There are three 45-minute teaching periods, with breaks outside the 135 minutes. Return to this outline at each transition. Source: the four-part structure in Fudan Spring Lecture 01, https://baojian.github.io/llm-26/slides/lecture-01-slides/.

---

<!-- .slide: id="goals" -->

## Today’s question

How does an LLM application turn our text into something a model can predict?

- Call a local model and examine its answers.
- Explain the difference between characters, bytes, and tokens.
- Train a small BPE tokenizer and test it on unseen text.

Note:
Ask for one hypothesis before giving the vocabulary. The notebook follows the same sequence; the core tokenizer needs only Python’s standard library.

---

<!-- .slide: id="survey" -->

## The LLM apps you use

Which apps do you use most in everyday life?

Choose **at most two** in the [first-lecture survey](https://github.com/baojian/llm-26-fall/issues/6).

Submit your response through one pull request, following the issue’s guide.

> What do you usually ask these apps to do?

Note:
Take two or three examples aloud. Explain that an app may combine a model, search, tools, and an interface. Introduce the GitHub workflow here; students can finish the PR after class. Do not spend the period debugging individual accounts.

---

<!-- .slide: id="app-model-tokenizer" -->

## App, model, tokenizer

| Component | In our local experiment |
| :--- | :--- |
| Interface | A Jupyter notebook |
| Model runtime | Ollama |
| Model | Qwen3, using an installed model tag |
| Tokenizer | The vocabulary and rules paired with that model |

Our toy tokenizer is a separate learning exercise.

Note:
Ollama serves the model; it is not itself the language model. The tokenizer we train today is not compatible with Qwen’s existing embedding table. A model tag identifies a packaged model; record its digest because tags can change. API source: https://docs.ollama.com/api/tags.

---

<!-- .slide: id="natural-language" -->

## Natural language needs context

> A man saw a boy with a telescope.

Who had the telescope?

> 冬天，能穿多少穿多少。<br>夏天，能穿多少穿多少。

The same words can support different interpretations.

Note:
Invite both attachment readings of the English sentence. In Chinese, contrast wearing as much as possible in winter with as little as possible in summer. These are ambiguity examples from Fudan Spring Lecture 01; context resolves meanings, while tokenization only chooses a representation.

---

<!-- .slide: id="nlp-tasks" -->

## Language tasks in one notebook

| Task | Input → output |
| :--- | :--- |
| Sentiment | Camera review → label and reason |
| Translation | Chinese paragraph → English text |
| Generation | Instructions → a short course welcome |
| Image understanding | Image + question → description |

Run text examples together; explore vision after class.

Note:
Preserve the application examples from the previous lecture, but execute their Ollama calls in Jupyter. Vision needs a separate compatible model. Do not spend class downloading it. Source: Fudan Spring Lecture 01, sentiment, translation, generation, and image-understanding examples.

---

<!-- .slide: class="exercise" id="exercise-01" -->

## Does “light” mean positive?

<p class="exercise-meta">Exercise E01 · 5 minutes · Notebook E01</p>

1. “Nice and compact to carry!”
2. “The camera is small and light; I avoid carrying bulky cameras.”
3. “The camera feels flimsy, plastic, and very light.”

Predict the labels, then compare them with Ollama’s responses.

<div class="answer fragment"><p>Expected: positive, positive, negative. The surrounding words change the meaning of “light.”</p></div>

Note:
Allow one minute to label individually, two to run notebook E01, and two to compare. These are shortened versions of the camera reviews in the previous Fudan lecture; the notebook keeps the full review text. The expected labels are human judgments for these examples, not guaranteed model outputs. Discuss any disagreement.

---

<!-- .slide: id="notebook-demo" -->

## A local model call

Open **Notebook** and run the Ollama section.

- Send a prompt; inspect the returned text.
- Keep the model and generation settings fixed.
- Change the task to translation or generation.

Record an error or an unsupported claim, even when the answer sounds fluent.

Note:
The notebook uses POST /api/generate with stream=false and think=false, then reads response. Ollama is installed separately from uv; the core tokenizer cells still run if it is unavailable. The instructor can run the demonstration on the classroom computer; students without a local model can work from their predictions. Source: https://docs.ollama.com/api/generate.

---

<!-- .slide: id="course-work" -->

## What we will build

Text representation → language models → Transformers → LLM applications

| Assessment | Weight |
| :--- | ---: |
| Quizzes | 10% |
| Assignments | 45% |
| Individual course project | 45% |

A1 is released in **Week 2**. See the [course page](../../index.html#assessment) for details.

Note:
Distinguish today’s ungraded practice from graded assignments. The course website is the source of current assessment rules. Do not import the Stanford workload or the previous semester’s deadlines. Connect small reproducible experiments to the individual project.

---

<!-- .slide: class="outline-slide" id="outline-development" -->

## Outline

<ul class="outline-topics">
<li>Course Overview</li>
<li aria-current="step">Development of NLP &amp; LLMs</li>
<li>Basics for Text Preprocessing</li>
<li>Text Tokenization</li>
</ul>

Note:
Period 1, minutes 25–45. Use about six minutes for the historical sketches and fourteen for prediction, training, and resource constraints.

---

<!-- .slide: id="development" -->

## Three approaches to language tasks

| Approach | Sentiment example |
| :--- | :--- |
| Rules | Count words such as “good” and “bad” |
| Statistical learning | Learn feature weights from labeled reviews |
| Neural language models | Learn representations; adapt or prompt the model |

These approaches can coexist inside an application.

Note:
This is a conceptual progression, not a claim that one method disappeared when another arrived. Revisit “light”: a bag of isolated sentiment words loses context. Background: Fudan Spring Lecture 01, development of NLP; Bengio et al. 2003, https://www.jmlr.org/papers/v3/bengio03a.html.

---

<!-- .slide: id="milestones" -->

## Selected milestones

| Work | Idea to remember |
| :--- | :--- |
| Neural language model, 2003 | Learn word representations and probabilities together |
| Transformer, 2017 | Attention-based sequence modeling |
| GPT-3, 2020 | Specify tasks using examples in the prompt |

We will study the mechanisms behind these ideas.

Note:
These are selected landmarks rather than a complete history. Bengio et al., A Neural Probabilistic Language Model: https://www.jmlr.org/papers/v3/bengio03a.html. Vaswani et al., Attention Is All You Need: https://arxiv.org/abs/1706.03762. Brown et al., Language Models are Few-Shot Learners: https://arxiv.org/abs/2005.14165. Avoid equating prompting with training: GPT-3 few-shot evaluation did not update model weights.

---

<!-- .slide: id="next-token" -->

## Next-token prediction

Prompt: “Fudan University is located in …”

The model assigns a probability to each possible **next token**.

$$p_\theta(t_1,\ldots,t_L)=\prod_{i=1}^{L}p_\theta(t_i\mid t_{<i})$$

Choose a token, append it, and repeat.

Note:
This is an autoregressive factorization, not an assumption that tokens are independent. “Shanghai” is an intended answer, but do not claim it is one token without inspecting the model tokenizer. The notebook’s further experiments inspect actual token log probabilities when supported. Background: CS336 Lecture 1, https://cs336.stanford.edu/lectures/?trace=lecture_01.

---

<!-- .slide: id="prediction-limits" -->

## Prediction and correctness

A plausible continuation can contain a factual error.

- A token probability measures a continuation under the model.
- A generated explanation is another model output.
- A correct-looking example does not establish task accuracy.

Use a separate set of examples with expected answers.

Note:
Ask what evidence would support a claim that sentiment classification improved: held-out labeled reviews, a fixed prompt, a specified metric, and comparable settings. Do not describe a self-reported confidence score or thinking trace as calibrated correctness. The optional log-probability experiment makes this distinction concrete.

---

<!-- .slide: id="training-pipeline" -->

## Building a language model

| Stage | A decision we must make |
| :--- | :--- |
| Data | Which text belongs in training and evaluation? |
| Tokenizer | Which pieces become vocabulary entries? |
| Model and training | Which architecture, objective, and compute budget? |
| Evaluation | Which tasks and failure cases matter? |

Today we build the tokenizer.

Note:
Adapted from CS336 Spring 2026 Lecture 1’s learn-by-building motivation and pipeline, https://cs336.stanford.edu/lectures/?trace=lecture_01. Explain that tokenizer training learns a vocabulary, while model training learns numerical parameters. Keep held-out text out of both training stages when measuring generalization.

---

<!-- .slide: id="resource-budget" -->

## Working within a compute budget

More data, a larger model, and longer sequences all use resources.

For today’s tokenizer, we can measure:

- vocabulary size;
- tokens needed for the same text;
- whether decoding recovers the original input.

We can test these choices on a laptop.

Note:
Source: CS336 Lecture 1’s emphasis on resource constraints and empirical experiments. Do not claim compression alone improves model quality. Pause for the first break after this slide. Next period begins with the actual representation of text.

---

<!-- .slide: class="outline-slide" id="outline-preprocessing" -->

## Outline

<ul class="outline-topics">
<li>Course Overview</li>
<li>Development of NLP &amp; LLMs</li>
<li aria-current="step">Basics for Text Preprocessing</li>
<li>Text Tokenization</li>
</ul>

Note:
Period 2, minutes 0–20: Unicode, UTF-8, and preserving the input. Python basics are prerequisites; this is not a full regex tutorial.

---

<!-- .slide: id="preprocessing" -->

## Preserve the text you intend to model

```python
text = "Hello,  世界!\n🙂"
print(repr(text))
print(repr(text.lower().strip()))
```

Spaces, punctuation, capitalization, and line breaks can carry information.

Choose normalization deliberately and document it.

Note:
There are two spaces after the comma and a newline before the emoji. Lowercasing changes H to h; strip only removes whitespace at the ends, not the internal spaces or newline. A lossless tokenizer should round-trip its input. A normalizing tokenizer may only recover the normalized input. Python Unicode HOWTO: https://docs.python.org/3/howto/unicode.html.

---

<!-- .slide: id="bytes" -->

## Code points and UTF-8 bytes

<div class="columns">
<div>
<h3>Different measurements</h3>
<p>A Unicode code point identifies a character value.</p>
<p>UTF-8 encodes that value using one or more bytes.</p>
</div>
<div>
<table>
<thead><tr><th>Text</th><th class="num">Code points</th><th class="num">Bytes</th></tr></thead>
<tbody>
<tr><td><code>hello</code></td><td class="num">5</td><td class="num">5</td></tr>
<tr><td><code>你好</code></td><td class="num">2</td><td class="num">6</td></tr>
<tr><td><code>🙂</code></td><td class="num">1</td><td class="num">4</td></tr>
</tbody>
</table>
</div>
</div>

<p class="source">These counts use the exact strings shown. A visible symbol can contain multiple code points.</p>

Note:
Python len counts code points for these strings. JavaScript string.length counts UTF-16 code units, so the live demonstration uses Array.from.

---

<!-- .slide: id="code" -->

## Measuring the same text in Python

```python
examples = ["hello", "你好", "🙂"]

for text in examples:
    code_points = len(text)
    utf8_bytes = len(text.encode("utf-8"))
    print(text, code_points, utf8_bytes)
```

Predict each line before running the cell in the notebook.

Note:
The notebook contains this exact example. Pause before showing the output.

---

<!-- .slide: id="browser-demo" -->

## Exploring the counts

<form class="demo-form" id="utf8-demo">
<label for="demo-text">Try English, Chinese, or emoji</label>
<input id="demo-text" type="text" value="你好🙂" maxlength="48" autocomplete="off" spellcheck="false">
<button type="button" id="demo-reset">Reset example</button>
</form>

<div class="demo-result" aria-live="polite">
<p><strong id="point-count">3</strong> Unicode code points</p>
<p><strong id="byte-count">10</strong> UTF-8 bytes</p>
</div>

<p class="caption">Try a space, a line of digits, or an accented letter. Explain each change.</p>

Note:
This demonstration runs entirely in the browser. Input stays on this page.
For a decomposed accent, compare the precomposed é with e followed by a combining acute accent.

---

<!-- .slide: class="exercise" id="exercise-02" -->

## Same appearance, different strings

<p class="exercise-meta">Exercise E02 · 3 minutes · Notebook E02</p>

Compare <code>"é"</code>, <code>"e\u0301"</code>, and <code>"你好🙂"</code>.

Predict the code-point and UTF-8 byte counts. Check the round trip.

<div class="answer fragment"><p>Counts: 1 / 2, 2 / 3, and 3 / 10.<br>The two accented strings look alike but are not equal.</p></div>

Note:
The second displayed string contains e followed by U+0301 COMBINING ACUTE ACCENT. Python len counts two code points; UTF-8 uses three bytes. NFC normalization makes the two accented strings equal but changes the original code-point sequence. Let students predict first, then run E02.

---

<!-- .slide: id="byte-roundtrip" -->

## A complete UTF-8 round trip

```python
text = "你好🙂"
ids = list(text.encode("utf-8"))
restored = bytes(ids).decode("utf-8")
assert restored == text
```

A token may contain only part of a UTF-8 character.

Join the token bytes **before** decoding the full text.

Note:
A byte has 256 possible values. A single byte from 你 is not valid UTF-8 by itself; strict decoding should fail instead of silently replacing it. The notebook demonstrates that failure and the successful full round trip. Source: Python Unicode HOWTO and CS336 Lecture 1 byte tokenizer.

---

<!-- .slide: class="outline-slide" id="outline-tokenization" -->

## Outline

<ul class="outline-topics">
<li>Course Overview</li>
<li>Development of NLP &amp; LLMs</li>
<li>Basics for Text Preprocessing</li>
<li aria-current="step">Text Tokenization</li>
</ul>

Note:
Period 2, minutes 20–45: representation choices and the first two merges. Reserve four minutes for E03. Continue with encoding and experiments after the break.

---

<!-- .slide: id="representation" -->

## Text representation

A tokenizer maps text to a sequence of token IDs.

The model reads those IDs through an embedding table.

> The tokenizer defines which pieces of text share a vocabulary entry.

Note:
Distinguish a tokenizer from the model that uses it. A token ID has meaning only within its tokenizer's vocabulary.

---

<!-- .slide: id="tokenizer-choices" -->

## Choosing the basic unit

| Unit | Benefit | Limitation |
| :--- | :--- | :--- |
| Word | Familiar units | Unseen words; segmentation choices |
| Code point | Simple character representation | Large alphabet; rare characters |
| Byte | Only 256 base entries | Long sequences |
| Learned subword | Frequent pieces become short | Depends on training data and rules |

Note:
Adapted from CS336 Lecture 1 character, byte, word, and BPE comparisons. A learned character vocabulary can have unknown characters; directly using Unicode scalar values avoids that but yields a sparse ID space. Chinese words are not reliably separated by spaces. BPE is one way to learn subwords, not the only algorithm.

---

<!-- .slide: id="bpe" -->

## Byte pair encoding

1. Begin with small units: bytes in our implementation.
2. Count adjacent token pairs in the training corpus.
3. Add the most frequent pair as one new vocabulary entry.
4. Replace its occurrences and repeat.

Save the **ordered merges** and the **vocabulary**.

Note:
Adapted from CS336 Lecture 1 and Sennrich et al. 2016 Section 3.2, ../../papers/sennrich-2016-subword-units.pdf. Our byte vocabulary differs from the character-based, word-boundary-aware algorithm in the paper. No merges cross our input-sequence boundaries. Stop when the merge budget is reached or no pair remains.

---

<!-- .slide: id="worked-example" -->

## A toy training corpus

| Initial sequence | Frequency |
| :--- | ---: |
| `l o w` | 5 |
| `l o w e r` | 2 |

The pairs `l o` and `o w` each occur 7 times.

<div class="fragment">
<p>Choose <code>l o</code> to break this tie. The next merge combines <code>lo w</code>.</p>
</div>

<p class="caption">Count pairs inside each word. Weight the counts by word frequency.</p>

Note:
This is a toy character-based example with no end-of-word symbol. State these choices explicitly.
Different tie-breaking choices can lead to different learned merges.

---

<!-- .slide: id="pair-counts" -->

## Count the pairs, including frequency

| Pair | In `low` × 5 | In `lower` × 2 | Total |
| :--- | ---: | ---: | ---: |
| `l o` | 5 | 2 | 7 |
| `o w` | 5 | 2 | 7 |
| `w e` | 0 | 2 | 2 |
| `e r` | 0 | 2 | 2 |

Tie rule: choose the smallest pair of token IDs.

Note:
ASCII base IDs are l=108, o=111, w=119, e=101, r=114. The maximum count is seven; (108,111) wins over (111,119). Break only the tied maximum, not all pairs. Deterministic ties make the lesson reproducible; production implementations may choose a different rule.

---

<!-- .slide: class="exercise" id="exercise-03" -->

## Counting overlaps; applying a merge

<p class="exercise-meta">Exercise E03 · 4 minutes · Notebook E03</p>

| Initial sequence | Frequency |
| :--- | ---: |
| `a a a b` | 2 |
| `a b` | 1 |

Which pair wins? How many tokens does one merge remove?

<div class="answer fragment"><p><code>a a</code>: 4 occurrences; <code>a b</code>: 3.<br>Replace left to right: <code>aa a b</code>. Total tokens: 10 → 8.</p></div>

Note:
One minute count, one minute propose replacements, two minutes check in the notebook. Pair counts include overlapping aa positions, but replacements cannot share an a. The winning count of four does not mean four replacements: only two across the weighted corpus. Expected E03 output is 10 then 8.

---

<!-- .slide: id="results" -->

## Sequence length after two merges

| Stage | Sequence for `low` | Sequence for `lower` | Total tokens |
| :--- | :--- | :--- | ---: |
| Initial | `l o w` | `l o w e r` | 25 |
| Merge `l o` | `lo w` | `lo w e r` | 18 |
| Merge `lo w` | `low` | `low e r` | 11 |

<p class="caption">Totals weight <code>low</code> by 5 and <code>lower</code> by 2.</p>

This example measures sequence length on the training corpus.

Note:
Verify the totals as 5*3+2*5, 5*2+2*4, and 5*1+2*3.
Ask what additional evidence would be needed to assess a tokenizer on unseen text.

---

<!-- .slide: id="interactive-results" -->

## Two merges: 25 → 11 tokens

<div class="plot" data-plotly="assets/sequence-length.json" role="img" aria-label="Toy corpus total tokens decrease from 25 to 18 to 11 after zero, one, and two merges."></div>

<p class="caption">The same toy corpus: <code>low</code> × 5 and <code>lower</code> × 2. Hover for exact counts.</p>

Note:
The chart loads automatically from a local Plotly specification. It plots the checked totals from the previous table, not model benchmark results. Drag to zoom; double-click to reset.

---

<!-- .slide: id="learned-vocabulary" -->

## The learned vocabulary

| Token ID | Stored bytes | Learned from |
| :--- | :--- | :--- |
| 0–255 | Each possible byte | Initial vocabulary |
| 256 | `b"lo"` | `(108, 111)` |
| 257 | `b"low"` | `(256, 119)` |

After two merges: **258 entries**.

The model would need an embedding for each entry.

Note:
The ASCII toy example is also a byte example, since each character here is one byte. Decoding uses this mapping, not the decimal digits in the token ID. IDs are specific to this tokenizer. Do not substitute these IDs into a pretrained Qwen model. Pause for the second break after this slide.

---

<!-- .slide: class="outline-slide" id="outline-tokenization-period-3" -->

## Outline

<ul class="outline-topics">
<li>Course Overview</li>
<li>Development of NLP &amp; LLMs</li>
<li>Basics for Text Preprocessing</li>
<li aria-current="step">Text Tokenization</li>
</ul>

Note:
Period 3, minutes 0–20: encoding and boundaries. Minutes 20–40: vocabulary tradeoffs and E05. Final five minutes: exit questions and reading.

---

<!-- .slide: id="training-encoding" -->

## Training and encoding

| Training | Encoding a new input |
| :--- | :--- |
| Count pairs in a training corpus | Begin with the input’s bytes |
| Learn the next merge | Apply the saved merges in order |
| Extend the vocabulary | Keep the vocabulary fixed |

Encoding a new prompt does **not** retrain the tokenizer.

Note:
Source: CS336 Lecture 1 BPETokenizer and train_bpe. Our simple implementation scans the whole sequence once per learned merge. Production code uses more efficient structures and may restrict merges with a pre-tokenizer. The conceptual distinction is essential for E04.

---

<!-- .slide: class="exercise" id="exercise-04" -->

## Encode a word we did not train on

<p class="exercise-meta">Exercise E04 · 5 minutes · Notebook E04</p>

Saved merges: **`l o` → `lo`**, then **`lo w` → `low`**.

Encode <code>"lowest"</code>. Then try <code>"你好🙂"</code>.

<div class="answer fragment"><p><code>lowest</code> → <code>[257, 101, 115, 116]</code>.<br>Chinese and emoji still round-trip through the base bytes.</p></div>

Note:
Two minutes trace lowest; three minutes run and inspect both examples. The pieces are low, e, s, t. The Chinese/emoji example has ten byte tokens here because neither learned ASCII pair occurs. No unknown token is needed for a valid UTF-8 string when all 256 base bytes are retained.

---

<!-- .slide: id="merge-order" -->

## Why the merge order matters

Suppose training learned:

1. `b c` → `bc`
2. `a b` → `ab`

Encoding <code>"abc"</code> gives <code>[a, bc]</code>.

It does not choose <code>[ab, c]</code> just because <code>ab</code> starts earlier.

Note:
An independently checkable example: training strings bc repeated three times and ab repeated twice learn bc before ab. At inference the earlier-ranked available merge wins. The notebook constructs this tokenizer and checks its output. Greedy longest-prefix matching is not a general substitute for BPE merge ranks.

---

<!-- .slide: id="boundaries" -->

## Where merges are allowed

Our toy corpus stores each string as a separate sequence.

Production tokenizers may first separate text using rules for spaces, letters, numbers, or punctuation.

> These boundaries can change the tokens, even when two implementations both use BPE.

Keep boundary rules fixed when comparing results.

Note:
Source: CS336 Lecture 1 notes on pre-tokenization and https://github.com/openai/tiktoken. Our learner has no within-string pre-tokenizer: it can learn pieces that include spaces, and it never merges across input strings. The toy low/lower table treats words as separate sequences for transparent counting. Do not describe that as the full production pipeline.

---

<!-- .slide: id="special-tokens" -->

## Text and control tokens

Chat systems also need to represent roles and message boundaries.

- Ordinary text is encoded using the text vocabulary.
- Special tokens can have reserved IDs and defined roles.
- The model’s chat template specifies the expected structure.

Typing a special token’s spelling is not always equivalent to inserting its ID.

Note:
Source: CS336 Lecture 1 special-token discussion and tiktoken README, https://github.com/openai/tiktoken. Our toy BPE only handles ordinary text. Do not invent Qwen chat-token IDs. The optional tiktoken cell uses encode_ordinary so user text is treated literally, including strings that resemble special tokens.

---

<!-- .slide: id="equation" -->

## Average sequence length

For a fixed collection of $N$ texts:

$$
\bar{L} = \frac{1}{N} \sum_{i=1}^{N} \left|\operatorname{encode}(x_i)\right|
$$

Use the same texts when comparing tokenizers.

Report the tokenizer version and any text normalization.

Note:
This is an average per text, so document length affects it. Explain why a comparison must hold the input collection fixed.

---

<!-- .slide: id="vocabulary-cost" -->

## Vocabulary size has a cost

An embedding table with vocabulary size $V$ and width $d$ has $Vd$ entries.

| Toy configuration, $d=1{,}024$ | Embedding entries |
| :--- | ---: |
| $V=32{,}000$ | 32,768,000 |
| $V=64{,}000$ | 65,536,000 |

A larger vocabulary may shorten sequences, but increases this table.

Note:
These are calculated configurations, not measurements of a named model. Values are parameters, not bytes; storage depends on dtype and optimizer state. An untied output head would have additional parameters. Source: CS336 Lecture 1 vocabulary/compression tradeoff.

---

<!-- .slide: id="sequence-cost" -->

## Sequence length also has a cost

For full self-attention, each position can interact with other positions.

$$1{,}000^2=1{,}000{,}000 \qquad 2{,}000^2=4{,}000{,}000$$

Doubling length gives four times as many position pairs.

This illustrates attention during training or prompt processing; total runtime depends on the implementation.

Note:
Toy count of all pairs. Causal attention masks future positions, but its pair count still grows quadratically. Efficient kernels need not materialize a full attention matrix. Cached generation has different per-step behavior, so do not say all inference costs scale as L squared. Source: Vaswani et al. 2017 Section 4, https://arxiv.org/abs/1706.03762.

---

<!-- .slide: class="exercise" id="exercise-05" -->

## Test on held-out English and Chinese

<p class="exercise-meta">Exercise E05 · 8 minutes · Notebook E05</p>

Train toy tokenizers with **0, 8, and 32 merges**.

Use the same held-out sentences each time. Report tokens and UTF-8 bytes per token separately for English and Chinese.

<div class="answer fragment"><p>Every input must round-trip. Compression depends on the corpus; a smaller token count alone does not establish a better language model.</p></div>

Note:
Two minutes inspect the training/held-out split, three minutes run the experiment, three minutes interpret. The notebook supplies small synthetic teaching corpora and an English-only training comparison. Record the actual number of learned merges, which can be below the budget if no pair remains. Ratios aggregate bytes and tokens within each language, not across an imbalanced mix.

---

<!-- .slide: id="heldout-results" -->

## Chinese text after different training

<div class="plot" data-plotly="assets/heldout-chinese.json" role="img" aria-label="On the same held-out Chinese text, mixed training reduces token counts from 54 to 48 to 35 at zero, eight, and thirty-two merges. English-only training leaves the count at 54."></div>

<p class="caption">Two fixed Chinese sentences · 54 UTF-8 bytes · Toy corpora from Notebook E05.</p>

Note:
These values are calculated with the exact training and held-out strings in notebook E05. With mixed English/Chinese training, counts are 54, 48, and 35; English-only training gives 54 throughout. English-only merges contain ASCII bytes that never occur in these Chinese strings. This small controlled demonstration is not a benchmark of real model tokenizers or language quality. Ask students to connect the plot to their own printed results.

---

<!-- .slide: id="comparison" -->

## What belongs in a tokenizer comparison?

| Keep fixed | Report |
| :--- | :--- |
| Input strings and normalization | Vocabulary and implementation |
| Training and held-out split | Token count by language |
| Treatment of spaces and boundaries | UTF-8 bytes per token |
| Ordinary versus special tokens | Exact round-trip results |

Measure downstream model quality separately.

Note:
The notebook also offers an optional comparison of the named cl100k_base and o200k_base encodings. Those are encoding names, not verified tokenizer identities for every commercial model. No unsupported claims about DeepSeek, Kimi, GLM, or Qwen versions belong in this lesson.

---

<!-- .slide: id="exit-questions" -->

## Before you leave

1. Why can one visible symbol require several tokens?
2. What changes during BPE training, and what stays fixed during encoding?
3. Does a shorter token sequence prove the model is more accurate?

Next: probabilities over token sequences and a first language-model baseline.

Note:
Use the final five minutes, including readings. Expected: visible symbols may have multiple code points and bytes; tokenizer vocabulary determines grouping. Training learns merges/vocabulary; encoding keeps them fixed. Compression alone gives no evidence about model accuracy. Ask students to finish any notebook discussion questions and the survey after class.

---

<!-- .slide: class="references" id="lecture-sources" -->

## Lecture sources and notebook extensions

- [Fudan Spring Lecture 01](https://baojian.github.io/llm-26/slides/lecture-01-slides/)<br>Language examples, applications, and the four-part outline.
- [Stanford CS336, Spring 2026, Lecture 1](https://cs336.stanford.edu/lectures/?trace=lecture_01)<br>Building language models, tokenizer baselines, and BPE.
- [Ollama API](https://docs.ollama.com/api/generate)<br>Notebook extensions: log probabilities, thinking, and vision.

The notebook contains the executable local-model demonstrations.

Note:
These sources support this adapted Fudan lecture. Stanford’s full lecture sequence and assignments are not requirements. The text experiments are core; the heavier extensions are available for interested students.

---

<!-- .slide: class="references" id="references" -->

## Readings for Lecture 01

- [Jurafsky and Martin, *Words and Tokens*](https://web.stanford.edu/~jurafsky/slp3/2.pdf)<br>Sections 2.3–2.4: Unicode and BPE (August 2026 draft).
- <a href="../../papers/sennrich-2016-subword-units.pdf" target="_blank" rel="noopener noreferrer">Sennrich, Haddow, and Birch (2016)</a><br>Section 3.2: BPE for subword segmentation.
- <a href="../../papers/kudo-2018-sentencepiece.pdf" target="_blank" rel="noopener noreferrer">Kudo and Richardson (2018), SentencePiece</a><br>Further reading on language-independent tokenization.

Note:
Read the textbook sections first, then trace the BPE example in Sennrich et al. Compare its word-boundary convention with our byte-based teaching implementation. SentencePiece is further reading, not an additional required implementation. Notebook references also link the Python Unicode HOWTO.
