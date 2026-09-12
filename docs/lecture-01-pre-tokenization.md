# Lecture 01: Text Preprocessing Before Tokenization

**Course:** NLP and LLMs, Fall 2026, Fudan University  
**Sources and tokenizer artifacts checked:** September 12, 2026  
**Scope:** Preparing text for tokenizer training, then applying the tokenizer
during language-model training and inference.

The most useful distinction is between **selecting and preparing a corpus**
and **transforming a string inside a tokenizer**. A production data pipeline
might extract articles, reject broken documents, deduplicate repositories,
and balance languages. Calling `tokenizer.encode(text)` does not repeat all
of that work. It applies the tokenizer's configured text transformations and
learned tokenization rules.

This note uses Kimi K3, GLM-5.2, and DeepSeek-V4 as concrete examples. Their
public artifacts reveal more about encoding than about the exact data used
to train their tokenizers. A language-model pretraining report is **not
automatically a recipe for training its tokenizer**.

Terminology: **text preprocessing** is the broad subject of this note;
**pre-tokenization** is the narrower step that establishes initial spans
before the learned subword model runs.

## Contents

1. [Three pipelines, three different jobs](#1-three-pipelines-three-different-jobs)
2. [What the named models actually disclose](#2-what-the-named-models-actually-disclose)
3. [A practical corpus-preparation workflow](#3-a-practical-corpus-preparation-workflow)
4. [The preprocessing inside a tokenizer](#4-the-preprocessing-inside-a-tokenizer)
5. [Which steps repeat during encoding?](#5-which-steps-repeat-during-encoding)
6. [Worked examples and common mistakes](#6-worked-examples-and-common-mistakes)
7. [Hands-on inspection and tokenizer training](#7-hands-on-inspection-and-tokenizer-training)
8. [Validation and classroom discussion](#8-validation-and-classroom-discussion)
9. [Source guide](#9-source-guide)

## 1. Three pipelines, three different jobs

### 1.1 Training the tokenizer

The tokenizer learner consumes strings, or statistics derived from strings,
and learns a vocabulary and segmentation rules. With ordinary BPE training,
this includes counting adjacent symbol pairs and learning merges. It does
not train the language model's neural-network weights.

```text
Raw sources
    → extract, repair, filter, deduplicate, select
    → representative tokenizer-training sample
    → configured normalization and pre-tokenization
    → learn vocabulary and segmentation rules
    → freeze and version the complete tokenizer
```

A representative sample can be much smaller than the eventual LLM corpus.
Its size, source mixture, and treatment of duplicates are design choices.
Do not assume that a report's trillion-token training budget is the amount
of text used to learn the tokenizer.

### 1.2 Training the language model

Once the tokenizer is fixed, it encodes the selected language-model corpus.
Training-example construction adds the required document boundaries,
formats, packing, attention masks, and loss masks. Some formatting happens
before encoding; some operations work on IDs afterward.

```text
Prepared language-model corpus
    → construct text/message formats where required
    → encode with the frozen tokenizer
    → construct training sequences and masks
    → train the language model
```

The tokenizer-training sample and the LLM-training corpus need compatible
text conventions, but they need not have identical sampling weights.

### 1.3 Running the language model

```text
User text, messages, files, or retrieved documents
    → application-specific extraction and message serialization
    → encode with the matching frozen tokenizer
    → model inference
```

For a plain string, extraction may be unnecessary. A PDF attachment needs a
document processor or a visual-input path. Chat messages need the model's
message format. These operations are not interchangeable with vocabulary
lookup.

The tokenizer itself commonly contains normalization, pre-tokenization,
the learned tokenization model, and post-processing. Libraries expose these
as separate components. [Hugging Face tokenization pipeline][hf-pipeline]

## 2. What the named models actually disclose

### 2.1 How to read the evidence

This note uses three evidence categories:

| Category | What it establishes | What it does not establish |
| --- | --- | --- |
| **Reported** | The model developers describe a training-data practice. | Every implementation detail, threshold, or tokenizer-training sample. |
| **Artifact-verified** | A particular released file contains a rule, or a local probe exhibits behavior. | Hidden upstream corpus cleaning or hosted API behavior. |
| **Recommended** | A practical design proposed for a course project or production pipeline. | That a named company uses this exact implementation. |

The exact releases inspected are pinned below. Following `main` later can
produce different files, so use the pinned links when reproducing results.

| Release | Inspected repository revision | Encoding evidence |
| --- | --- | --- |
| `moonshotai/Kimi-K3` | `f831ab66814297da540d832a5235f8e904f29d06` | [Tokenizer implementation][kimi-tokenizer], [BPE ranks][kimi-ranks], [configuration][kimi-config] |
| `zai-org/GLM-5.2` | `cf457fa734ab149ffef225f80893eb38c6ff5cdc` | [Tokenizer JSON][glm-tokenizer], [configuration][glm-config], [chat template][glm-template] |
| `deepseek-ai/DeepSeek-V4-Pro` | `b5968e9190ef611bbf34a7229255be88a0e937c1` | [Tokenizer JSON][ds-tokenizer], [configuration][ds-config], [message-encoding documentation][ds-encoding] |

### 2.2 Kimi K3

**Reported corpus preparation.** Section 3.1 describes web, code, mathematics,
and knowledge data, alongside a vision corpus. Text selection combines
heuristics, quality classifiers, and deduplication; domain sampling rates
come from smaller-model ablations. Knowledge and mathematics data also
undergo controlled rephrasing with varied prompts, chunk-wise generation,
and source-fidelity verification. Section 3.4 describes additional cleaning
of long inputs, including exact/fuzzy deduplication, quality filtering, and
structural validation. These are training-data operations, not steps that
`encode()` applies to every user prompt. [Kimi K3 report, Sections 3.1 and
3.4][kimi-report]

**Artifact-verified encoding.** The released implementation uses a
tiktoken BPE encoding. Its regex separates Han-character runs, handles
non-Han letters with case-sensitive branches, groups numbers with
`\p{N}{1,3}`, and preserves whitespace through explicit branches. The
ordinary-text path contains no lowercasing or Unicode-normalization call.
The chat path distinguishes structural markers from ordinary message
content when deciding whether special-token spellings receive special
IDs. Very long strings also pass through implementation-specific size
guards. [Kimi K3 tokenizer implementation][kimi-tokenizer]

**Not established by these sources:** the exact tokenizer-training sample,
its language weights, every upstream text repair, or a fully reproducible
corpus-cleaning configuration. The report's nominal 160K vocabulary size
alone does not prove that K3 and an earlier Kimi release have identical
token IDs or preprocessing.

### 2.3 GLM-5.2

**Release-specific evidence.** GLM-5.2 has its own released tokenizer files
and chat template. These are the basis for the encoding claims here. The
official GLM repository links a GLM-5 technical report; the GLM-5.2 release
materials checked do not provide a complete, separately reproducible
tokenizer-training or corpus-preprocessing recipe. [Official GLM
repository][glm-repo], [GLM-5.2 model card][glm-card]

**Earlier-family evidence, explicitly GLM-5.** Section 2.2 describes an
additional DCLM-style classifier and a world-knowledge classifier for web
selection. Code preparation includes fuzzy deduplication, corrected
Software Heritage metadata alignment, better language classification, and
quality-aware sampling. Math/science preparation improves web extraction
and PDF parsing, uses educational-quality scoring, and scores long
documents by combining chunk-level assessments. Its math/science pipeline
also filters synthetic and templated material. These are useful concrete
examples, but are not proof that GLM-5.2 uses identical rules or thresholds.
[GLM-5 report, Section 2.2][glm-report]

**Artifact-verified GLM-5.2 encoding.** `normalizer` is `null`. A regex split
runs before a `ByteLevel` conversion with `add_prefix_space=false` and
`use_regex=false`; the latter avoids applying a second default regex.
Digits are grouped into one to three characters. Its general letter branch
can include both Latin and Han letters. The wrapper configuration also sets
`do_lower_case=false`, `remove_space=false`, and
`clean_up_tokenization_spaces=false`. These settings describe the checked
encoding/decoding implementation, not the original web extractor.
[Tokenizer JSON][glm-tokenizer], [configuration][glm-config]

### 2.4 DeepSeek-V4

**Reported corpus preparation.** Section 4.1 describes filtering bulk
automatically generated and templated web content, expanding multilingual
coverage, and emphasizing long scientific and technical documents. Math
and code remain central. It says preprocessing largely follows V3 and that
V4 extends the V3 tokenizer with context-construction special tokens while
retaining a nominal 128K vocabulary. It also inherits token splitting and
fill-in-the-middle training, packs documents to reduce truncation, and
uses sample-level attention masking. These last operations belong to
training-example construction, not routine string cleaning. [DeepSeek-V4
report, Section 4.1][ds-report]

**Artifact-verified V4-Pro encoding.** The normalizer is an empty sequence,
so it applies no normalization. Pre-tokenization is a sequence: isolate
one-to-three-digit groups; isolate runs in specified Chinese/Japanese
character ranges; apply further letter, punctuation, and whitespace rules;
then convert through `ByteLevel` with no added prefix space or second
regex. The specified ranges are not equivalent to all Unicode CJK
characters. [DeepSeek-V4-Pro tokenizer JSON][ds-tokenizer]

**A training-time exception worth understanding.** V3's report describes
randomly splitting some tokens that combine punctuation and line breaks,
to reduce token-boundary bias. V4 explicitly inherits token splitting.
This exposes the LLM to alternative segmentations during training. It does
not mean inference should randomly modify punctuation or run a fresh BPE
learner. [DeepSeek-V3 report, Section 4.1][ds-v3-report], [DeepSeek-V4
report, Section 4.1][ds-report]

**Not established:** exact filtering thresholds, a complete tokenizer
training manifest, or identical behavior across every V4 variant and
hosted endpoint. The artifact measurements below apply to V4-Pro at the
pinned revision.

### 2.5 Direct comparison of the released text paths

| Property | Kimi K3 | GLM-5.2 | DeepSeek-V4-Pro |
| --- | --- | --- | --- |
| Learned encoding | tiktoken BPE | BPE | BPE |
| Ordinary-text normalization observed | No normalization call in inspected path | `null` normalizer | Empty normalizer sequence |
| Digit pre-tokenization | Groups of 1–3 | Groups of 1–3 | Groups of 1–3 |
| Han handling | Explicit Han-run branch | General letter branch can span scripts | Separate split for specified Chinese/Japanese ranges |
| Whitespace | Explicit regex branches | Regex, then byte conversion | Multiple splits, then byte conversion |
| Message formatting | Custom segmented chat encoding | Released Jinja template | Released message encoder |

Sources: [Kimi implementation][kimi-tokenizer], [GLM tokenizer][glm-tokenizer],
[GLM template][glm-template], [DeepSeek tokenizer][ds-tokenizer],
[DeepSeek message encoder][ds-encoding].

These are similar families of byte-based subword encoding, but **one
model's regex, vocabulary, or message template is not a substitute for
another's**.

## 3. A practical corpus-preparation workflow

The workflow below is a **recommended design**, not a reconstruction of
any one lab's private pipeline. Actual systems branch by source type and
may perform several filtering or deduplication passes. Cheap filters often
precede expensive model scoring; a later quality pass may choose which
member of a duplicate cluster to keep.

For runnable industrial examples, DataTrove exposes readers, extractors,
filters, deduplication stages, and writers, including a FineWeb pipeline.
Dolma similarly separates documents from derived annotations. These are
better implementation references than a single universal `clean_text()`
function. [DataTrove][datatrove], [Dolma data format][dolma-format]

### 3.1 Define the intended data and preserve provenance

Before processing, decide which languages, domains, and formats the model
must handle. A coding assistant, a multilingual chatbot, and a scientific
model should not automatically receive the same corpus mixture.

Keep a document record with at least:

- A stable ID and source location or source ID.
- Collection time and source snapshot/version.
- Declared or detected content type and character encoding.
- Source-specific metadata, such as repository path or document title.
- Applicable source-use metadata and any collection exclusions.
- Extraction/cleaning versions, quality scores, and rejection reasons.

Keep metadata separate from model-visible text unless the training format
deliberately includes it. Otherwise JSON keys, crawl timestamps, and storage
paths can accidentally influence the tokenizer's vocabulary.

A simple record shape is:

```json
{
  "id": "example-document-001",
  "source_type": "html_article",
  "text": "A paragraph extracted from an article.\n",
  "metadata": {
    "snapshot": "course-demo-2026-09",
    "language": "en",
    "extractor_version": "article-extractor-v1",
    "decision": "keep"
  }
}
```

Only the selected `text` field enters the simple tokenizer-training example
in Section 7. More elaborate formats should make additional fields explicit.

### 3.2 Decode source bytes correctly

Files arrive as bytes; most tokenizer APIs consume Unicode strings.

1. Identify the format and expected encoding from reliable source metadata.
2. Decode with an explicit policy; detect malformed input.
3. Quarantine or repair failures with recorded evidence.
4. Serialize the prepared corpus using a consistent encoding, commonly UTF-8.

Examples of problems to catch include a legacy encoding interpreted as
UTF-8, text decoded twice, HTML entities left as visible text, and binary
payloads mistakenly treated as prose.

Avoid treating `errors="ignore"` as a general repair strategy: it silently
deletes bytes. Replacement decoding can also insert `U+FFFD` characters.
Measure such replacements and retain the original source for diagnosis.

**Do not confuse file decoding with byte-level tokenization.** Correctly
decoding a file into text comes first. A byte-level tokenizer subsequently
represents that text through bytes, usually UTF-8 bytes. Byte-level BPE does
not recover information already lost during faulty file decoding.

### 3.3 Extract content according to its format

| Source | Useful extraction behavior | Typical failure |
| --- | --- | --- |
| HTML article | Parse the DOM, identify main content, preserve paragraphs and lists, decode entities once | Navigation and cookie banners overwhelm the article |
| HTML containing math | Preserve equation sources, such as LaTeX annotations, while extracting prose | Formulas vanish when tags are stripped |
| PDF or scanned page | Recover reading order, tables, formulas, and captions; use OCR when appropriate | Columns interleave or page headers repeat inside sentences |
| Source-code repository | Identify text files and languages; retain file boundaries, indentation, and relevant paths | Generated files dominate; tabs or line breaks are destroyed |
| Notebook | Parse cells deliberately and decide which outputs to retain | Images, traceback dumps, or execution metadata become training text |
| JSON or tool trace | Parse the schema and select intended fields; preserve message/tool relationships | Raw transport envelopes are mistaken for conversation content |

OpenWebMath is a concrete example of why generic web extraction is
insufficient: preserving mathematical notation is central to its dataset
construction. A formula is part of the content, even when its HTML
representation is not visible prose. [OpenWebMath paper][openwebmath]

Not every task should remove HTML. A model trained to write or understand
webpage source needs HTML examples. Route the document according to its
intended use rather than stripping every occurrence of `<...>`.

### 3.4 Repair extraction artifacts conservatively

Examples of useful, source-dependent repairs:

- Remove repeated page headers introduced by a PDF extractor.
- Rejoin a line-wrapped word when the source layout supports that decision.
- Standardize extractor-generated paragraph separators.
- Decode an HTML entity during HTML extraction.
- Exclude malformed records that contain truncated binary data.

Each repair changes the text distribution. Distinguish an artifact from
intentional source content. For example, `inter-\nnational` might be a PDF
line-wrap artifact, whereas a minus sign followed by a newline in code may
be meaningful. Never apply a global hyphen-removal rule to both.

For repeated application, test whether a deterministic repair is idempotent:
`repair(repair(text)) == repair(text)`. Idempotence helps, but it does not
prove semantic correctness. HTML unescaping twice, for instance, can change
literal examples that intentionally contain escaped entities.

### 3.5 Identify language, domain, and document structure

Language identification can be used for inclusion, routing, and sampling.
Domain labels help distinguish prose, code, math, reference material, and
conversation. Long-document scoring may need chunk-level assessment
because a short preview can misrepresent the whole document.

Do not use English-only heuristics indiscriminately. A rule requiring a
minimum number of whitespace-delimited words behaves differently on
Chinese. Code snippets and mixed-language pages may also receive unreliable
document-level language labels.

For a multilingual course corpus, retain the label confidence and inspect
uncertain or mixed-language samples. A language tag should be an auditable
decision, not a reason to translate all input into English.

### 3.6 Filter quality with several kinds of evidence

Use inexpensive signals to identify obvious problems, then apply more
expensive scoring where it adds value.

| Signal | What it can detect | What to inspect before filtering |
| --- | --- | --- |
| Empty or exceptionally short extraction | Failed extraction or navigation-only pages | Short but valid definitions, dialogue, and code |
| Repeated lines or n-grams | Boilerplate loops and templated pages | Tables, poems, lists, and repetitive algorithms |
| Excessively long lines | Minified assets, encoded payloads | Valid JSON, formulas, and generated-but-useful source |
| Unusual character distribution | Corruption or binary-like text | Non-Latin scripts, emoji, and mathematical notation |
| Language-model perplexity | Text unlike a chosen reference distribution | Specialist terminology and low-resource languages |
| Learned quality/educational score | Useful distinctions beyond simple heuristics | Classifier bias and out-of-domain failures |

Keep samples near the decision boundary and samples of rejected documents.
Measure retention separately by language and domain. A filter that improves
average English prose quality can still remove much of the code or Chinese
corpus.

FineWeb provides a public example combining extraction, language selection,
quality filters, and MinHash deduplication. Its published recipe is for a
particular web corpus, not a universal definition of good text.
[FineWeb dataset card][fineweb]

### 3.7 Deduplicate at the right level

Distinguish at least three cases:

1. **Exact duplicates:** identical document content, often found using hashes.
2. **Near duplicates:** largely overlapping documents with small edits,
   headers, or mirrors; shingle-based MinHash is one possible approach.
3. **Repeated passages:** shared paragraphs or boilerplate inside otherwise
   different documents.

Near-duplicate detection needs a similarity definition, candidate search,
threshold, and policy for choosing survivors. It is not synonymous with
semantic equivalence. Two explanations of the same theorem can be valuable
without being textual duplicates.

Keep a separate **comparison representation** when matching needs aggressive
canonicalization. Lowercasing a deduplication key does not require
lowercasing the text sent to the model. Avoid deduplicating code using a
representation that erases meaningful indentation or identifier case.

Deduplication can change tokenizer training because repeated content
changes pair frequencies. However, deleting every repeated word or common
sentence would also destroy useful frequency information. Deduplicate at a
document/passage level appropriate to the data, then choose deliberate
sampling weights. Dolma documents configurable deduplication over document
fields and paragraphs. [Dolma deduplication documentation][dolma-dedup]

### 3.8 Apply content-specific exclusions and evaluation isolation

For production datasets, define how to handle unwanted personal contact
details, credentials accidentally present in code, or other source-specific
excluded content. Depending on the record and purpose, remove the document,
mask a span, or retain a legitimate public example. A regex match alone is
not a complete interpretation of the content.

Dolma reports masking certain classes of personal information, illustrating
that these decisions can be part of corpus preparation. Such masking is
not an inherent property of BPE. [Dolma paper][dolma-paper]

Keep evaluation material separate from training data. Search for exact and
near overlaps using a documented matching representation. Split related
documents or repositories as groups where necessary, so near duplicates do
not land on both sides of a train/validation split.

For tokenizer experiments, hold out evaluation text from vocabulary learning
as well. Exposure in tokenizer training is different from training an LLM
on benchmark answers, but it still compromises a clean held-out comparison
of tokenizer compression or coverage.

### 3.9 Choose the tokenizer-training mixture

This is a frequently overlooked step. BPE allocates vocabulary capacity
according to the statistics it sees, subject to the chosen boundaries and
training algorithm. A large English-heavy sample can therefore give poor
compression on a small but important domain.

A practical procedure is to:

1. Define language/domain strata from the prepared corpus.
2. Select a reproducible sample within each stratum.
3. Cap extreme document lengths or sample chunks while retaining real
   whitespace and document structure.
4. Train candidate tokenizers with explicit mixture weights.
5. Compare held-out token counts and fidelity by stratum.
6. Adjust the mixture and vocabulary budget based on those results.

State the unit of weighting: documents, characters, UTF-8 bytes, or tokens
under a named reference tokenizer. Equal document counts do not imply equal
text volume. Equal byte counts do not imply equal linguistic coverage.

Do not claim that a tokenizer was trained on, for example, a 50/30/20 mixture
unless that mixture is documented. No exact tokenizer-training mixture is
asserted for the three model releases in this note.

### 3.10 Freeze the text contract and write reproducible shards

Record the chosen extraction rules, optional normalization, pre-tokenizer,
special-token policy, sample manifest, random seed, and library versions.
Use stable ordering or a recorded shuffle. Preserve record boundaries and
store outputs in restartable shards with counts and checksums.

Prefer streaming records to concatenating the entire corpus into a single
string. In JSONL, parse each JSON record and yield its actual `text` value;
do not accidentally train on JSON serialization syntax. Literal `\n` in a
JSON file becomes a newline after parsing.

The learner still needs memory for statistics, so a streaming iterator does
not make tokenizer training constant-memory. Measure unique pre-token
counts and pathological long spans before scaling up.

### 3.11 A publicly specified recipe: FineWeb

For comparison with the partially disclosed model pipelines, FineWeb's
dataset card specifies the following sequence:

1. Filter URLs using source exclusions.
2. Extract text with Trafilatura from Common Crawl WARC HTML.
3. Retain documents with a fastText English score of at least **0.65**.
4. Apply Gopher, selected C4, and additional FineWeb quality filters; omit
   C4's terminal-punctuation requirement.
5. Deduplicate each crawl separately using MinHash over **5-grams**, with
   **14 × 8** hash functions.
6. Mask email addresses and public IP addresses.

This is a concrete corpus recipe with inspectable settings. It does not
establish that Kimi, GLM, or DeepSeek use those settings, and its English
selection rule is unsuitable for a corpus intended to retain Chinese.
[FineWeb dataset card, data processing][fineweb]

## 4. The preprocessing inside a tokenizer

### 4.1 Normalization is a design choice

Unicode normalization makes particular Unicode representations consistent.
It is distinct from lowercasing, accent removal, language conversion, and
whitespace stripping.

| Choice | Example | Information to consider |
| --- | --- | --- |
| No normalization | Preserve `café` and `cafe\u0301` as different code-point sequences | Same visual word can tokenize differently |
| NFC | Compose `e` + combining acute accent into `é` where applicable | Changes code-point representation while preserving canonical equivalence |
| NFKC | Convert compatibility forms such as `Ａ` to `A` and `①` to `1` | Can erase distinctions relevant to notation or exact text reproduction |
| Lowercasing | `US` → `us` | Loses case, including distinctions in names and code |
| Whitespace collapsing | Four spaces → one space | Can destroy indentation, alignment, and formatting |

NFC and NFKC have different equivalence goals; neither means “remove every
unusual character.” [Unicode normalization specification][unicode]

For a new general-purpose or code-capable tokenizer, preserving input is a
reasonable baseline to test. For an existing model, follow its released
configuration. The inspected models in Section 2 do not justify adding an
extra NFKC or lowercasing pass.

SentencePiece offers a useful contrast: it normally uses an NFKC-based
normalization rule, supports custom rules, and stores the rule in the model.
Its `identity` rule disables that Unicode rewrite; whitespace-related
options still need inspection. “Uses SentencePiece” also does not specify
whether the learned model is BPE or Unigram. [SentencePiece normalization
documentation][sentencepiece-normalization], [SentencePiece options][sentencepiece-options]

### 4.2 Pre-tokenization defines candidate boundaries

A pre-tokenizer splits text into spans that a subsequent subword model
processes. With the BPE pipelines studied here, merges do not cross those
pre-token boundaries. A span may nevertheless become several final tokens.
[Hugging Face tokenization pipeline][hf-pipeline]

For example, the digit rule `\p{N}{1,3}` groups a run into chunks of at most
three Unicode numeric characters. It does not guarantee one final token
per chunk, and it does not mean “a complete number.” Decimal points, signs,
and separators have their own treatment.

These results were measured from the pinned artifacts:

| Input | Kimi K3 pre-token spans | GLM-5.2 pre-token spans | DeepSeek-V4-Pro pre-token spans |
| --- | --- | --- | --- |
| `2026 1234567` | `['202', '6', ' ', '123', '456', '7']` | Same | Same |
| `Hello世界ABC` | `['Hello', '世界', 'ABC']` | `['Hello世界ABC']` | `['Hello', '世界', 'ABC']` |
| `Hi.\nNext` | `['Hi', '.\n', 'Next']` | Same | Same |

The second example does not imply GLM emits one token for the entire string:
the measured final encoding has three IDs. For the first example, Kimi and
DeepSeek emit six IDs; GLM emits seven because its `'456'` chunk becomes
`'45'` and `'6'`. A pre-tokenization table and a final-token table answer
different questions. [Kimi ranks][kimi-ranks], [GLM tokenizer][glm-tokenizer],
[DeepSeek tokenizer][ds-tokenizer]

Python's built-in `re` does not implement the Unicode-property syntax used
in these patterns. The Kimi inspection below uses the `regex` package with
version-1 set semantics; GLM and DeepSeek use the pre-tokenizer saved in
their tokenizer JSON. Do not silently substitute an approximate regex and
call the result an exact reproduction.

External Chinese word segmentation is not required by these released
pipelines. Adding spaces from a word segmenter would change the input
distribution. A tokenizer may split within a word, a Chinese expression,
or an emoji sequence; token boundaries are not linguistic annotations.

### 4.3 Byte conversion preserves representability, not meaning

In byte-based BPE, a character can span multiple bytes and a token can cover
part of a character or multiple characters. Byte-to-symbol mappings used
internally by some implementations are reversible representations, not
Unicode cleanup.

A byte-level alphabet covering all 256 byte values can encode valid input
text without needing a whole-word entry for every word. That does not make
all scripts equally efficient: a rare character may consume several IDs.
Never assess multilingual quality only by whether an unknown token appears.

The JSON setting `byte_fallback=false` is not proof that a tokenizer lacks
byte coverage. A byte-level alphabet and a fallback mechanism for a
different base alphabet are separate designs.

### 4.4 Learn and freeze the tokenizer

Specify the vocabulary budget, alphabet, special tokens, and relevant
trainer settings. For BPE, useful controls include minimum pair frequency
and maximum learned-token length. Extremely repetitive material can
otherwise spend vocabulary capacity on long repeated strings. [Hugging
Face trainer API][hf-trainers]

Saving only the vocabulary is insufficient. Keep the merge rules or scores,
normalizer, pre-tokenizer, decoder, added-token definitions, wrapper
configuration, and message-format implementation as applicable. Record
their versions together with the model checkpoint.

Also distinguish vocabulary accounting conventions:

- Kimi's checked rank file has **163,584** mergeable entries; its wrapper
  reserves another **256** IDs.
- GLM's checked JSON has **154,820** model-vocabulary entries and **154,856**
  distinct IDs after added tokens; its model configuration has **154,880**
  vocabulary slots.
- DeepSeek's checked JSON has **128,000** model-vocabulary entries and
  **129,280** distinct IDs after added tokens. Some added-token records
  overlap existing IDs, so simply adding record counts is wrong.

These are artifact counts, not alternate estimates of corpus size.
[Kimi ranks][kimi-ranks], [Kimi implementation][kimi-tokenizer],
[GLM tokenizer][glm-tokenizer], [GLM model configuration][glm-model-config],
[DeepSeek tokenizer][ds-tokenizer]

## 5. Which steps repeat during encoding?

### 5.1 The practical answer

| Operation | Before tokenizer learning | During LLM training | During ordinary inference |
| --- | --- | --- | --- |
| HTML/PDF extraction | If sources require it | If sources require it | If the application accepts those sources |
| Corpus quality filtering | Usually part of sample selection | Part of corpus selection | Not an operation performed by the tokenizer |
| Corpus deduplication and mixture sampling | Design choice for the learning sample | Design choice for the LLM corpus | Not performed on each prompt by the tokenizer |
| Required deterministic text preparation | Apply the chosen input contract | Use a compatible contract | Apply if part of the application's input contract |
| Tokenizer normalization | Applied as configured | Applied as configured | Applied as configured |
| Tokenizer pre-tokenization | Applied as configured | Applied as configured | Applied as configured |
| Learning BPE merges or Unigram parameters | Yes | Normally no: tokenizer is already frozen | No |
| Chat/tool serialization | Only if deliberately included in the learning sample | Required for corresponding training examples | Required for the corresponding model interface |
| Training augmentation or alternative segmentation | Only if part of the tokenizer experiment | May be used | Usually disabled unless explicitly requested |
| Packing, padding, attention masks, loss masks | Not vocabulary-learning cleanup | Constructed for training | Batching/context handling may apply; training loss masks do not |

“Same preprocessing” therefore means **compatible model-visible text and
the matching tokenizer rules**, not rerunning the entire data factory on
every message.

External preprocessing is not automatically saved inside a tokenizer. If
you lowercase a corpus in a separate script but configure no lowercase
normalizer, the saved tokenizer will not start lowercasing future inputs.
Either encode the required transformation in a supported tokenizer
component or maintain and version it as part of the application pipeline.

### 5.2 Serialization is part of the model interface

A list of messages is not yet the model's input sequence. Roles, turn
boundaries, tool schemas, and generation prefixes must be represented in
the format the model expects. Applying a chat template changes the input
even when the underlying content strings remain untouched.

With a supported Hugging Face chat template, tokenizing directly through
`apply_chat_template(..., tokenize=True)` avoids a common double-insertion
mistake. If a complete template is rendered to a string first, subsequent
encoding normally uses `add_special_tokens=False` so boundary tokens are
not added twice. Follow model-specific instructions when a custom encoder
replaces this generic path. [Hugging Face chat templates][hf-chat]

For example, DeepSeek publishes a message encoder that handles chat,
reasoning, and tool-call structure. That encoder constructs a prompt; it is
not a web-corpus cleaner. GLM ships a Jinja template, while Kimi uses a
custom segmented message path. [DeepSeek encoding documentation][ds-encoding],
[GLM template][glm-template], [Kimi tokenizer][kimi-tokenizer]

Special-token matching is another reason a simple four-box tokenizer
diagram is incomplete. A library may recognize added tokens outside the
ordinary BPE path. A literal marker typed inside message content must be
handled according to the model's serializer and token-matching policy.
It is not safe to assume that arbitrary text concatenation reproduces the
official message path.

### 5.3 Deterministic encoding has explicit exceptions

For a fixed ordinary-text input, tokenizer artifact, runtime, and options,
normal inference is deterministic. Exceptions include explicitly enabled
sampling or BPE dropout, training-only alternative segmentations, and
implementation-specific handling of very long strings.

Even without randomness, encoding pieces separately can change IDs:

```text
encode("hello world")
may differ from
encode("hel") + encode("lo world")
```

Whether they differ depends on the boundary and vocabulary. Arbitrarily
splitting a long file every N characters can therefore preserve the
decoded text while changing its segmentation. Use the official runtime's
behavior when matching an existing model.

## 6. Worked examples and common mistakes

### 6.1 Web article with an equation

Suppose an HTML page contains navigation, an article, a cookie banner, and
an equation stored in a math annotation.

```text
Source HTML
    → parse and select the article
    → preserve the equation as an explicit text representation
    → retain paragraph boundaries
    → quality assessment and deduplication
    → optional inclusion in the tokenizer-training sample
```

Later, if a user types the already extracted article into a chat box, the
tokenizer does not need to run the HTML extractor. If a retrieval system
fetches the HTML page, that system does need a compatible extraction path.

Do not remove all punctuation to make the article “cleaner”: `x != y`,
`x = y`, and `x < y` are different statements.

### 6.2 A Python function

```python
def sign(x):
    if x > 0:
        return 1
    return -1
```

Preserve indentation, identifiers, operators, comments, and line endings
according to the chosen source policy. Do not lowercase identifiers or
strip punctuation. Deciding whether to exclude a generated dependency
directory is corpus selection; preserving the indentation of a retained
file is a text-fidelity requirement.

For code infilling, a training pipeline may rearrange a document into
prefix/suffix/middle sections with special markers. That is a training
task format. It does not mean ordinary encoding rearranges every code
snippet.

### 6.3 Two visually similar Unicode strings

`"café"` and `"cafe\u0301"` can look alike while containing different
code-point sequences. An identity-preserving tokenizer may assign
different IDs. An NFC normalizer can make them identical before
segmentation. Neither behavior is automatically a bug.

The checked tokenizer paths preserve both forms on the tested inputs. A
downstream `unicodedata.normalize("NFKC", text)` or `text.lower()` call
would create an additional application-level transformation not justified
by those artifacts.

### 6.4 An incomplete prompt ending at punctuation

`"Answer:\n"` and `"Answer:"` need not have token sequences that are prefixes
of each other. A learned token may contain both the colon and newline.
The difference matters when comparing few-shot prompt formats or caching
prefixes. This is the kind of boundary sensitivity behind the DeepSeek-V3
training intervention discussed in Section 2.4; it is not evidence that
newlines should be deleted. [DeepSeek-V3 report][ds-v3-report]

### 6.5 Transformations to avoid as universal defaults

| Tempting default | Why it can be harmful | Better decision |
| --- | --- | --- |
| Remove stopwords | Deletes negation, relations, and ordinary syntax | Keep natural text for general LLM training |
| Stem or lemmatize everything | Changes the text the model must reproduce | Use only for a task explicitly defined over that representation |
| Lowercase everything | Loses case-sensitive meaning and code identifiers | Follow the target tokenizer; test a normalization choice when training a new one |
| Remove punctuation or digits | Destroys code, math, dates, and quantities | Preserve content; filter malformed documents separately |
| Collapse all whitespace | Destroys indentation, tables, and intentional line breaks | Use format-specific repairs |
| Apply NFKC without evaluation | Can collapse distinct symbols or text forms | Choose normalization explicitly and measure its consequences |
| Segment Chinese into words first | Adds boundaries absent from the released model pipeline | Use the model's own pre-tokenizer |
| Translate all languages into English | Changes both the corpus and target capabilities | Select and balance languages deliberately |
| Always rewrite text with an LLM | Can change facts, style, and frequency statistics | Treat rewriting as a separately validated augmentation |

## 7. Hands-on inspection and tokenizer training

These examples require no model weights or GPU. Put the snippets in
`workspace/`, which is ignored by Git. They are inspection and teaching
examples, not full replacements for the models' message processors.

### 7.1 Inspect real artifacts without executing model repository code

Save the following as `workspace/inspect_tokenizer_preprocessing.py`.
Run it from the repository root:

```bash
uv run --with tokenizers==0.23.2 --with tiktoken==0.14.0 --with regex==2026.9.3 python workspace/inspect_tokenizer_preprocessing.py
```

The first run downloads only tokenizer JSON, BPE ranks, and one Python
source file for static inspection. The source is parsed as an AST; it is
not imported or executed. Kimi's reconstructed encoding below handles
ordinary text without special markers. It does not implement Kimi chat,
multimodal processing, or the wrapper's long-input guards.

```python
import ast
import base64
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import regex
import tiktoken
from tokenizers import Tokenizer


ROOT = Path("workspace/tokenizer-artifacts")
ROOT.mkdir(parents=True, exist_ok=True)
RELEASES = {
    "kimi": ("moonshotai/Kimi-K3", "f831ab66814297da540d832a5235f8e904f29d06"),
    "glm": ("zai-org/GLM-5.2", "cf457fa734ab149ffef225f80893eb38c6ff5cdc"),
    "deepseek": ("deepseek-ai/DeepSeek-V4-Pro", "b5968e9190ef611bbf34a7229255be88a0e937c1"),
}
SAMPLES = [
    "2026 1234567",
    "Hello世界ABC",
    "Hello WORLD!",
    "ＡＢＣ ① café cafe\u0301",
    "if x:\n    print(x)\n",
    "Hi.\nNext",
    "👩\u200d💻",
]


def fetch(name, filename):
    repo, revision = RELEASES[name]
    path = ROOT / f"{name}-{revision}-{filename}"
    if not path.exists():
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
        with urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print("artifact", path.name, "sha256", digest)
    return path


for name in ("glm", "deepseek"):
    path = fetch(name, "tokenizer.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(path))
    print(name, "normalizer", config["normalizer"])
    print(name, "pre_tokenizer", config["pre_tokenizer"])
    for text in SAMPLES:
        # ByteLevel displays byte-mapped symbols internally. Recover the
        # source spans using offsets for these particular probe strings.
        spans = tokenizer.pre_tokenizer.pre_tokenize_str(text)
        chunks = [text[start:end] for _, (start, end) in spans]
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        restored = tokenizer.decode(ids, skip_special_tokens=False)
        assert restored == text, (name, repr(text), repr(restored))
        print(name, repr(text), "chunks", chunks, "ids", ids)


source = fetch("kimi", "tokenization_kimi.py").read_text(encoding="utf-8")
tree = ast.parse(source)
assignment = next(
    node for node in ast.walk(tree)
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "pat_str"
            for target in node.targets)
)
pattern = "|".join(ast.literal_eval(assignment.value.args[0]))
splitter = regex.compile(pattern, regex.VERSION1)
rank_lines = fetch("kimi", "tiktoken.model").read_bytes().splitlines()
ranks = {
    base64.b64decode(token): int(rank)
    for token, rank in (line.split() for line in rank_lines)
}
encoding = tiktoken.Encoding(
    name="kimi-k3-ordinary-text-inspection",
    pat_str=pattern,
    mergeable_ranks=ranks,
    special_tokens={},
)
for text in SAMPLES:
    chunks = splitter.findall(text)
    assert "".join(chunks) == text
    ids = encoding.encode_ordinary(text)
    assert encoding.decode(ids) == text
    print("kimi", repr(text), "chunks", chunks, "ids", ids)
```

On the checked artifacts and package versions, all seven ordinary-text
samples round-trip exactly for all three encodings. This is a small probe,
not proof about arbitrary byte strings, all Unicode edge cases, chat
templates, or hosted services. For the combining-accent example, GLM's
pre-tokenizer separates the final combining mark, while the Kimi and
DeepSeek pre-tokenizers keep it with the preceding letters. All preserve
the resulting text.

### 7.2 Train a small illustrative byte-level tokenizer

This example shows where preprocessing belongs in the training API. It
uses a **teaching configuration**, not the regex or trainer settings of
Kimi, GLM, or DeepSeek. The small vocabulary is for demonstrating the
mechanics, not for measuring production quality.

Save it as `workspace/train_demo_tokenizer.py`, then run:

```bash
uv run --with tokenizers==0.23.2 python workspace/train_demo_tokenizer.py
```

```python
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers


samples = [
    "Hello world!\n",
    "Hello世界ABC\n",
    "Numbers: 2026 and 1234567.\n",
    "def sign(x):\n    return 1 if x > 0 else -1\n",
    "ＡＢＣ ① café cafe\u0301 👩\u200d💻\n",
]
tokenizer = Tokenizer(models.BPE())
# No normalizer: retain the input code-point sequence.
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(Regex(r"\p{N}{1,3}"), behavior="isolated"),
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
])
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=400,
    min_frequency=2,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<eos>"],
    max_token_length=24,
    show_progress=False,
)
tokenizer.train_from_iterator(samples, trainer=trainer)
path = Path("workspace/demo-tokenizer.json")
path.parent.mkdir(parents=True, exist_ok=True)
tokenizer.save(str(path))
loaded = Tokenizer.from_file(str(path))

for text in samples + ["Unseen text: naïve, 中文, 987654321.\n"]:
    before = tokenizer.encode(text, add_special_tokens=False).ids
    after = loaded.encode(text, add_special_tokens=False).ids
    assert before == after
    assert loaded.decode(after, skip_special_tokens=False) == text
    print(repr(text), after)
```

Here the configured pre-tokenizer runs both while learning and while
encoding. The 256-byte initial alphabet supports unseen characters by
encoding their bytes. Registering `<eos>` reserves a special token; it does
not append that token to every record. The requested vocabulary size is a
budget: a tiny corpus may not provide enough eligible merges to reach it.
[Hugging Face trainer API][hf-trainers]

To use a prepared JSONL corpus, replace `samples` at the training call with
an iterator that parses each record and yields `record["text"]`. Perform
corpus selection upstream. Do not turn the full cleaning workflow into a
normalizer that runs every time a user asks a question.

## 8. Validation and classroom discussion

### 8.1 What to measure before scaling up

| Check | Why it matters |
| --- | --- |
| Extraction examples before and after processing | Detects lost equations, tables, code, and reading order |
| Retained documents/bytes by source, language, and domain | Reveals unintended changes in mixture |
| Reasons for rejection and samples of rejected text | Makes quality filters inspectable |
| Duplicate-cluster sizes and survivor policy | Detects over-aggressive matching and dominant mirrors |
| Train/validation overlap after preprocessing | Checks whether the held-out comparison is meaningful |
| Token counts by language and domain | Finds poor compression hidden by the global average |
| Round-trip text comparisons | Detects unintended loss or transformations |
| IDs before and after save/load | Checks that the saved artifact reproduces the configured encoding |
| Leading spaces, tabs, CRLF/LF, blank lines, combining marks, emoji | Exercises formatting and Unicode edge cases |
| Literal marker strings and real chat/tool examples | Checks the boundary between content and control tokens |
| Very long spans and documents | Exposes runtime guards, memory growth, and truncation behavior |

For token-count comparisons, report the denominator explicitly: tokens per
UTF-8 byte, tokens per code point, or tokens per document. Cross-language
comparisons benefit from parallel or otherwise matched content. A single
English “tokens per word” ratio is not a multilingual metric.

For a tokenizer with intentional normalization, the expected round trip may
be the normalized text rather than the original string. Some decoders also
alter whitespace or omit special tokens. Define the expected contract
before deciding that a round trip passed or failed.

### 8.2 Questions to answer before choosing a pipeline

1. Which transformations select documents, and which change the text inside
   a retained document?
2. Which rules must be shared at encoding time, and which are only for
   training-data selection or augmentation?
3. Does the tokenizer-training mixture cover Chinese, English, code, math,
   and the actual deployment domains adequately?
4. Could a cleanup rule erase information needed to reproduce the input?
5. Is a model-specific claim supported by that release's artifact, by a
   report, or only by an older model in the family?
6. Can another student reproduce the tokenizer from the saved artifacts
   and manifest without guessing hidden preprocessing steps?

For an exercise, predict the pre-token spans of `Hello世界ABC`,
`2026 1234567`, and `Hi.\nNext` before running Section 7. Then explain why
matching pre-token spans need not imply matching token IDs or token counts.

## 9. Source guide

The links attached to claims above are primary sources. Reports describe
training practices; pinned artifacts support encoding claims; the workflow
and classroom design recommendations are identified separately.

| Source | Read for |
| --- | --- |
| [Kimi K3 technical report][kimi-report] | Sections 3.1 and 3.4: corpus selection, rephrasing, and long-input preparation |
| [GLM-5 technical report][glm-report] | Section 2.2: web classifiers, code metadata, math/science extraction; earlier-family evidence |
| [DeepSeek-V4 technical report][ds-report] | Section 4.1: data construction and inherited training strategies |
| [DeepSeek-V3 technical report][ds-v3-report] | Section 4.1: punctuation/newline token splitting and FIM |
| [FineWeb dataset card][fineweb] | An openly described web-corpus processing recipe |
| [DataTrove][datatrove] | Executable corpus-processing building blocks and examples |
| [Dolma paper][dolma-paper] | Public data-curation decisions and their evaluation |
| [OpenWebMath][openwebmath] | Why mathematical content needs specialized extraction |
| [Unicode normalization specification][unicode] | Canonical versus compatibility normalization |
| [SentencePiece normalization][sentencepiece-normalization] | Normalization rules stored with a tokenizer model |
| [Hugging Face tokenizer pipeline][hf-pipeline] | Normalizer, pre-tokenizer, learned model, and post-processor |
| [Hugging Face chat templates][hf-chat] | Message serialization and avoiding duplicate special tokens |

[kimi-report]: https://arxiv.org/html/2607.24653v1
[kimi-tokenizer]: https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/tokenization_kimi.py
[kimi-ranks]: https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/tiktoken.model
[kimi-config]: https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/tokenizer_config.json
[glm-repo]: https://github.com/zai-org/GLM-5
[glm-card]: https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/README.md
[glm-report]: https://arxiv.org/html/2602.15763v2
[glm-tokenizer]: https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/tokenizer.json
[glm-config]: https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/tokenizer_config.json
[glm-template]: https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/chat_template.jinja
[glm-model-config]: https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/config.json
[ds-report]: https://arxiv.org/html/2606.19348v1
[ds-v3-report]: https://arxiv.org/html/2412.19437v2
[ds-tokenizer]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/tokenizer.json
[ds-config]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/tokenizer_config.json
[ds-encoding]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/encoding/README.md
[fineweb]: https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/main/README.md
[datatrove]: https://github.com/huggingface/datatrove
[dolma-format]: https://github.com/allenai/dolma/blob/main/docs/data-format.md
[dolma-dedup]: https://github.com/allenai/dolma/blob/main/docs/deduplication.md
[dolma-paper]: https://arxiv.org/abs/2402.00159
[openwebmath]: https://arxiv.org/abs/2310.06786
[unicode]: https://www.unicode.org/reports/tr15/
[sentencepiece-normalization]: https://github.com/google/sentencepiece/blob/master/doc/normalization.md
[sentencepiece-options]: https://github.com/google/sentencepiece/blob/master/doc/options.md
[hf-pipeline]: https://huggingface.co/docs/tokenizers/main/en/pipeline
[hf-trainers]: https://huggingface.co/docs/tokenizers/api/trainers
[hf-chat]: https://huggingface.co/docs/transformers/en/chat_templating
