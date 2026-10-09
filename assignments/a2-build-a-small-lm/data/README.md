# A2 TinyStories teaching subset

Source: Ronen Eldan and Yuanzhi Li's [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
specifically the GPT-4 V2 text files. The upstream dataset card declares
[CDLA-Sharing-1.0](CDLA-Sharing-1.0.txt); these source-derived text records are
distributed under the same terms. The stories are synthetic, not student data.

The course's A1 corpus snapshot already contains document provenance and source
hashes. A2 selects 2,048 complete training documents and 128 each for development
and final evaluation, using SHA-256 ranks with seed 20261009. `manifest.json`
pins the A1 input files, source repository revision, and all delivered artifacts.
The original mirror did not pin an upstream Git revision; content hashes are
the version authority. Each JSONL record retains its upstream file and document
index. A1's train comes from official training data; its disjoint development
and test files come from the official validation file. A2 preserves those splits.

Exact families are disjoint across splits before any repetition is injected.
Near-duplicate or semantic overlap is not ruled out. The tokenizer is byte-level
BPE with 4,096 tokens, fitted only on the selected unique training documents;
its four special IDs are UNK=0, BOS=1, EOS=2, PAD=3. The initial byte alphabet
avoids unknown ordinary UTF-8 byte sequences. The same tokenizer is frozen for
all conditions. Students use the supplied tokenizer rather than fitting another.

After splitting and tokenizer fitting, three extra copies of each of the first
256 ID-sorted training documents are injected. Thus `train.jsonl` contains 2,816
records representing 2,048 distinct documents. Repeated IDs have `-repeat-N`
suffixes. The experiment measures the effect of this controlled construction;
it does not establish natural web duplication rates.

Documents remain intact in the JSONL files. The driver adds BOS/EOS and packs
them in ID order; training windows may cross EOS/BOS boundaries. Training uses
full windows and no padding. Evaluation uses nonoverlapping context blocks,
scores every next token including the final short block, and aggregates loss
by token count. See the handout for the budget and final-evaluation procedure.

Do not modify these files. The driver verifies artifact checksums before use.
Preparation code and instructor results remain in the private repository;
all data and tokenizer artifacts needed for student commands are included here.
