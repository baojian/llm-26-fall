# grounded-lookup-v2

Course-authored synthetic records, questions, demonstrations, and preference
pairs. License: **CC0-1.0**. Generation seed: **20261111**, with separate seeded
streams for train/dev/test. No external text, student records, or human ratings
are used. The model checkpoint has its own license.

## Task

Read a record of four to six fictional items. Each item lists some of its
color, shape, material, and size. Answer the question with exactly one value,
or `unknown` if the item or requested attribute is missing. Do not add prose,
punctuation, or explanations to the model's answer.

There are eight values in each of four attribute domains (32 values total).
The complete answer vocabulary is in `manifest.json`. All preferred/rejected
responses are one GPT-2 token plus EOS under the pinned tokenizer; input length
is checked separately and never silently truncated.

## Fixed inventory

| File | Independent families | Rows |
| --- | ---: | ---: |
| train.jsonl | 4,096 | 8,192 |
| dev.jsonl | 128 | 256 |
| test.jsonl | 128 | 256 |

Each family contains one record/query and two question phrasings with shuffled
record and field order. Keep both rows in one split. The split is by source
family; lexical vocabulary and question templates intentionally overlap.

Each split balances the four requested attributes. Exactly 75% of rows have an
answer, 12.5% omit the queried item, and 12.5% omit its requested attribute.
Present and missing item IDs use the same random range. Rejected answers cover
wrong item values, false abstention, and unsupported answers. Both preference
sides use identical formatting. `unknown` is preferred in 25% and rejected in
25% of rows, controlling that particular unconditional answer shortcut.
Other vocabulary/template shortcuts may remain: report them as limitations.

## Schema and stage use

Each JSONL row contains:
- `id`: unique example ID; `family`: the unit for split membership/bootstrap.
- `prompt`, `chosen`, `rejected`: demonstration and preference inputs.
- `target`: the independently determined exact answer.
- `slice`: present/absent; `condition`: present/missing_object/missing_attribute.
- `attribute`, `query`, `facts`, `template`, `negative_kind`: auditable construction metadata.

SFT uses train prompt/chosen pairs. Reward fitting uses train prompt/chosen/
rejected triples. PPO uses train prompts and the frozen learned reward model;
it must not use `facts`, `target`, or the independent checker for rewards.
No evaluation row is used in any training stage.

Use development data for debugging. Freeze the configuration before final
test evaluation. Public test labels are a protocol boundary, not a hidden
benchmark. Report exact correctness and valid-answer rate independently from
the learned reward. Bootstrap paired differences by family, not by row.

The manifest contains file SHA-256 hashes, source-generator hash, seed, counts,
condition/attribute/target distributions, and answer vocabulary. The runner
verifies file hashes before use. The instructor retains the deterministic
generator and semantic audit tests; students receive the fixed dataset.
