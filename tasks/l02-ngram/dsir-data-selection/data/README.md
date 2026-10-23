# DSIR teaching data

Version: `lecture02-dsir-v2`. Generator seed: `20261003`.

Original synthetic English snippets, generated offline. Each chunk has **128
supplied tokens**. IDs and exact token sequences are distinct across all splits.

| Split | Chunks | Tokens |
| --- | ---: | ---: |
| Raw fitting | 512 | 65,536 |
| Target fitting | 256 | 32,768 |
| Development | 128 | 16,384 |
| Final evaluation | 256 | 32,768 |
| Candidate pool | 2,048 | 262,144 |
| Total | 3,200 | 409,600 |

## How the text is generated

Each chunk contains 32 four-token clauses from an original vocabulary.
A primary topic is sampled with these probabilities:

| Topic | Raw/pool | Target/development/final |
| --- | ---: | ---: |
| Daily activities | 0.52 | 0.15 |
| Physics | 0.12 | 0.28 |
| Ecology | 0.10 | 0.28 |
| Algebra | 0.10 | 0.10 |
| Chemistry | 0.08 | 0.12 |
| Software | 0.08 | 0.07 |

Styles are sampled independently: 60% focused, 30% mixed, 10% boilerplate-heavy.
Ordinary clauses use the daily-activities vocabulary with probability 0.25.
Mixed chunks then replace a clause's vocabulary with another topic's vocabulary
with probability 0.4. In boilerplate-heavy chunks, each clause has probability
0.65 of being a shared notice such as “please read the notes”; other clauses
follow the ordinary rule. These are sampling probabilities, not exact quotas.

Domain and subtopic labels describe the primary topic, not every token.
The runner reports these labels and styles for analysis; it passes only tokens
and IDs to `solve`. Longer chunks can concentrate importance weights. Mixed
content and boilerplate make primary-topic counts an incomplete quality measure.

## Files and experiment

- `examples.json`: three small prediction inputs, without answers.
- `checks.json`: separate public correctness cases with expected outputs.
- `corpus.json`: splits, generation/experiment settings, and hash collisions
  computed from fitting/candidate text only.
- `baselines.json`: uniform, target-only, and greedy ratio selections at 16/256
  buckets, `alpha = 1`, and **64 documents / 8,192 tokens**. Sampling uses seeds
  0–4; greedy selection runs once per hash size. Uniforms are keyed by document ID.

Choose the hash size on development text. The final command requires its report
and matching submission/data hashes. Both hash sizes use the same unhashed
Jensen–Shannon diagnostic in nats. Report unfavorable results as well as favorable
ones; this experiment does not train a language model.

The private generators are `tasks/l02-ngram/build_data.py` and `build_checks.py`.
V1 used 544 chunks of 32 tokens across four topics and a 32-document selection
budget. Its recorded results remain historical examples in the instructor notes.
