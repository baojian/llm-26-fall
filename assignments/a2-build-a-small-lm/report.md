# A2 report

Target 600–900 words excluding tables and the disclosure. Distinguish your
measurements from supplied traces. Negative results are valid evidence.

## 1. Architecture and causal checks

TODO: Explain one numerical attention calculation, the prefix and future-gradient
checks, and why the supplied unmasked control should fail. State what these
checks do and do not establish.

## 2. Training and normalization

TODO: Interpret the fitting floor, baseline curve, and restart evidence. Compare
pre/post-norm using matched initial weights, batches, and token budgets. Include
the curve or a relative link to your submitted numeric evidence. Discuss one
failure mode and the limits of a single seed.

## 3. Decoding

TODO: Compare the nine fixed-prompt generations. Explain temperature, nucleus
selection, EOS/truncation, and repetition. Separate model quality from sampling.

## 4. Data policy

TODO: Inspect at least three retained and three removed document IDs, including
a case where repetition might be useful. Report retained corpus tokens, training
tokens processed, repeated exposure, and dev/final loss differences. Explain the
split controls, synthetic duplicate construction, and any negative result.

## AI-use disclosure

TODO: Name assistants/tools used, how they helped, and what you independently
checked. If none, explicitly state no AI assistance. No chat transcript required.
