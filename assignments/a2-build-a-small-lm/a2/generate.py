"""Part 3: transform a fixed model's distribution and generate tokens."""
import math

import torch


def sampling_probs(logits, temperature=1.0, top_p=1.0):
    """Finite 1D logits -> normalized probabilities in original token order.

    Require temperature>0 and 0<top_p<=1. Break probability ties by smaller
    token ID. Keep the smallest sorted prefix reaching top_p, including its
    crossing token. At top_p=1 keep every token.
    """
    if (logits.ndim != 1 or not torch.isfinite(logits).all() or
            not math.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1):
        raise ValueError("finite vector logits, positive temperature, and top_p in (0,1] required")
    # YOUR CODE HERE
    raise NotImplementedError


def choose_token(logits, mode, temperature, top_p, generator):
    """Return a Python int. Greedy chooses the lowest-ID maximum; sample uses a CPU generator."""
    if mode not in ("greedy", "sample"):
        raise ValueError("mode must be greedy or sample")
    # YOUR CODE HERE
    raise NotImplementedError


@torch.no_grad()
def generate(model, prompt, max_new_tokens, eos_id, *, mode="sample", temperature=1.0,
             top_p=1.0, generator=None):
    """Return {'tokens': new IDs only, 'stop': 'eos'|'length'}.

    Supplied context management: crop to the last config.context tokens,
    renumber learned positions from zero on each call. Prompt must be nonempty.
    The prompt is context, even if it ends with EOS. Include a newly sampled EOS
    in returned tokens. Zero max_new_tokens returns [] and 'length'. Restore mode.
    """
    if not prompt or max_new_tokens < 0:
        raise ValueError("nonempty prompt and nonnegative generation cap required")
    was_training = model.training
    model.eval()
    tokens = list(prompt)
    output = []
    stop = "length"
    device = next(model.parameters()).device
    try:
        for _ in range(max_new_tokens):
            window = torch.tensor([tokens[-model.config.context:]], dtype=torch.long, device=device)
            logits = model(window)[0, -1]
            # YOUR CODE HERE
            raise NotImplementedError
    finally:
        model.train(was_training)
    return {"tokens": output, "stop": stop}
