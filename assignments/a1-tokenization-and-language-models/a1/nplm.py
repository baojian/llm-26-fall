"""A1 Part 3: a neural n-gram language model with embeddings (Bengio et al., 2003) in PyTorch.

The model predicts the next token id from the last ``context`` ids:
    e = concat(E[x_{t-3}], E[x_{t-2}], E[x_{t-1}])     embedding lookup
    h = tanh(W e + b)                                   hidden layer
    logits = U h + c                                    unembedding
    loss = cross-entropy(logits, x_t)

Training on the BPE ids of ``tinystories-train.txt`` takes a few minutes on a
laptop CPU with the default sizes. torch only; no other packages.
"""

from __future__ import annotations

import math
import random

import torch
from torch import nn

BOS = -1  # mapped to the extra id vocab_size inside the model
EOS = -2  # mapped to vocab_size + 1


class NeuralNGramLM(nn.Module):
    def __init__(self, vocab_size: int, context: int = 3, dim: int = 64, hidden: int = 64,
                 tie_output: bool = False):
        """``tie_output=True`` shares the output projection's weight with the embedding table
        (requires ``hidden == dim``), the weight tying used by many LLMs; it removes
        n_ids * dim parameters. The bias of the output layer stays separate."""
        super().__init__()
        self.vocab_size = vocab_size
        self.context = context
        self.n_ids = vocab_size + 2  # ids 0..vocab_size-1, then BOS, then EOS
        self.tie_output = tie_output
        if tie_output and hidden != dim:
            raise ValueError("weight tying needs hidden == dim")
        # YOUR CODE HERE
        raise NotImplementedError

    def map_id(self, token: int) -> int:
        """Map a token id, BOS, or EOS to a row of the embedding table."""
        if token == BOS:
            return self.vocab_size
        if token == EOS:
            return self.vocab_size + 1
        return token

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        """contexts: (batch, context) long tensor of mapped ids -> (batch, n_ids) logits."""
        # YOUR CODE HERE
        raise NotImplementedError


def make_examples(sequences: list[list[int]], model: NeuralNGramLM) -> tuple[torch.Tensor, torch.Tensor]:
    """Every (context window, next token) pair from padded sequences, as mapped ids."""
    # YOUR CODE HERE
    raise NotImplementedError


def train(model: NeuralNGramLM, sequences: list[list[int]], dev_sequences: list[list[int]],
          epochs: int = 3, batch_size: int = 256, lr: float = 3e-3, seed: int = 0) -> list[dict]:
    """Minibatch Adam on cross-entropy; returns one dict per epoch with train and dev loss (nats per token)."""
    torch.manual_seed(seed)
    random.seed(seed)
    history: list[dict] = []
    # YOUR CODE HERE
    raise NotImplementedError
    return history


def log2_prob(model: NeuralNGramLM, sequences: list[list[int]], batch_size: int = 1024) -> tuple[float, int]:
    """Total log2 probability (with EOS), evaluated in bounded batches to limit memory."""
    # YOUR CODE HERE
    raise NotImplementedError


def parameter_count(model: nn.Module) -> int:
    """Number of distinct trainable parameters (shared weights are counted once)."""
    return sum(p.numel() for p in model.parameters())


def nearest_neighbours(model: NeuralNGramLM, token: int, k: int = 5) -> list[tuple[int, float]]:
    """The k token ids whose embeddings have the highest cosine similarity to ``token``'s (excluding itself)."""
    # YOUR CODE HERE
    raise NotImplementedError
