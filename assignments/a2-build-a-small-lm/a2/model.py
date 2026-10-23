"""Part 1: a causal decoder. Constructors and initialization are supplied.

All linear layers are bias-free. LayerNorm uses affine parameters and eps=1e-5.
No dropout; learned positions; no embedding scaling; tied vocabulary readout.
"""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 4096
    context: int = 128
    width: int = 128
    heads: int = 4
    layers: int = 2
    ff_width: int = 512
    norm: str = "pre"

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (
            self.vocab_size, self.context, self.width, self.heads, self.layers, self.ff_width
        )):
            raise ValueError("model dimensions must be positive integers")
        if self.width % self.heads or self.norm not in ("pre", "post"):
            raise ValueError("width must divide into heads; norm must be pre or post")


def scaled_dot_product_attention(q, k, v, allowed_mask):
    """q (...,Q,D), k (...,K,D), v (...,K,Dv) -> (...,Q,Dv).

    Boolean True means allowed; mask broadcasts to (...,Q,K). Every row must
    permit a key. Scale by sqrt(D), mask BEFORE softmax, normalize over keys.
    """
    if allowed_mask.dtype != torch.bool or not allowed_mask.any(dim=-1).all():
        raise ValueError("use a Boolean mask with at least one allowed key per row")
    # YOUR CODE HERE
    raise NotImplementedError


class CausalSelfAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("positive width must be divisible by positive heads")
        self.width, self.heads = width, heads
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.out_proj = nn.Linear(width, width, bias=False)

    def forward(self, x):
        """(B,T,width) -> (B,T,width); split heads without exchanging time/head axes."""
        # YOUR CODE HERE
        raise NotImplementedError


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_order = config.norm
        self.attention = CausalSelfAttention(config.width, config.heads)
        self.ln1 = nn.LayerNorm(config.width, eps=1e-5)
        self.ln2 = nn.LayerNorm(config.width, eps=1e-5)
        self.ff = nn.Sequential(nn.Linear(config.width, config.ff_width, bias=False),
                                nn.ReLU(), nn.Linear(config.ff_width, config.width, bias=False))

    def forward(self, x):
        """Pre: x+A(LN(x)), then x+FF(LN(x)); post: LN(x+A(x)), then LN(x+FF(x))."""
        # YOUR CODE HERE
        raise NotImplementedError


class TinyLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.layers)])
        self.final_norm = nn.LayerNorm(config.width, eps=1e-5)
        self.readout = nn.Linear(config.width, config.vocab_size, bias=False)
        self.readout.weight = self.token.weight
        # Initialize each shared parameter only once.
        with torch.no_grad():
            for parameter in self.parameters():
                if parameter.ndim >= 2:
                    nn.init.normal_(parameter, mean=0.0, std=0.02)

    def forward(self, ids):
        """Integer (B,T) -> raw (B,T,V) logits. Require 1 <= T <= context."""
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.config.context:
            raise ValueError("ids must have shape (batch, time) with time in 1..context")
        # YOUR CODE HERE
        raise NotImplementedError
