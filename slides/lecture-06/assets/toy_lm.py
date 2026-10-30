"""Lecture 06 CPU toy model, adapted from Lecture 05's Noa notebook.

This is supplied ungraded teaching code. It has its own data and interfaces;
it is not an A2 reference implementation. Run on CPU float32.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

VOCABULARY = "Noa can be annoying but she is a great cat friend EOS".split()
NOA_DOCUMENTS = [list(range(10)) + [11], list(range(9)) + [10, 11]]
V, max_length, d, n_heads, d_ff, n_layers = 12, 10, 16, 4, 32, 2

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def split_heads(x, n_heads):
    batch, length, width = x.shape
    if width % n_heads:
        raise ValueError('model width must be divisible by the number of heads')
    return x.reshape(batch, length, n_heads, width // n_heads).transpose(1, 2)


def merge_heads(x):
    batch, heads, length, head_width = x.shape
    return x.transpose(1, 2).reshape(batch, length, heads * head_width)


class MultiHeadAttention(nn.Module):
    def __init__(self, width, n_heads):
        super().__init__()
        if width % n_heads:
            raise ValueError('model width must be divisible by the number of heads')
        self.n_heads = n_heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, x, causal=True):
        q, k, v = (split_heads(layer(x), self.n_heads)
                   for layer in (self.query, self.key, self.value))
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        if causal:
            allowed = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(~allowed, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        return self.output(merge_heads(weights @ v)), weights


class FeedForward(nn.Module):
    def __init__(self, width, hidden_width):
        super().__init__()
        self.up = nn.Linear(width, hidden_width, bias=False)
        self.down = nn.Linear(hidden_width, width, bias=False)

    def forward(self, x):
        return self.down(torch.relu(self.up(x)))


class Block(nn.Module):
    def __init__(self, width, n_heads, hidden_width, norm_first=True):
        super().__init__()
        self.norm_first = norm_first
        self.attention_norm = nn.LayerNorm(width, eps=1e-5)
        self.attention = MultiHeadAttention(width, n_heads)
        self.ffn_norm = nn.LayerNorm(width, eps=1e-5)
        self.ffn = FeedForward(width, hidden_width)

    def forward(self, x, causal=True):
        if self.norm_first:  # pre-LN: x + F(LN(x))
            x = x + self.attention(self.attention_norm(x), causal)[0]
            return x + self.ffn(self.ffn_norm(x))
        x = self.attention_norm(x + self.attention(x, causal)[0])  # post-LN: LN(x + F(x))
        return self.ffn_norm(x + self.ffn(x))


class DecoderLM(nn.Module):
    def __init__(self, vocab_size=V, max_length=max_length, width=d, n_heads=n_heads,
                 hidden_width=d_ff, n_layers=n_layers, norm_first=True, tie_weights=True):
        super().__init__()
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(max_length, width)
        self.blocks = nn.ModuleList(Block(width, n_heads, hidden_width, norm_first)
                                    for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(width, eps=1e-5)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.apply(init_weights)
        if tie_weights:  # the readout reuses the same Parameter object
            self.lm_head.weight = self.token_embedding.weight

    def embed(self, ids):
        if ids.ndim != 2 or ids.dtype != torch.long or ids.shape[0] == 0:
            raise ValueError('ids must be a (batch, length) tensor of token IDs')
        if not 1 <= ids.shape[1] <= self.max_length:
            raise ValueError(f'sequence length must be between 1 and {self.max_length}')
        if ids.min() < 0 or ids.max() >= self.token_embedding.num_embeddings:
            raise ValueError('token ID outside the vocabulary')
        positions = torch.arange(ids.shape[1], device=ids.device)
        return self.token_embedding(ids) + self.position_embedding(positions)

    def forward_hidden(self, x, causal=True):
        for block in self.blocks:
            x = block(x, causal)
        return self.final_norm(x)

    def forward(self, ids, causal=True):
        return self.lm_head(self.forward_hidden(self.embed(ids), causal))
