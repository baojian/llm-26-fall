"""Public contract examples; incomplete starter functions should fail clearly."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from a2.model import ModelConfig, TinyLM, scaled_dot_product_attention
from a2.train import learning_rate, make_batch, train_step
from a2.generate import sampling_probs, generate
from a2.data_policy import deduplicate


def test_attention_uniform_allowed_keys():
    q = torch.zeros(1, 2, 3)
    v = torch.tensor([[[2.0], [8.0]]])
    actual = scaled_dot_product_attention(q, q, v, torch.ones(2, 2, dtype=torch.bool).tril())
    torch.testing.assert_close(actual, torch.tensor([[[2.0], [5.0]]]))


def test_complete_decoder_causality():
    torch.manual_seed(7)
    model = TinyLM(ModelConfig(vocab_size=9, context=4, width=8, heads=2, layers=1, ff_width=16)).eval()
    a, b = torch.tensor([[1, 2, 3, 4]]), torch.tensor([[1, 2, 7, 8]])
    assert model(a).shape == (1, 4, 9)
    torch.testing.assert_close(model(a)[:, :2], model(b)[:, :2])


def test_targets_and_schedule_endpoints():
    x, y = make_batch(torch.tensor([[0, 2, 5]]))
    assert x.tolist() == [[0, 2]] and y.tolist() == [[2, 5]]
    assert learning_rate(0, 8, 2, 0.01) == pytest.approx(0.005)
    assert learning_rate(1, 8, 2, 0.01) == pytest.approx(0.01)
    assert learning_rate(7, 8, 2, 0.01) == pytest.approx(0.001)


def test_training_independent_of_part1():
    model = nn.Embedding(6, 6)
    before = model.weight.detach().clone()
    metrics = train_step(model, torch.optim.AdamW(model.parameters(), lr=0.1), torch.tensor([[1, 2, 3]]), 1.0)
    assert metrics["loss"] > 0 and metrics["grad_norm"] > 0
    assert not torch.equal(before, model.weight)


def test_nucleus_crossing_token():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.2]))
    torch.testing.assert_close(sampling_probs(logits, 1, 0.7), torch.tensor([0.625, 0.375, 0.0]))


def test_dedup_keeps_one_smallest_id():
    docs = [{"id": "b", "text": " story "}, {"id": "a", "text": "story"}, {"id": "c", "text": "Story"}]
    kept, audit = deduplicate(docs)
    assert [row["id"] for row in kept] == ["a", "c"] and audit == {"b": "a"}
