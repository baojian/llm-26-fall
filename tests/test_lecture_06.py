"""Independent checks for the supplied Lecture 06 CPU experiments."""
import json
import math
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "slides/lecture-06/assets"
sys.path.insert(0, str(ASSETS))
from toy_lm import DecoderLM, NOA_DOCUMENTS, VOCABULARY
from training import RunConfig, ToyRun, distribution, encode, evaluate, generate, lr_at, shifted
from resume_demo import run_comparison
sys.path.pop(0)


def test_model_matches_lecture_05_and_causal_prefix():
    torch.manual_seed(7)
    model = DecoderLM().double()
    assert sum(p.numel() for p in model.parameters()) == 4608
    assert model.lm_head.weight is model.token_embedding.weight
    first = torch.tensor([NOA_DOCUMENTS[0][:-1]])
    altered = first.clone()
    altered[:, 6:] = 10
    with torch.no_grad():
        a, b = model(first), model(altered)
    torch.testing.assert_close(a[:, :6], b[:, :6], atol=1e-12, rtol=0)
    assert not torch.allclose(a[:, 6:], b[:, 6:])


def test_corpus_has_fixed_distinct_documents_and_valid_lengths():
    corpus = json.loads((ASSETS / "corpus.json").read_text(encoding="utf-8"))
    assert len(corpus["train"]) == 16 and len(corpus["dev"]) == 8
    assert len(set(corpus["train"] + corpus["dev"])) == 24
    for text in corpus["train"] + corpus["dev"]:
        document = encode(text)
        x, y = shifted(document)
        assert x.shape == y.shape == (1, len(document) - 1)
        assert y[0, -1] == 11
        assert torch.equal(x[:, 1:], y[:, :-1])


def test_weighted_evaluation_and_mode_restoration():
    class FixedLogits(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([0., 1., 2., 3.]).expand(*x.shape, 4)
    model = FixedLogits()
    result = evaluate(model, [[0, 1, 2], [0, 3]])
    normalizer = math.log(sum(math.exp(i) for i in range(4)))
    assert result["tokens"] == 3
    assert result["nll"] == pytest.approx(normalizer - (1+2+3)/3)
    assert model.training
    model.eval()
    evaluate(model, [[0, 1]])
    assert not model.training
    with pytest.raises(ValueError):
        evaluate(model, [[0]])
    assert not model.training


def test_learning_rate_boundaries_and_monotone_segments():
    rates = [lr_at(s, 8, 2, .001, .1) for s in range(8)]
    assert [rates[0], rates[1], rates[-1]] == pytest.approx([.0005, .001, .0001])
    assert all(a > b for a, b in zip(rates[1:], rates[2:]))
    for kwargs in [dict(update=8, total=8), dict(update=-1), dict(update=0,warmup=40),
                   dict(update=0, maximum=float("nan"))]:
        with pytest.raises(ValueError):
            lr_at(**kwargs)


def test_optimizer_step_clips_and_reports_unclipped_norm():
    run = ToyRun(RunConfig(clip=.01))
    original = [p.detach().clone() for p in run.model.parameters()]
    record = run.advance()
    actual = torch.sqrt(sum(p.grad.square().sum() for p in run.model.parameters())).item()
    assert record["grad_norm"] > .01
    assert actual <= .010001
    assert record["completed"] == 1 and record["tokens"] == 10
    assert any(not torch.equal(a,b) for a,b in zip(original,run.model.parameters()))


def test_gradient_accumulation_matches_equal_full_batch():
    torch.manual_seed(2)
    model = torch.nn.Linear(3, 2, bias=False).double()
    x, y = torch.randn(4, 3, dtype=torch.float64), torch.tensor([0,1,1,0])
    torch.nn.functional.cross_entropy(model(x), y).backward()
    full = model.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    for begin in [0,2]:
        (torch.nn.functional.cross_entropy(model(x[begin:begin+2]), y[begin:begin+2])/2).backward()
    torch.testing.assert_close(model.weight.grad, full, atol=1e-12, rtol=0)


def test_checkpoint_identity_rejected_before_loading(tmp_path):
    run = ToyRun()
    run.advance()
    path = tmp_path / "checkpoint.pt"
    run.save(path)
    other = ToyRun(RunConfig(maximum_lr=.01))
    initial = other.model.token_embedding.weight.detach().clone()
    with pytest.raises(ValueError, match="differs"):
        other.restore(path)
    assert torch.equal(initial, other.model.token_embedding.weight)
    assert other.completed == 0


def test_fresh_process_restart_and_missing_optimizer_control(tmp_path):
    report = run_comparison(tmp_path / "restart")
    good, control = (report["comparisons"][k] for k in ["resumed", "weights_only"])
    assert good["same_batches"] and good["same_lrs"] and good["within_tolerance"]
    assert good["next_loss_difference"] <= 1e-7
    assert control["same_batches"] and control["same_lrs"]
    # The next forward pass precedes the first changed optimizer update.
    assert control["next_loss_difference"] <= 1e-7
    assert control["second_loss_difference"] > 1e-7
    assert not control["within_tolerance"]


def test_nucleus_crossing_token_ties_and_all_tokens():
    logits = torch.tensor([.4,.3,.2,.1], dtype=torch.float64).log()
    torch.testing.assert_close(distribution(logits, top_p=.6),
                               torch.tensor([4/7,3/7,0,0], dtype=torch.float64))
    torch.testing.assert_close(distribution(logits, top_p=1), logits.exp())
    torch.testing.assert_close(distribution(torch.zeros(4), top_p=.5),
                               torch.tensor([.5,.5,0,0], dtype=torch.float64))
    assert distribution(logits, top_p=.01).tolist() == [1,0,0,0]
    assert distribution(logits, temperature=.3).argmax() == logits.argmax()


@pytest.mark.parametrize("temperature,top_p", [(0,1),(-1,1),(float("nan"),1),(1,0),(1,1.1),(1,float("nan"))])
def test_invalid_sampling_parameters(temperature, top_p):
    with pytest.raises(ValueError):
        distribution(torch.zeros(4), temperature, top_p)


def test_generation_eos_budget_crop_ties_and_mode():
    class Stub(torch.nn.Module):
        max_length = 3
        def __init__(self, eos):
            super().__init__()
            self.seen = []
            self.eos = eos
        def forward(self, x):
            self.seen.append(x.tolist()[0])
            result = torch.zeros((*x.shape,12))
            if self.eos:
                result[...,11] = 20
            return result
    model = Stub(True)
    assert generate(model, [0,1,2,3],mode="greedy")["stop"] == "eos"
    assert model.seen == [[1,2,3]] and model.training
    assert generate(model,[0],budget=0) == {"ids":[],"stop":"length"}
    model = Stub(False).eval()
    result = generate(model,[1],mode="greedy",budget=2)
    assert result == {"ids":[0,0],"stop":"length"}
    assert not model.training
    a = generate(model,[1],seed=4,budget=5)
    b = generate(model,[1],seed=4,budget=5)
    assert a == b


def test_two_document_analytic_loss_floor():
    # Count next-token outcomes for each exact prefix without using the model.
    from collections import Counter, defaultdict
    counts = defaultdict(Counter)
    for document in NOA_DOCUMENTS:
        for end in range(1,len(document)):
            counts[tuple(document[:end])][document[end]] += 1
    entropy_sum = sum(-n * math.log(n/sum(c.values())) for c in counts.values() for n in c.values())
    assert entropy_sum/20 == pytest.approx(math.log(2)/10)
