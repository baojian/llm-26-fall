"""Small public examples. Passing these does not establish full correctness."""
import math
import pytest
import torch
import a3


def test_response_eos_and_padding():
    ids = torch.tensor([[1, 2, 3, 9, 9]])
    labels = a3.response_labels(ids, torch.tensor([[1, 1, 1, 1, 0]]), torch.tensor([2]))
    assert labels.tolist() == [[-100, -100, 3, 9, -100]]
    assert ids.tolist() == [[1, 2, 3, 9, 9]]


def test_nonuniform_sft_shift_and_token_mean():
    # Three targets: two in the first row and one in the second. Only their
    # preceding logit positions count, with equal weight per target token.
    probabilities = torch.tensor([
        [[.1, .3, .6], [.7, .2, .1], [.2, .3, .5], [.2, .7, .1]],
        [[.2, .2, .6], [.3, .6, .1], [.5, .2, .3], [.1, .1, .8]],
    ])
    logits = probabilities.log().requires_grad_()
    labels = torch.tensor([[-100, 2, 1, -100], [-100, -100, -100, 0]])
    loss = a3.sft_loss(logits, labels)
    assert loss.item() == pytest.approx(-(math.log(.6) + math.log(.2) + math.log(.5)) / 3)
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, 2:]) == 0
    assert torch.count_nonzero(logits.grad[1, [0, 1, 3]]) == 0
    assert logits.grad[0, 0, 2] < 0 and logits.grad[1, 2, 0] < 0


def test_terminal_pooling_excludes_padding():
    scores = torch.tensor([[9., 2., 99.], [4., 6., 8.]], requires_grad=True)
    pooled = a3.terminal_scores(scores, torch.tensor([[1, 1, 0], [1, 1, 1]]))
    assert pooled.shape == (2,) and pooled.tolist() == [2., 8.]
    pooled.sum().backward()
    assert scores.grad.tolist() == [[0., 1., 0.], [0., 0., 1.]]


def test_reward_pair():
    assert a3.preference_loss(torch.tensor([math.log(3)]), torch.tensor([0.])).item() == pytest.approx(math.log(4/3))


def test_shaping_puts_score_at_terminal_and_detaches():
    old = torch.tensor([[-.4, 20.], [.2, -.6]], requires_grad=True)
    reference = torch.zeros(2, 2, requires_grad=True)
    scores = torch.tensor([2., 1.], requires_grad=True)
    rewards = a3.shaped_rewards(old, reference, scores, torch.tensor([[1., 0.], [1., 1.]]), beta=.25)
    # A negative sampled log ratio gives a positive shaping term.
    assert torch.allclose(rewards, torch.tensor([[2.1, 0.], [-.05, 1.15]]))
    assert not rewards.requires_grad


def test_multistep_gae_respects_terminal_and_padding():
    rewards = torch.tensor([[1., 2., 90.], [3., 80., 70.]], requires_grad=True)
    values = torch.tensor([[.2, .4, 60.], [.5, 50., 40.]], requires_grad=True)
    mask = torch.tensor([[1., 1., 0.], [1., 0., 0.]])
    advantages, targets = a3.gae(rewards, values, mask, gamma=.5, lam=.8)
    # First row: A_1 = 2-.4; A_0 = (1+.5*.4-.2) + .5*.8*A_1.
    assert torch.allclose(advantages, torch.tensor([[1.64, 1.6, 0.], [2.5, 0., 0.]]))
    assert torch.allclose(targets, torch.tensor([[1.84, 2., 0.], [3., 0., 0.]]))
    assert not advantages.requires_grad and not targets.requires_grad


def test_positive_advantage_clipping():
    loss, _ = a3.ppo_loss(torch.tensor([[math.log(1.5)]]), torch.zeros(1,1), torch.tensor([[2.]]),
                         torch.zeros(1,1), torch.zeros(1,1), torch.ones(1,1))
    assert loss.item() == pytest.approx(-2.4)


def test_negative_clipping_fixed_targets_and_diagnostics():
    new = torch.tensor([[math.log(.5), math.log(1.1), math.log(10.)]], requires_grad=True)
    old = torch.zeros(1, 3, requires_grad=True)
    advantages = torch.tensor([[-2., 1., 99.]], requires_grad=True)
    values = torch.tensor([[1., 2., 99.]], requires_grad=True)
    returns = torch.zeros(1, 3, requires_grad=True)
    loss, info = a3.ppo_loss(new, old, advantages, values, returns,
                             torch.tensor([[1., 1., 0.]]), value_coef=.4)
    # Policy = -(-1.6+1.1)/2; value = .5*(1+4)/2; total = .25+.4*1.25.
    assert loss.item() == pytest.approx(.75)
    assert set(info) == {"policy_loss", "value_loss", "clip_fraction"}
    assert all(type(value) is float and math.isfinite(value) for value in info.values())
    assert info == pytest.approx({"policy_loss": .25, "value_loss": 1.25, "clip_fraction": .5})
    loss.backward()
    assert old.grad is None and advantages.grad is None and returns.grad is None
    assert torch.allclose(new.grad, torch.tensor([[0., -.55, 0.]]))
    assert torch.allclose(values.grad, torch.tensor([[.2, .4, 0.]]))


def test_explicit_empty_tensor_errors():
    with pytest.raises(ValueError):
        a3.sft_loss(torch.zeros(1, 2, 3), torch.tensor([[1, -100]]))
    with pytest.raises(ValueError):
        a3.terminal_scores(torch.zeros(1, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError):
        a3.shaped_rewards(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1), torch.zeros(1, 2), .1)
    with pytest.raises(ValueError):
        a3.ppo_loss(*(torch.zeros(1, 2) for _ in range(6)))


def test_paired_family_weighting_bootstrap_and_schema():
    before = [
        {"id": "a", "family": "zeta", "target": "red", "prediction": " RED "},
        {"id": "b", "family": "zeta", "target": "blue", "prediction": "blue."},
        {"id": "c", "family": "alpha", "target": "green", "prediction": "wrong"},
        {"id": "d", "family": "mu", "target": "yellow", "prediction": "wrong"},
    ]
    after = [dict(row) for row in reversed(before)]
    after[1]["prediction"] = "green"  # c improves; a regresses.
    after[3]["prediction"] = "wrong"
    result = a3.paired_summary(before, after, seed=11, bootstrap_samples=9)
    assert set(result) == {"families", "examples", "before_accuracy", "after_accuracy",
                           "delta", "improved", "regressed", "ci95"}
    for key, expected in {"families": 3, "examples": 4, "improved": 1, "regressed": 1}.items():
        assert type(result[key]) is int and result[key] == expected
    for key, expected in {"before_accuracy": 1/6, "after_accuracy": 1/3, "delta": 1/6}.items():
        assert type(result[key]) is float and math.isfinite(result[key])
        assert result[key] == pytest.approx(expected)
    # Sorted family differences are [1, 0, -.5]. Nine random.Random(11).choices
    # draws give order-statistic endpoints -1/6 and 2/3 (no interpolation).
    assert type(result["ci95"]) is list and len(result["ci95"]) == 2
    assert all(type(value) is float and math.isfinite(value) for value in result["ci95"])
    assert result["ci95"] == pytest.approx([-1/6, 2/3])
    assert result == a3.paired_summary(list(reversed(before)), list(reversed(after)), 11, 9)


def test_paired_summary_rejects_invalid_comparisons():
    row = {"id": "a", "family": "f", "target": "red", "prediction": "red"}
    for after in ([], [row, row], [dict(row, id="b")],
                  [dict(row, family="other")], [dict(row, target=" RED ")]):
        with pytest.raises(ValueError):
            a3.paired_summary([row], after)
    with pytest.raises(ValueError):
        a3.paired_summary([row], [row], bootstrap_samples=1)
