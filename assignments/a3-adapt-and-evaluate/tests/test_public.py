"""Small public examples. Passing these does not establish full correctness."""
import math
import pytest
import torch
import a3


def test_response_eos_and_padding():
    ids = torch.tensor([[1, 2, 3, 9, 9]])
    assert a3.response_labels(ids, torch.tensor([[1,1,1,1,0]]), torch.tensor([2])).tolist() == [[-100,-100,3,9,-100]]


def test_uniform_response_loss():
    assert a3.sft_loss(torch.zeros(1,4,8), torch.tensor([[-100,-100,2,3]])).item() == pytest.approx(math.log(8))


def test_reward_pair():
    assert a3.preference_loss(torch.tensor([math.log(3)]), torch.tensor([0.])).item() == pytest.approx(math.log(4/3))


def test_one_step_gae():
    advantage, target = a3.gae(torch.tensor([[2.]]), torch.tensor([[.5]]), torch.ones(1,1))
    assert advantage.item() == 1.5 and target.item() == 2.


def test_positive_advantage_clipping():
    loss, _ = a3.ppo_loss(torch.tensor([[math.log(1.5)]]), torch.zeros(1,1), torch.tensor([[2.]]),
                         torch.zeros(1,1), torch.zeros(1,1), torch.ones(1,1))
    assert loss.item() == pytest.approx(-2.4)
