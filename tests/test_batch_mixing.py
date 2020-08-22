import pytest
import torch
from torch import nn as nn

from verification.batch_gradient_mixing import BatchMixingVerification


class MixingModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        # x: (B, 5, 2)
        x = x.view(10, -1).permute(1, 0).view(-1, 10)  # oops!
        return self.linear(x)


class NonMixingModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        # x: (B, 5, 2)
        x = x.view(-1, 10)  # good!
        return self.linear(x)


@pytest.mark.parametrize(["model", "is_valid"], [
    pytest.param(MixingModel(), False),
    pytest.param(NonMixingModel(), True)
])
def test_mixing_model(model, is_valid):
    verification = BatchMixingVerification(model)
    result = verification.check(input_array=torch.rand(10, 5, 2))
    assert result == is_valid


