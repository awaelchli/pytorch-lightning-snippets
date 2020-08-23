import pytest
import torch
from torch import nn as nn

from verification.batch_gradient_mixing import BatchMixingVerification, default_output_mapping, default_input_mapping


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


def test_default_input_mapping():
    b = 3
    tensor0 = torch.rand(b, 2, 5)
    tensor1 = torch.rand(b, 9)
    tensor2 = torch.rand(b, 5, 1)

    # Tensor
    data = tensor0.double()
    output = default_input_mapping(data)
    assert torch.all(output == data)

    # tuple
    data = ("foo", tensor1, tensor2, [])
    output = default_input_mapping(data)
    assert output is tensor1

    # dict + nesting
    data = {
        "one": ["foo", tensor2],
        "two": tensor0,
    }
    output = default_input_mapping(data)
    assert output is tensor2


def test_default_output_mapping():
    b = 3
    tensor0 = torch.rand(b, 2, 5)
    tensor1 = torch.rand(b, 9)
    tensor2 = torch.rand(b, 5, 1)
    tensor3 = torch.rand(b)
    scalar = torch.tensor(3.14)

    # Tensor
    data = tensor0.double()
    output = default_output_mapping(data)
    assert output is data

    # tuple + nesting
    data = (tensor0, None, tensor1, "foo", [tensor2])
    expected = torch.cat((tensor0.view(b, -1), tensor1.view(b, -1), tensor2.view(b, -1)), dim=1)
    output = default_output_mapping(data)
    assert torch.all(output == expected)

    # dict + nesting
    data = {
        "one": tensor1,
        "two": {"three": tensor3.double()},  # will convert to float
        "four": scalar,  # ignored
        "five": [tensor0, tensor0]
    }
    expected = torch.cat((tensor1.view(b, -1), tensor3.view(b, -1), tensor0.view(b, -1), tensor0.view(b, -1)), dim=1)
    output = default_output_mapping(data)
    assert torch.all(output == expected)


