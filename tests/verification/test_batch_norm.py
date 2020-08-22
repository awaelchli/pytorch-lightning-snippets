import torch
from torch import nn as nn

from verification.batch_norm import BatchNormVerification


class ConvBiasBatchNormModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 5, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(5)
        self.example_input_array = torch.rand(2, 3, 10, 10)

    def forward(self, x):
        # x: (B, 3, H, W)
        return self.bn(self.conv(x))


def test_conv_bias_batch_norm_model():
    model = ConvBiasBatchNormModel()
    print(model.conv.bias)
    verification = BatchNormVerification(model)
    verification.check()