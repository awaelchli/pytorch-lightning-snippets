import pytest
import torch
from torch import nn as nn

from pytorch_lightning import Trainer, LightningModule
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from verification.batch_norm import BatchNormVerification, BatchNormVerificationCallback


class ConvBiasBatchNormModel(nn.Module):
    def __init__(self, use_bias=True):
        super().__init__()
        self.conv = nn.Conv2d(3, 5, kernel_size=1, bias=use_bias)
        self.bn = nn.BatchNorm2d(5)
        self.input_array = torch.rand(2, 3, 10, 10)

    def forward(self, x):
        # x: (B, 3, H, W)
        return self.bn(self.conv(x))


class LitModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = ConvBiasBatchNormModel(*args, **kwargs)
        self.example_input_array = self.model.input_array

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.rand(32, 3, 10, 10)), batch_size=4)

    def configure_optimizers(self):
        return Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        output = self(batch[0])
        loss = output.sum()
        return loss


@pytest.mark.parametrize(["use_bias"], [pytest.param(True), pytest.param(False)])
@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda", 0)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
def test_conv_bias_batch_norm_model(use_bias, device):
    model = ConvBiasBatchNormModel(use_bias=use_bias).to(device)
    verification = BatchNormVerification(model)
    expected = not use_bias
    result = verification.check(input_array=model.input_array)
    assert result == expected


@pytest.mark.parametrize("gpus", [0, 1])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
def test_conv_bias_batch_norm_callback(gpus):
    model = LitModel(use_bias=True)
    expected = "'model.conv' with bias followed by a normalization layer 'model.bn'"

    callback = BatchNormVerificationCallback()
    trainer = Trainer(gpus=gpus, callbacks=[callback], max_steps=1)
    with pytest.warns(UserWarning, match=expected):
        trainer.fit(model)

    callback = BatchNormVerificationCallback(error=True)
    trainer = Trainer(gpus=gpus, callbacks=[callback], max_steps=1)
    with pytest.raises(RuntimeError, match=expected):
        trainer.fit(model)
